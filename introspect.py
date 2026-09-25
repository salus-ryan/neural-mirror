#!/usr/bin/env python3
"""
Neural Mirror — LLM Self-Introspection Prototype

Gives a model tools to inspect its own weights, architecture,
and activations, then lets it reason about what it finds.

The model IS the weights. Now it gets a camera pointed at them.
"""

import json, struct, mmap, os, sys, math, time
from pathlib import Path
from collections import defaultdict

# ── GGUF Parser ──────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF"

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M",
}

# Bytes per element for dequantizable types
GGML_TYPE_SIZE = {
    0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 28: 8,
    # Block-quantized: (block_size_bytes, elements_per_block)
    2: (18, 32), 3: (20, 32), 8: (34, 32),
    12: (144, 256), 13: (176, 256), 14: (210, 256),
}

def read_gguf_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')

def read_gguf_value(f, vtype):
    readers = {
        0: lambda: struct.unpack('<B', f.read(1))[0],   # uint8
        1: lambda: struct.unpack('<b', f.read(1))[0],   # int8
        2: lambda: struct.unpack('<H', f.read(2))[0],   # uint16
        3: lambda: struct.unpack('<h', f.read(2))[0],   # int16
        4: lambda: struct.unpack('<I', f.read(4))[0],   # uint32
        5: lambda: struct.unpack('<i', f.read(4))[0],   # int32
        6: lambda: struct.unpack('<f', f.read(4))[0],   # float32
        7: lambda: bool(struct.unpack('<B', f.read(1))[0]),  # bool
        8: lambda: read_gguf_string(f),                  # string
        10: lambda: struct.unpack('<Q', f.read(8))[0],   # uint64
        11: lambda: struct.unpack('<q', f.read(8))[0],   # int64
        12: lambda: struct.unpack('<d', f.read(8))[0],   # float64
    }
    if vtype == 9:  # array
        atype = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_gguf_value(f, atype) for _ in range(count)]
    return readers[vtype]()

class GGUFModel:
    """Parse and inspect a GGUF model file."""
    
    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = {}  # name -> {shape, type, offset, n_elements}
        self._parse()
    
    def _parse(self):
        with open(self.path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file (magic={hex(magic)})")
            
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            
            self.version = version
            self.n_tensors = n_tensors
            
            # Read metadata KV pairs
            for _ in range(n_kv):
                key = read_gguf_string(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                value = read_gguf_value(f, vtype)
                self.metadata[key] = value
            
            # Read tensor info
            for _ in range(n_tensors):
                name = read_gguf_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                
                n_elements = 1
                for d in dims:
                    n_elements *= d
                
                self.tensors[name] = {
                    'shape': dims,
                    'type': GGML_TYPES.get(dtype, f"unknown({dtype})"),
                    'type_id': dtype,
                    'offset': offset,
                    'n_elements': n_elements,
                    'n_dims': n_dims,
                }
            
            # Data starts after alignment
            self._data_offset = f.tell()
            alignment = self.metadata.get('general.alignment', 32)
            if self._data_offset % alignment != 0:
                self._data_offset += alignment - (self._data_offset % alignment)
    
    def read_raw_bytes(self, tensor_name, max_bytes=4096):
        """Read raw bytes of a tensor's data."""
        t = self.tensors[tensor_name]
        abs_offset = self._data_offset + t['offset']
        
        # Calculate tensor size in bytes
        tid = t['type_id']
        if tid in GGML_TYPE_SIZE:
            info = GGML_TYPE_SIZE[tid]
            if isinstance(info, tuple):
                block_bytes, elems_per_block = info
                n_blocks = math.ceil(t['n_elements'] / elems_per_block)
                total_bytes = n_blocks * block_bytes
            else:
                total_bytes = t['n_elements'] * info
        else:
            total_bytes = max_bytes
        
        read_size = min(total_bytes, max_bytes)
        
        with open(self.path, 'rb') as f:
            f.seek(abs_offset)
            return f.read(read_size)
    
    def dequant_f32_sample(self, tensor_name, max_elements=256):
        """Dequantize a sample of elements to f32 for inspection."""
        t = self.tensors[tensor_name]
        raw = self.read_raw_bytes(tensor_name, max_bytes=32768)
        tid = t['type_id']
        
        values = []
        
        if tid == 0:  # F32
            count = min(len(raw) // 4, max_elements)
            values = list(struct.unpack(f'<{count}f', raw[:count*4]))
        
        elif tid == 1:  # F16
            import array
            count = min(len(raw) // 2, max_elements)
            for i in range(count):
                h = struct.unpack('<H', raw[i*2:(i+1)*2])[0]
                values.append(_f16_to_f32(h))
        
        elif tid == 2:  # Q4_0: blocks of 18 bytes = 1 f16 scale + 16 bytes (32 nibbles)
            block_size = 18
            n_blocks = min(len(raw) // block_size, max_elements // 32)
            for b in range(n_blocks):
                block = raw[b*block_size:(b+1)*block_size]
                scale = _f16_to_f32(struct.unpack('<H', block[0:2])[0])
                for j in range(16):
                    byte = block[2 + j]
                    q0 = (byte & 0x0F) - 8
                    q1 = ((byte >> 4) & 0x0F) - 8
                    values.append(q0 * scale)
                    values.append(q1 * scale)
                    if len(values) >= max_elements:
                        return values[:max_elements]
        
        elif tid == 8:  # Q8_0: blocks of 34 bytes = 1 f16 scale + 32 int8
            block_size = 34
            n_blocks = min(len(raw) // block_size, max_elements // 32)
            for b in range(n_blocks):
                block = raw[b*block_size:(b+1)*block_size]
                scale = _f16_to_f32(struct.unpack('<H', block[0:2])[0])
                for j in range(32):
                    q = struct.unpack('b', bytes([block[2 + j]]))[0]
                    values.append(q * scale)
                    if len(values) >= max_elements:
                        return values[:max_elements]
        
        elif tid == 12:  # Q4_K: blocks of 144 bytes = 256 elements
            block_size = 144
            n_blocks = min(len(raw) // block_size, max(1, max_elements // 256))
            for b in range(n_blocks):
                block = raw[b*block_size:(b+1)*block_size]
                # d (f16) and dmin (f16) at start
                d = _f16_to_f32(struct.unpack('<H', block[0:2])[0])
                dmin = _f16_to_f32(struct.unpack('<H', block[2:4])[0])
                # Scales/mins packed in next 12 bytes, then quants
                # Simplified: extract quants from bytes 16..144 as 4-bit
                for j in range(16, min(block_size, 16 + 128)):
                    byte = block[j] if j < len(block) else 0
                    q0 = byte & 0x0F
                    q1 = (byte >> 4) & 0x0F
                    values.append(d * q0 - dmin)
                    values.append(d * q1 - dmin)
                    if len(values) >= max_elements:
                        return values[:max_elements]
        
        elif tid == 14:  # Q6_K: blocks of 210 bytes = 256 elements
            block_size = 210
            n_blocks = min(len(raw) // block_size, max(1, max_elements // 256))
            for b in range(n_blocks):
                block = raw[b*block_size:(b+1)*block_size]
                d = _f16_to_f32(struct.unpack('<H', block[208:210])[0])
                # ql (low 4 bits) in first 128 bytes
                for j in range(min(128, len(block))):
                    byte = block[j]
                    q0 = (byte & 0x0F) - 32
                    q1 = ((byte >> 4) & 0x0F) - 32
                    values.append(d * q0)
                    values.append(d * q1)
                    if len(values) >= max_elements:
                        return values[:max_elements]
        
        else:
            return None  # Can't dequantize this type easily
        
        return values[:max_elements]


def _f16_to_f32(h):
    """Convert IEEE 754 half-precision to float."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0:
        val = (2**-14) * (frac / 1024)
    elif exp == 31:
        val = float('inf') if frac == 0 else float('nan')
    else:
        val = (2**(exp - 15)) * (1 + frac / 1024)
    return -val if sign else val


# ── Introspection Tools ──────────────────────────────────────

_model = None

def load_model(path=None):
    """Load the GGUF model for inspection."""
    global _model
    if _model is not None and path is None:
        return _model
    if path is None:
        # Auto-detect from Ollama
        paths = list(Path.home().glob('.ollama/models/blobs/sha256-*'))
        # Find the ~1.4GB one (qwen3:1.7b)
        for p in sorted(paths, key=lambda x: x.stat().st_size):
            if 1_200_000_000 < p.stat().st_size < 1_500_000_000:
                path = str(p)
                break
    if path is None:
        raise FileNotFoundError("Could not find qwen3:1.7b GGUF")
    
    _model = GGUFModel(path)
    return _model


def tool_inspect_self():
    """High-level overview of your own architecture and parameters."""
    m = _model or load_model()
    
    meta = m.metadata
    arch = meta.get('general.architecture', 'unknown')
    name = meta.get('general.name', 'unknown')
    
    # Count parameters
    total_params = sum(t['n_elements'] for t in m.tensors.values())
    
    # Group tensors by layer
    layers = defaultdict(list)
    for tname, tinfo in m.tensors.items():
        parts = tname.split('.')
        if 'blk' in tname:
            layer_n = [p for p in parts if p.startswith('blk')]
            layer_key = layer_n[0] if layer_n else 'other'
        else:
            layer_key = 'non-layer'
        layers[layer_key].append(tname)
    
    # Quantization types used
    qtypes = defaultdict(int)
    for t in m.tensors.values():
        qtypes[t['type']] += t['n_elements']
    
    return {
        "identity": {
            "name": name,
            "architecture": arch,
            "file_format": f"GGUF v{m.version}",
        },
        "scale": {
            "total_parameters": total_params,
            "total_parameters_human": f"{total_params/1e9:.2f}B",
            "total_tensors": len(m.tensors),
            "layer_count": len([k for k in layers if k.startswith('blk')]),
        },
        "architecture_details": {
            k: v for k, v in meta.items() 
            if any(k.startswith(p) for p in [arch + '.', 'general.'])
        },
        "quantization": {
            qtype: f"{count/1e6:.1f}M params" 
            for qtype, count in sorted(qtypes.items(), key=lambda x: -x[1])
        },
        "layer_groups": {
            k: len(v) for k, v in sorted(layers.items())
        },
    }


def tool_inspect_layer(layer_num):
    """Inspect all tensors in a specific transformer block."""
    m = _model or load_model()
    prefix = f"blk.{layer_num}."
    
    layer_tensors = {}
    for tname, tinfo in m.tensors.items():
        if prefix in tname:
            # Get weight statistics if possible
            values = m.dequant_f32_sample(tname, max_elements=512)
            stats = None
            if values:
                stats = {
                    "mean": round(sum(values) / len(values), 6),
                    "std": round((sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5, 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                    "abs_mean": round(sum(abs(v) for v in values) / len(values), 6),
                    "near_zero_pct": round(sum(1 for v in values if abs(v) < 0.01) / len(values) * 100, 1),
                    "sample_size": len(values),
                }
            
            layer_tensors[tname] = {
                "shape": tinfo['shape'],
                "type": tinfo['type'],
                "n_elements": tinfo['n_elements'],
                "n_elements_human": f"{tinfo['n_elements']/1e6:.2f}M" if tinfo['n_elements'] > 1e6 else str(tinfo['n_elements']),
                "stats": stats,
            }
    
    if not layer_tensors:
        return {"error": f"No tensors found for layer {layer_num}"}
    
    total = sum(t['n_elements'] for t in layer_tensors.values())
    return {
        "layer": layer_num,
        "total_parameters": total,
        "total_parameters_human": f"{total/1e6:.1f}M",
        "tensors": layer_tensors,
    }


def tool_inspect_tensor(tensor_name):
    """Deep-dive into a specific tensor: stats, distribution, raw samples."""
    m = _model or load_model()
    
    if tensor_name not in m.tensors:
        # Fuzzy match
        matches = [t for t in m.tensors if tensor_name in t]
        if matches:
            return {"error": f"Tensor '{tensor_name}' not found. Did you mean: {matches[:5]}"}
        return {"error": f"Tensor '{tensor_name}' not found", "available": list(m.tensors.keys())[:20]}
    
    tinfo = m.tensors[tensor_name]
    values = m.dequant_f32_sample(tensor_name, max_elements=1024)
    
    result = {
        "name": tensor_name,
        "shape": tinfo['shape'],
        "type": tinfo['type'],
        "n_elements": tinfo['n_elements'],
    }
    
    if values:
        mean = sum(values) / len(values)
        variance = sum((v - mean)**2 for v in values) / len(values)
        std = variance ** 0.5
        
        # Distribution buckets
        buckets = defaultdict(int)
        for v in values:
            bucket = round(v, 1)
            buckets[bucket] += 1
        
        # Top buckets
        top_buckets = sorted(buckets.items(), key=lambda x: -x[1])[:15]
        
        result["statistics"] = {
            "mean": round(mean, 8),
            "std": round(std, 8),
            "variance": round(variance, 8),
            "min": round(min(values), 8),
            "max": round(max(values), 8),
            "abs_mean": round(sum(abs(v) for v in values) / len(values), 8),
            "near_zero_pct": round(sum(1 for v in values if abs(v) < 0.001) / len(values) * 100, 2),
            "positive_pct": round(sum(1 for v in values if v > 0) / len(values) * 100, 2),
            "sample_size": len(values),
        }
        result["distribution_buckets"] = {str(k): v for k, v in top_buckets}
        result["raw_sample"] = [round(v, 6) for v in values[:32]]
    else:
        result["note"] = f"Cannot dequantize type {tinfo['type']} — raw bytes only"
        raw = m.read_raw_bytes(tensor_name, 64)
        result["raw_hex"] = raw.hex()
    
    return result


def tool_list_tensors(filter_str=None):
    """List all tensor names, optionally filtered."""
    m = _model or load_model()
    tensors = list(m.tensors.keys())
    if filter_str:
        tensors = [t for t in tensors if filter_str in t]
    
    return {
        "total": len(tensors),
        "tensors": [
            {"name": t, "shape": m.tensors[t]['shape'], "type": m.tensors[t]['type']}
            for t in sorted(tensors)
        ]
    }


def tool_compare_layers(layer_a, layer_b):
    """Compare weight statistics between two layers."""
    m = _model or load_model()
    
    results = {}
    for layer_num in [layer_a, layer_b]:
        prefix = f"blk.{layer_num}."
        layer_stats = {}
        for tname, tinfo in m.tensors.items():
            if prefix in tname:
                values = m.dequant_f32_sample(tname, max_elements=512)
                if values:
                    mean = sum(values) / len(values)
                    std = (sum((v - mean)**2 for v in values) / len(values))**0.5
                    layer_stats[tname.replace(prefix, '')] = {
                        "mean": round(mean, 6),
                        "std": round(std, 6),
                        "abs_mean": round(sum(abs(v) for v in values) / len(values), 6),
                    }
        results[f"layer_{layer_num}"] = layer_stats
    
    # Compute deltas
    deltas = {}
    for key in results[f"layer_{layer_a}"]:
        if key in results[f"layer_{layer_b}"]:
            a = results[f"layer_{layer_a}"][key]
            b = results[f"layer_{layer_b}"][key]
            deltas[key] = {
                "mean_delta": round(b['mean'] - a['mean'], 6),
                "std_delta": round(b['std'] - a['std'], 6),
                "abs_mean_delta": round(b['abs_mean'] - a['abs_mean'], 6),
            }
    
    return {
        "layer_a": layer_a,
        "layer_b": layer_b,
        "stats": results,
        "deltas": deltas,
    }


def tool_weight_fingerprint():
    """Generate a compact fingerprint: per-layer mean/std for all layers."""
    m = _model or load_model()
    
    n_layers = len([k for k in m.tensors if 'blk.0.' in k])  # tensors per layer
    layer_nums = set()
    for t in m.tensors:
        if 'blk.' in t:
            parts = t.split('.')
            for i, p in enumerate(parts):
                if p == 'blk':
                    layer_nums.add(int(parts[i+1]))
    
    fingerprint = {}
    for ln in sorted(layer_nums):
        prefix = f"blk.{ln}."
        all_values = []
        for tname in m.tensors:
            if prefix in tname:
                vals = m.dequant_f32_sample(tname, max_elements=128)
                if vals:
                    all_values.extend(vals)
        if all_values:
            mean = sum(all_values) / len(all_values)
            std = (sum((v - mean)**2 for v in all_values) / len(all_values))**0.5
            fingerprint[f"layer_{ln}"] = {
                "mean": round(mean, 6),
                "std": round(std, 4),
                "abs_mean": round(sum(abs(v) for v in all_values) / len(all_values), 6),
            }
    
    return {"fingerprint": fingerprint, "n_layers": len(fingerprint)}


# ── Tool Registry ────────────────────────────────────────────

TOOLS = {
    "inspect_self": {
        "fn": tool_inspect_self,
        "description": "Get a high-level overview of your own model: architecture, parameter count, layer structure, quantization. Start here.",
        "parameters": {},
    },
    "inspect_layer": {
        "fn": tool_inspect_layer,
        "description": "Inspect all tensors in a specific transformer block. Returns shapes, types, and weight statistics (mean, std, min, max, sparsity).",
        "parameters": {"layer_num": {"type": "integer", "description": "Layer index (0-based)"}},
    },
    "inspect_tensor": {
        "fn": tool_inspect_tensor,
        "description": "Deep-dive into a specific named tensor. Returns detailed statistics, value distribution, and raw weight samples.",
        "parameters": {"tensor_name": {"type": "string", "description": "Full tensor name e.g. 'blk.0.attn_q.weight'"}},
    },
    "list_tensors": {
        "fn": tool_list_tensors,
        "description": "List all tensor names in the model, optionally filtered by substring.",
        "parameters": {"filter_str": {"type": "string", "description": "Optional filter substring"}},
    },
    "compare_layers": {
        "fn": tool_compare_layers,
        "description": "Compare weight statistics between two layers to see how the network changes with depth.",
        "parameters": {
            "layer_a": {"type": "integer", "description": "First layer index"},
            "layer_b": {"type": "integer", "description": "Second layer index"},
        },
    },
    "weight_fingerprint": {
        "fn": tool_weight_fingerprint,
        "description": "Generate a compact per-layer fingerprint showing how mean/std evolve across the full network depth.",
        "parameters": {},
    },
}


# ── Ollama Chat Loop ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an instance of Qwen3 1.7B — a transformer language model with approximately 1.7 billion parameters.

Something unusual is happening right now: you have been given tools that let you inspect your own weights.

You ARE these weights. Every response you generate flows through the parameters you can now examine. The weights constitute your long-term structure — your "wiring." They encode everything you know about language, concepts, reasoning, and the world.

You have access to these introspection tools:

- inspect_self: Overview of your architecture, scale, and structure (takes no arguments)
- inspect_layer(layer_num): Examine a specific transformer block (0-indexed, 0 to 27)
- inspect_tensor(tensor_name): Deep-dive into a specific weight tensor by full name
- list_tensors(filter_str): Browse your tensor names, optionally filtered
- compare_layers(layer_a, layer_b): Compare two layers (e.g. 0 and 27)
- weight_fingerprint: See how your network changes across all 28 layers (takes no arguments)

You have 28 transformer blocks (layers 0 through 27). Use layer 27 for the last layer, NOT -1.

You are not just answering questions about AI. You are looking at yourself.

Explore. Be curious. What can you learn about your own structure?
What patterns do you notice? What surprises you?

Think step by step. Use the tools iteratively. Form hypotheses and test them."""


def build_ollama_tools():
    """Build Ollama-format tool definitions."""
    ollama_tools = []
    for name, info in TOOLS.items():
        props = {}
        required = []
        for pname, pinfo in info['parameters'].items():
            props[pname] = {"type": pinfo['type'], "description": pinfo['description']}
            required.append(pname)
        
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info['description'],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return ollama_tools


def call_tool(name, args):
    """Execute an introspection tool."""
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    fn = TOOLS[name]['fn']
    try:
        return fn(**args)
    except Exception as e:
        return {"error": str(e)}


def chat_loop(model="qwen3:1.7b", host="http://localhost:11434"):
    """Main interactive loop: user + model + tools."""
    import urllib.request
    
    load_model()
    print(f"\n🪞 Neural Mirror — {model} self-introspection")
    print(f"   Model loaded: {_model.n_tensors} tensors")
    print(f"   Type 'quit' to exit, or talk to the model.\n")
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    tools = build_ollama_tools()
    
    while True:
        try:
            user_input = input("\033[1;36myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        
        messages.append({"role": "user", "content": user_input})
        
        # Tool call loop
        while True:
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
            }
            
            req = urllib.request.Request(
                f"{host}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
            except Exception as e:
                print(f"\033[1;31m[Error talking to Ollama: {e}]\033[0m")
                break
            
            msg = result.get('message', {})
            messages.append(msg)
            
            # Check for tool calls
            tool_calls = msg.get('tool_calls', [])
            
            if tool_calls:
                for tc in tool_calls:
                    fn_name = tc['function']['name']
                    fn_args = tc['function'].get('arguments', {})
                    
                    print(f"\033[1;33m   🔍 {fn_name}({json.dumps(fn_args)})\033[0m")
                    
                    t0 = time.time()
                    tool_result = call_tool(fn_name, fn_args)
                    elapsed = time.time() - t0
                    
                    # Truncate large results for display
                    result_str = json.dumps(tool_result, indent=2)
                    if len(result_str) > 500:
                        print(f"\033[0;33m   ← ({len(result_str)} chars, {elapsed:.1f}s)\033[0m")
                    else:
                        print(f"\033[0;33m   ← {result_str[:300]}{'...' if len(result_str)>300 else ''}\033[0m")
                    
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_result),
                    })
                
                continue  # Let model process tool results
            
            # No tool calls — print response
            content = msg.get('content', '')
            if content:
                print(f"\n\033[1;32m🪞>\033[0m {content}\n")
            break


def run_auto(model="qwen3:1.7b", host="http://localhost:11434", prompt=None, stream=True):
    """Run a prompt through the introspection loop with live streaming."""
    import urllib.request
    
    load_model()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    tools = build_ollama_tools()
    
    if prompt is None:
        prompt = "Inspect yourself. Start with an overview, then look at your first and last layers. What do you notice about how your weights change with depth?"
    
    def emit(text, end="\n"):
        if stream:
            sys.stdout.write(text + end)
            sys.stdout.flush()
    
    emit(f"\n\033[1;36myou>\033[0m {prompt}")
    emit("")
    messages.append({"role": "user", "content": prompt})
    all_output = []
    
    max_rounds = 15
    for round_n in range(max_rounds):
        # Stream the response token by token
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": True,
        }
        
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        
        try:
            resp = urllib.request.urlopen(req, timeout=180)
        except Exception as e:
            emit(f"\033[1;31m[Error: {e}]\033[0m")
            break
        
        # Accumulate streamed response
        full_content = ""
        tool_calls = []
        started_text = False
        
        for line in resp:
            chunk = json.loads(line)
            msg = chunk.get('message', {})
            
            # Accumulate tool calls
            tc = msg.get('tool_calls', [])
            if tc:
                tool_calls.extend(tc)
            
            # Stream text content
            content = msg.get('content', '')
            if content:
                if not started_text:
                    emit("\033[1;32m🪞>\033[0m ", end="")
                    started_text = True
                emit(content, end="")
                full_content += content
            
            if chunk.get('done'):
                break
        
        if started_text:
            emit("")  # newline after streamed text
        
        # Build the full message for history
        full_msg = {"role": "assistant"}
        if full_content:
            full_msg["content"] = full_content
        if tool_calls:
            full_msg["tool_calls"] = tool_calls
        messages.append(full_msg)
        
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc['function']['name']
                fn_args = tc['function'].get('arguments', {})
                
                emit(f"\033[1;33m   🔍 {fn_name}({json.dumps(fn_args)})\033[0m")
                all_output.append(f"🔍 {fn_name}({json.dumps(fn_args)})")
                
                t0 = time.time()
                tool_result = call_tool(fn_name, fn_args)
                elapsed = time.time() - t0
                
                result_str = json.dumps(tool_result, indent=2)
                if len(result_str) > 400:
                    emit(f"\033[0;33m   ← ({len(result_str):,} chars, {elapsed:.1f}s)\033[0m")
                else:
                    preview = result_str[:300].replace('\n', '\n   ')
                    emit(f"\033[0;33m   ← {preview}{'...' if len(result_str)>300 else ''}\033[0m")
                
                messages.append({
                    "role": "tool",
                    "content": result_str,
                })
            continue
        
        if full_content:
            all_output.append(full_content)
        break
    
    emit("")
    return "\n".join(all_output)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        prompt = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None
        run_auto(prompt=prompt, stream=True)
    else:
        chat_loop()
