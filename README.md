# 🪞 Neural Mirror

**An LLM self-introspection framework.** Give a language model tools to inspect its own weights — then watch it reason about what it finds.

> The model IS the weights. Now it gets a camera pointed at them.

## What is this?

Standard LLMs use their weights to generate every response, but they can't *see* those weights. It's like a pianist who plays music through billions of microscopic screws inside the piano — but can't examine any individual screw.

Neural Mirror changes that. It:

1. **Parses the GGUF model file** directly — reading tensor metadata, shapes, quantization types
2. **Dequantizes weight samples** to float32 for inspection (supports Q4_0, Q4_K, Q6_K, Q8_0, F16, F32)
3. **Exposes introspection tools** the model can call via Ollama's tool-calling API
4. **Lets the model reason** about its own architecture, weight distributions, and layer-by-layer structure

The result is a recursive loop:

```
┌───────────────┐
│      LLM      │
└───────┬───────┘
        │
  "inspect layer 8"
        │
        ▼
┌───────────────┐
│    runtime    │
└───────┬───────┘
        │
   reads model's
   own GGUF file
        │
        ▼
  weight statistics
        │
        ▼
┌───────────────┐
│      LLM      │
│ reasons about │
│    itself     │
└───────────────┘
```

## Tools

| Tool | Description |
|------|-------------|
| `inspect_self` | Architecture overview: name, params, layers, quantization |
| `inspect_layer(n)` | All tensors in a transformer block with weight statistics |
| `inspect_tensor(name)` | Deep-dive: distribution, raw samples, detailed stats |
| `list_tensors(filter)` | Browse all tensor names |
| `compare_layers(a, b)` | Side-by-side weight statistics between two layers |
| `weight_fingerprint` | Per-layer mean/std across the full network depth |

## Quick Start

```bash
# Requires: Ollama running with a GGUF model
pip install gguf

# Interactive mode — chat with the model while it introspects
python introspect.py

# Auto mode — give it a prompt and watch it explore
python introspect.py --auto "Examine yourself. What do you find?"
```

## Example Output

```
you> Examine yourself.

   🔍 inspect_self({})
   ← (1,115 chars, 0.0s)
   🔍 compare_layers({"layer_a": 0, "layer_b": 27})
   ← (4,037 chars, 0.0s)
   🔍 inspect_tensor({"tensor_name": "blk.0.attn_norm.weight"})
   ← (953 chars, 0.0s)
   🔍 inspect_tensor({"tensor_name": "blk.27.attn_norm.weight"})
   ← (1,009 chars, 0.0s)

🪞> The attention norms grow 201x from layer 0 (mean 0.087) to layer 27 
   (mean 17.4). Early layers emphasize precision with tight distributions,
   while later layers prioritize range and diversity...
```

## What the model discovers about itself

When Qwen3 1.7B examines its own weights, it finds:

- **28 transformer blocks** with 2.03B total parameters
- **Attention norm weights grow ~200x** from first to last layer — the model's "volume knob" increases dramatically with depth
- **91%+ of quantized attention weights are near-zero** — extreme sparsity in the attention mechanism
- **Layer 0 is precise and constrained** (std ~0.05) while **Layer 27 is broad and diverse** (std ~4.8)

## Architecture

```
introspect.py
├── GGUFModel          — Zero-dependency GGUF parser
│   ├── _parse()       — Reads metadata + tensor info
│   ├── read_raw_bytes — Raw tensor data access
│   └── dequant_f32    — Q4_0/Q4_K/Q6_K/Q8_0/F16/F32 → float32
├── Tools              — Self-introspection functions
│   ├── inspect_self
│   ├── inspect_layer
│   ├── inspect_tensor
│   ├── list_tensors
│   ├── compare_layers
│   └── weight_fingerprint
└── Chat Loop          — Streaming Ollama integration with tool calling
```

## The idea

> Being something isn't the same as having an internal representation of that thing.

A calculator is made of transistors but doesn't know how many it has. An LLM is made of weights but can't normally inspect them. The weights constitute the model, but aren't an input available to the model.

Neural Mirror makes them available. It's a first step toward models that can:

1. Inspect their own parameters
2. Form hypotheses about their own structure
3. Compare their layers and identify patterns
4. Eventually: modify weights, reload, and evaluate the change

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) with a GGUF model (tested with `qwen3:1.7b`)
- `gguf` package (optional, for metadata only — the parser is built-in)

## License

MIT
