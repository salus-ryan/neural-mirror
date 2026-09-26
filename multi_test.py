#!/usr/bin/env python3
"""
Neural Mirror — Multi-Model Self-Introspection Test

Parse every local GGUF model and have each one examine itself.
Then run the ones with tool-calling capability through the full loop.
"""

import sys, json, time, subprocess, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from introspect import GGUFModel, load_model, tool_inspect_self, tool_inspect_layer, tool_weight_fingerprint, tool_compare_layers, tool_inspect_tensor, run_auto, _model, TOOLS
import introspect

# ── Discover all Ollama models ────────────────────────────────

def discover_models():
    """Find all local Ollama models and their GGUF paths."""
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    models = []
    for line in result.stdout.strip().split('\n')[1:]:  # skip header
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        size = parts[2] + ' ' + parts[3] if len(parts) > 3 else '?'
        
        # Get GGUF path
        mf = subprocess.run(['ollama', 'show', name, '--modelfile'], 
                          capture_output=True, text=True)
        path = None
        for mline in mf.stdout.split('\n'):
            if mline.startswith('FROM /'):
                path = mline[5:].strip()
                break
        
        if path and os.path.isfile(path):
            models.append({'name': name, 'size': size, 'path': path})
    
    # Deduplicate by path (some models share the same GGUF)
    seen = {}
    unique = []
    for m in models:
        if m['path'] not in seen:
            seen[m['path']] = m['name']
            unique.append(m)
        else:
            m['note'] = f"same GGUF as {seen[m['path']]}"
            unique.append(m)
    
    return unique


# ── Static Analysis (no inference needed) ─────────────────────

def analyze_model_static(path, name):
    """Parse GGUF and extract architecture info without running inference."""
    try:
        model = GGUFModel(path)
    except Exception as e:
        return {'name': name, 'error': str(e)}
    
    meta = model.metadata
    arch = meta.get('general.architecture', '?')
    model_name = meta.get('general.name', name)
    
    total_params = sum(t['n_elements'] for t in model.tensors.values())
    
    # Count layers
    layer_nums = set()
    for t in model.tensors:
        if 'blk.' in t:
            parts = t.split('.')
            for i, p in enumerate(parts):
                if p == 'blk' and i + 1 < len(parts):
                    try:
                        layer_nums.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    
    n_layers = len(layer_nums)
    
    # Quantization breakdown
    from collections import defaultdict
    qtypes = defaultdict(int)
    for t in model.tensors.values():
        qtypes[t['type']] += t['n_elements']
    
    # Key architecture params
    arch_params = {}
    for k, v in meta.items():
        if k.startswith(arch + '.') or k in ['general.name', 'general.size_label', 'general.file_type']:
            short_k = k.replace(arch + '.', '').replace('general.', '')
            arch_params[short_k] = v
    
    # Sample weight stats from first and last layer
    layer_stats = {}
    introspect._model = model
    for ln in [0, max(layer_nums) if layer_nums else 0]:
        prefix = f"blk.{ln}."
        for tname, tinfo in model.tensors.items():
            if prefix in tname and 'norm' in tname:
                vals = model.dequant_f32_sample(tname, max_elements=256)
                if vals:
                    mean = sum(vals) / len(vals)
                    std = (sum((v - mean)**2 for v in vals) / len(vals))**0.5
                    layer_stats[tname] = {'mean': round(mean, 4), 'std': round(std, 4)}
    
    return {
        'name': model_name,
        'ollama_name': name,
        'architecture': arch,
        'total_params': total_params,
        'total_params_human': f"{total_params/1e9:.2f}B",
        'n_tensors': len(model.tensors),
        'n_layers': n_layers,
        'context_length': meta.get(f'{arch}.context_length', '?'),
        'embedding_dim': meta.get(f'{arch}.embedding_length', '?'),
        'attention_heads': meta.get(f'{arch}.attention.head_count', '?'),
        'kv_heads': meta.get(f'{arch}.attention.head_count_kv', '?'),
        'quantization': {k: f"{v/1e6:.1f}M" for k, v in sorted(qtypes.items(), key=lambda x: -x[1])},
        'norm_stats': layer_stats,
    }


# ── Live Introspection (requires inference) ───────────────────

def run_model_introspection(ollama_name, gguf_path, prompt=None):
    """Have a model examine itself via tool calling."""
    introspect._model = None  # reset
    introspect.load_model(gguf_path)
    
    if prompt is None:
        prompt = (
            "You have tools to inspect your own weights. "
            "Use inspect_self to see your architecture, then compare_layers 0 and your last layer. "
            "What is the most interesting thing about your own structure?"
        )
    
    return run_auto(model=ollama_name, prompt=prompt, stream=True)


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Neural Mirror Multi-Model Test')
    parser.add_argument('--static-only', action='store_true', help='Only parse GGUFs, skip inference')
    parser.add_argument('--models', nargs='*', help='Specific model names to test')
    parser.add_argument('--live', nargs='?', const='all', help='Run live introspection (model name or "all")')
    args = parser.parse_args()
    
    print("═" * 60)
    print("  🪞 Neural Mirror — Multi-Model Analysis")
    print("═" * 60)
    print()
    
    models = discover_models()
    if args.models:
        models = [m for m in models if any(f in m['name'] for f in args.models)]
    
    print(f"Found {len(models)} models:\n")
    for m in models:
        note = f" ({m['note']})" if 'note' in m else ""
        print(f"  • {m['name']:30s} {m['size']:>10s}{note}")
    print()
    
    # ── Static analysis ──
    print("─" * 60)
    print("  📊 Static GGUF Analysis")
    print("─" * 60)
    
    results = []
    seen_paths = set()
    
    for m in models:
        if m['path'] in seen_paths:
            print(f"\n  ⏭  {m['name']} — same GGUF as above, skipping")
            continue
        seen_paths.add(m['path'])
        
        print(f"\n  🔬 {m['name']}")
        t0 = time.time()
        info = analyze_model_static(m['path'], m['name'])
        elapsed = time.time() - t0
        
        if 'error' in info:
            print(f"     ❌ {info['error']}")
            continue
        
        print(f"     Model:       {info['name']}")
        print(f"     Arch:        {info['architecture']}")
        print(f"     Parameters:  {info['total_params_human']} ({info['n_tensors']} tensors)")
        print(f"     Layers:      {info['n_layers']}")
        print(f"     Context:     {info['context_length']}")
        print(f"     Embed dim:   {info['embedding_dim']}")
        print(f"     Attn heads:  {info['attention_heads']} (KV: {info['kv_heads']})")
        print(f"     Quantization: {', '.join(f'{k}: {v}' for k, v in info['quantization'].items())}")
        
        if info['norm_stats']:
            print(f"     Norm evolution:")
            for tname, stats in sorted(info['norm_stats'].items()):
                layer = tname.split('.')[1]
                print(f"       Layer {layer:>2s}: mean={stats['mean']:>10.4f}  std={stats['std']:>8.4f}")
        
        print(f"     Parsed in {elapsed:.2f}s")
        results.append(info)
    
    # ── Comparison table ──
    if len(results) > 1:
        print()
        print("─" * 60)
        print("  📋 Comparison")
        print("─" * 60)
        print()
        print(f"  {'Model':<25s} {'Params':>8s} {'Layers':>7s} {'Ctx':>7s} {'Embed':>6s} {'Heads':>6s}")
        print(f"  {'─'*25} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")
        for r in results:
            print(f"  {r['name']:<25s} {r['total_params_human']:>8s} {r['n_layers']:>7d} {str(r['context_length']):>7s} {str(r['embedding_dim']):>6s} {str(r['attention_heads']):>6s}")
    
    if args.static_only:
        return
    
    # ── Live introspection ──
    if args.live:
        print()
        print("─" * 60)
        print("  🧠 Live Self-Introspection")
        print("─" * 60)
        
        live_models = models
        if args.live != 'all':
            live_models = [m for m in models if args.live in m['name']]
        
        # Deduplicate
        live_seen = set()
        for m in live_models:
            if m['path'] in live_seen:
                continue
            live_seen.add(m['path'])
            
            # Only run models that support tool calling well
            print(f"\n  🪞 {m['name']} examining itself...")
            print(f"  {'─' * 50}")
            
            try:
                run_model_introspection(m['name'], m['path'])
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            print()


if __name__ == '__main__':
    main()
