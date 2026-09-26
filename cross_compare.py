#!/usr/bin/env python3
"""
Neural Mirror — Cross-Model Comparison

Compares the internal structure of every local GGUF model side-by-side.
Finds what's universal vs unique across architectures.
"""

import sys, os, json, time, math, subprocess
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from introspect import GGUFModel, _f16_to_f32

# ── Discover Models ──────────────────────────────────────────

def discover_models():
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    models = []
    seen_paths = set()
    for line in result.stdout.strip().split('\n')[1:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        mf = subprocess.run(['ollama', 'show', name, '--modelfile'],
                          capture_output=True, text=True)
        path = None
        for mline in mf.stdout.split('\n'):
            if mline.startswith('FROM /'):
                path = mline[5:].strip()
                break
        if path and os.path.isfile(path) and path not in seen_paths:
            seen_paths.add(path)
            models.append({'name': name, 'path': path})
    return models


# ── Deep Analysis ────────────────────────────────────────────

def analyze_deep(path, name):
    """Full structural analysis of a GGUF model."""
    try:
        m = GGUFModel(path)
    except Exception as e:
        return {'name': name, 'error': str(e)}

    meta = m.metadata
    arch = meta.get('general.architecture', '?')
    model_name = meta.get('general.name', name)
    total_params = sum(t['n_elements'] for t in m.tensors.values())

    # Find layer numbers
    layer_nums = set()
    for t in m.tensors:
        if 'blk.' in t:
            parts = t.split('.')
            for i, p in enumerate(parts):
                if p == 'blk' and i + 1 < len(parts):
                    try:
                        layer_nums.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    n_layers = len(layer_nums)
    max_layer = max(layer_nums) if layer_nums else 0

    # Quantization
    qtypes = defaultdict(int)
    for t in m.tensors.values():
        qtypes[t['type']] += t['n_elements']

    # Per-layer norm analysis (all layers)
    norm_curve = {}
    for ln in sorted(layer_nums):
        prefix = f"blk.{ln}."
        norms = []
        for tname, tinfo in m.tensors.items():
            if prefix in tname and 'norm' in tname:
                vals = m.dequant_f32_sample(tname, max_elements=256)
                if vals:
                    mean = sum(vals) / len(vals)
                    norms.append(mean)
        if norms:
            norm_curve[ln] = sum(norms) / len(norms)

    # Tensor type breakdown per component
    component_params = defaultdict(int)
    for tname, tinfo in m.tensors.items():
        if 'attn_q' in tname or 'attn_k' in tname or 'attn_v' in tname:
            component_params['attention_qkv'] += tinfo['n_elements']
        elif 'attn_output' in tname or 'attn_o' in tname:
            component_params['attention_output'] += tinfo['n_elements']
        elif 'ffn' in tname or 'mlp' in tname:
            component_params['feedforward'] += tinfo['n_elements']
        elif 'norm' in tname:
            component_params['normalization'] += tinfo['n_elements']
        elif 'embed' in tname or 'token_embd' in tname:
            component_params['embedding'] += tinfo['n_elements']
        elif 'output' in tname:
            component_params['output_head'] += tinfo['n_elements']
        else:
            component_params['other'] += tinfo['n_elements']

    # Weight distribution stats per component
    component_stats = {}
    sample_tensors = {
        'first_attn_q': None, 'last_attn_q': None,
        'first_ffn': None, 'last_ffn': None,
        'first_norm': None, 'last_norm': None,
    }
    for tname in m.tensors:
        if f'blk.0.' in tname:
            if 'attn_q' in tname and 'norm' not in tname:
                sample_tensors['first_attn_q'] = tname
            elif ('ffn_gate' in tname or 'ffn_up' in tname or 'mlp.gate' in tname):
                sample_tensors['first_ffn'] = tname
            elif 'norm' in tname and sample_tensors['first_norm'] is None:
                sample_tensors['first_norm'] = tname
        if f'blk.{max_layer}.' in tname:
            if 'attn_q' in tname and 'norm' not in tname:
                sample_tensors['last_attn_q'] = tname
            elif ('ffn_gate' in tname or 'ffn_up' in tname or 'mlp.gate' in tname):
                sample_tensors['last_ffn'] = tname
            elif 'norm' in tname and sample_tensors['last_norm'] is None:
                sample_tensors['last_norm'] = tname

    for label, tname in sample_tensors.items():
        if tname:
            vals = m.dequant_f32_sample(tname, max_elements=512)
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean)**2 for v in vals) / len(vals))**0.5
                abs_mean = sum(abs(v) for v in vals) / len(vals)
                near_zero = sum(1 for v in vals if abs(v) < 0.001) / len(vals) * 100
                component_stats[label] = {
                    'tensor': tname,
                    'mean': round(mean, 6),
                    'std': round(std, 6),
                    'abs_mean': round(abs_mean, 6),
                    'near_zero_pct': round(near_zero, 1),
                }

    # Sparsity analysis
    sparsity_by_type = {}
    for tname, tinfo in m.tensors.items():
        if 'blk.0.' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=256)
            if vals:
                nz = sum(1 for v in vals if abs(v) < 0.001) / len(vals)
                ttype = tname.split('.')[-1] if '.' in tname else tname
                ttype = '.'.join(tname.split('.')[2:])  # e.g. attn_q.weight
                sparsity_by_type[ttype] = round(nz * 100, 1)

    # Norm growth ratio
    if norm_curve:
        first_norm = norm_curve.get(0, norm_curve.get(min(norm_curve.keys())))
        last_norm = norm_curve.get(max_layer, norm_curve.get(max(norm_curve.keys())))
        if first_norm and first_norm != 0:
            norm_growth = round(last_norm / first_norm, 2)
        else:
            norm_growth = None
    else:
        first_norm = last_norm = norm_growth = None

    return {
        'name': model_name,
        'ollama_name': name,
        'architecture': arch,
        'total_params': total_params,
        'total_params_human': f"{total_params/1e9:.2f}B",
        'n_tensors': len(m.tensors),
        'n_layers': n_layers,
        'max_layer': max_layer,
        'context_length': meta.get(f'{arch}.context_length', '?'),
        'embedding_dim': meta.get(f'{arch}.embedding_length', '?'),
        'attention_heads': meta.get(f'{arch}.attention.head_count', '?'),
        'kv_heads': meta.get(f'{arch}.attention.head_count_kv', '?'),
        'ff_dim': meta.get(f'{arch}.feed_forward_length', '?'),
        'quantization': dict(sorted(qtypes.items(), key=lambda x: -x[1])),
        'component_params': dict(component_params),
        'norm_curve': norm_curve,
        'norm_growth': norm_growth,
        'first_norm_mean': first_norm,
        'last_norm_mean': last_norm,
        'component_stats': component_stats,
        'sparsity': sparsity_by_type,
    }


# ── Display ──────────────────────────────────────────────────

C_RESET = '\033[0m'
C_BOLD = '\033[1m'
C_DIM = '\033[2m'
C_CYAN = '\033[36m'
C_GREEN = '\033[32m'
C_YELLOW = '\033[33m'
C_RED = '\033[31m'
C_MAGENTA = '\033[35m'

def bar(value, max_val, width=20, char='█'):
    if max_val == 0:
        return ''
    filled = int(value / max_val * width)
    return char * filled + '░' * (width - filled)


def print_comparison(results):
    n = len(results)

    # ── Architecture Overview ──
    print(f"\n{C_BOLD}{'═' * 70}")
    print(f"  🪞 Neural Mirror — Cross-Model Comparison ({n} models)")
    print(f"{'═' * 70}{C_RESET}\n")

    # Basic table
    max_params = max(r['total_params'] for r in results)
    print(f"  {C_BOLD}{'Model':<28s} {'Arch':<8s} {'Params':>8s} {'Layers':>7s} {'Embed':>6s} {'Ctx':>8s}{C_RESET}")
    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*7} {'─'*6} {'─'*8}")
    for r in results:
        params_bar = bar(r['total_params'], max_params, 12)
        print(f"  {r['name']:<28s} {r['architecture']:<8s} {r['total_params_human']:>8s} {r['n_layers']:>7d} {str(r['embedding_dim']):>6s} {str(r['context_length']):>8s}")

    # ── Parameter Budget ──
    print(f"\n{C_BOLD}  📊 Parameter Budget (% of total){C_RESET}\n")
    components = ['attention_qkv', 'attention_output', 'feedforward', 'normalization', 'embedding', 'output_head']
    comp_labels = {'attention_qkv': 'Attn QKV', 'attention_output': 'Attn Out', 'feedforward': 'FFN',
                   'normalization': 'Norms', 'embedding': 'Embed', 'output_head': 'Output'}

    print(f"  {'':<28s}", end='')
    for c in components:
        print(f" {comp_labels.get(c, c):>9s}", end='')
    print()
    print(f"  {'─'*28}", end='')
    for _ in components:
        print(f" {'─'*9}", end='')
    print()

    for r in results:
        print(f"  {r['name']:<28s}", end='')
        for c in components:
            pct = r['component_params'].get(c, 0) / r['total_params'] * 100 if r['total_params'] > 0 else 0
            if pct > 0:
                print(f" {pct:>8.1f}%", end='')
            else:
                print(f" {'—':>9s}", end='')
        print()

    # ── Quantization Comparison ──
    print(f"\n{C_BOLD}  🗜️  Quantization Mix{C_RESET}\n")
    all_qtypes = set()
    for r in results:
        all_qtypes.update(r['quantization'].keys())
    all_qtypes = sorted(all_qtypes)

    print(f"  {'':<28s}", end='')
    for qt in all_qtypes:
        print(f" {qt:>7s}", end='')
    print()
    print(f"  {'─'*28}", end='')
    for _ in all_qtypes:
        print(f" {'─'*7}", end='')
    print()

    for r in results:
        print(f"  {r['name']:<28s}", end='')
        for qt in all_qtypes:
            count = r['quantization'].get(qt, 0)
            if count > 0:
                pct = count / r['total_params'] * 100
                print(f" {pct:>6.1f}%", end='')
            else:
                print(f" {'—':>7s}", end='')
        print()

    # ── Norm Growth (the signature finding) ──
    print(f"\n{C_BOLD}  📈 Normalization Growth (first → last layer){C_RESET}\n")

    max_growth = max((abs(r['norm_growth']) for r in results if r.get('norm_growth')), default=1)

    for r in results:
        ng = r.get('norm_growth')
        fn = r.get('first_norm_mean', 0)
        ln = r.get('last_norm_mean', 0)
        if ng is not None:
            if ng > 10:
                color = C_RED
                label = "🔥 EXPLOSIVE"
            elif ng > 2:
                color = C_YELLOW
                label = "📈 Growing"
            elif ng > 0.8:
                color = C_GREEN
                label = "✅ Stable"
            else:
                color = C_MAGENTA
                label = "📉 Shrinking"

            growth_bar = bar(abs(ng), max_growth, 15)
            print(f"  {r['name']:<28s} {fn:>8.4f} → {ln:>8.4f}  {color}{ng:>7.1f}x{C_RESET}  {growth_bar}  {label}")
        else:
            print(f"  {r['name']:<28s} {'no data':>20s}")

    # ── Norm Curves (ASCII sparkline) ──
    print(f"\n{C_BOLD}  🌊 Norm Curve (mean norm per layer, scaled){C_RESET}\n")

    spark_chars = ' ▁▂▃▄▅▆▇█'

    for r in results:
        curve = r.get('norm_curve', {})
        if not curve:
            continue
        values = [curve.get(i, 0) for i in range(r['n_layers'])]
        if not values:
            continue
        vmin, vmax = min(values), max(values)
        vrange = vmax - vmin if vmax != vmin else 1

        sparkline = ''
        for v in values:
            idx = int((v - vmin) / vrange * (len(spark_chars) - 1))
            sparkline += spark_chars[idx]

        print(f"  {r['name']:<28s} [{sparkline}] {vmin:.2f}→{vmax:.2f}")

    # ── Weight Distribution: First vs Last Layer ──
    print(f"\n{C_BOLD}  ⚖️  Weight Distribution: First vs Last Layer{C_RESET}\n")

    print(f"  {'':<28s} {'── Attention Q ──':>30s}  {'── FFN Gate/Up ──':>30s}")
    print(f"  {'':<28s} {'first':>14s} {'last':>14s}  {'first':>14s} {'last':>14s}")
    print(f"  {'─'*28} {'─'*14} {'─'*14}  {'─'*14} {'─'*14}")

    for r in results:
        cs = r.get('component_stats', {})
        faq = cs.get('first_attn_q', {})
        laq = cs.get('last_attn_q', {})
        fff = cs.get('first_ffn', {})
        lff = cs.get('last_ffn', {})

        def fmt_stat(s):
            if not s:
                return f"{'—':>14s}"
            return f"{s.get('abs_mean', 0):>7.5f}±{s.get('std', 0):.4f}"

        print(f"  {r['name']:<28s} {fmt_stat(faq)} {fmt_stat(laq)}  {fmt_stat(fff)} {fmt_stat(lff)}")

    # ── Sparsity (Layer 0) ──
    print(f"\n{C_BOLD}  🕸️  Sparsity (% near-zero in layer 0){C_RESET}\n")

    all_sparse_types = set()
    for r in results:
        all_sparse_types.update(r.get('sparsity', {}).keys())
    # Pick common ones
    common_types = sorted([t for t in all_sparse_types if any(k in t for k in ['attn_q', 'attn_k', 'ffn_gate', 'ffn_down', 'attn_norm'])])[:6]

    if common_types:
        print(f"  {'':<28s}", end='')
        for st in common_types:
            short = st.replace('.weight', '')[:12]
            print(f" {short:>12s}", end='')
        print()
        print(f"  {'─'*28}", end='')
        for _ in common_types:
            print(f" {'─'*12}", end='')
        print()

        for r in results:
            print(f"  {r['name']:<28s}", end='')
            sp = r.get('sparsity', {})
            for st in common_types:
                val = sp.get(st)
                if val is not None:
                    if val > 80:
                        color = C_RED
                    elif val > 50:
                        color = C_YELLOW
                    else:
                        color = C_GREEN
                    print(f" {color}{val:>11.1f}%{C_RESET}", end='')
                else:
                    print(f" {'—':>12s}", end='')
            print()

    # ── Key Insights ──
    print(f"\n{C_BOLD}  💡 Key Findings{C_RESET}\n")

    # Most explosive norm growth
    explosive = max(results, key=lambda r: abs(r.get('norm_growth', 0) or 0))
    if explosive.get('norm_growth') and abs(explosive['norm_growth']) > 5:
        print(f"  🔥 {explosive['name']} has the most extreme norm growth: {explosive['norm_growth']}x")

    # Most stable
    stable = min(results, key=lambda r: abs((r.get('norm_growth', 1) or 1) - 1))
    print(f"  ✅ {stable['name']} has the most stable norms: {stable.get('norm_growth', 'N/A')}x")

    # Shrinking norms
    shrinking = [r for r in results if r.get('norm_growth') and r['norm_growth'] < 1]
    if shrinking:
        for s in shrinking:
            print(f"  📉 {s['name']} norms SHRINK with depth: {s['norm_growth']}x (unique!)")

    # Largest model
    biggest = max(results, key=lambda r: r['total_params'])
    smallest = min(results, key=lambda r: r['total_params'])
    print(f"  📏 Size range: {smallest['total_params_human']} ({smallest['name']}) → {biggest['total_params_human']} ({biggest['name']})")

    # Architecture diversity
    archs = set(r['architecture'] for r in results)
    print(f"  🏗️  Architectures: {', '.join(sorted(archs))}")

    # Attention efficiency
    for r in results:
        kv = r.get('kv_heads')
        heads = r.get('attention_heads')
        if isinstance(kv, int) and isinstance(heads, int) and kv < heads:
            ratio = heads // kv
            print(f"  🔄 {r['name']}: GQA with {ratio}:1 head sharing ({heads} heads, {kv} KV)")

    print()


def main():
    print(f"{C_BOLD}Discovering models...{C_RESET}")
    models = discover_models()
    print(f"Found {len(models)} unique models\n")

    print(f"{C_BOLD}Analyzing...{C_RESET}")
    results = []
    for m in models:
        t0 = time.time()
        info = analyze_deep(m['path'], m['name'])
        elapsed = time.time() - t0
        if 'error' not in info:
            results.append(info)
            print(f"  ✅ {info['name']:<30s} ({info['total_params_human']}, {elapsed:.1f}s)")
        else:
            print(f"  ❌ {m['name']}: {info['error']}")

    # Sort by parameter count
    results.sort(key=lambda r: r['total_params'])

    print_comparison(results)


if __name__ == '__main__':
    main()
