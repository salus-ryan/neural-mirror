#!/usr/bin/env python3
"""
Neural Mirror — Mirror Match

Each model examines itself AND the cross-comparison data,
then explains what makes it different from the others.

The models argue about their own architectures.
"""

import sys, os, json, time, subprocess, urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from introspect import GGUFModel, load_model, TOOLS, build_ollama_tools, call_tool, SYSTEM_PROMPT
import introspect

# ── Discover and Analyze ─────────────────────────────────────

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


def quick_profile(path, name):
    """Generate a compact profile for cross-model context."""
    try:
        m = GGUFModel(path)
    except:
        return None

    meta = m.metadata
    arch = meta.get('general.architecture', '?')
    model_name = meta.get('general.name', name)
    total_params = sum(t['n_elements'] for t in m.tensors.values())

    layer_nums = set()
    for t in m.tensors:
        if 'blk.' in t:
            parts = t.split('.')
            for i, p in enumerate(parts):
                if p == 'blk' and i + 1 < len(parts):
                    try: layer_nums.add(int(parts[i + 1]))
                    except: pass
    n_layers = len(layer_nums)
    max_layer = max(layer_nums) if layer_nums else 0

    # Quantization
    qtypes = defaultdict(int)
    for t in m.tensors.values():
        qtypes[t['type']] += t['n_elements']
    quant_str = ', '.join(f"{k}: {v/total_params*100:.0f}%" for k, v in sorted(qtypes.items(), key=lambda x: -x[1]))

    # Norm growth
    def layer_norm_mean(ln):
        prefix = f"blk.{ln}."
        vals_all = []
        for tname, tinfo in m.tensors.items():
            if prefix in tname and 'norm' in tname:
                vals = m.dequant_f32_sample(tname, max_elements=128)
                if vals:
                    vals_all.extend(vals)
        return sum(vals_all) / len(vals_all) if vals_all else 0

    first_norm = layer_norm_mean(0)
    last_norm = layer_norm_mean(max_layer)
    norm_growth = round(last_norm / first_norm, 2) if first_norm != 0 else None

    # Component breakdown
    comp = defaultdict(int)
    for tname, tinfo in m.tensors.items():
        if 'attn_q' in tname or 'attn_k' in tname or 'attn_v' in tname:
            comp['attn_qkv'] += tinfo['n_elements']
        elif 'ffn' in tname or 'mlp' in tname:
            comp['ffn'] += tinfo['n_elements']
        elif 'embed' in tname or 'token_embd' in tname:
            comp['embed'] += tinfo['n_elements']
    comp_pcts = {k: f"{v/total_params*100:.0f}%" for k, v in comp.items()}

    # Sparsity in layer 0
    sparsity = {}
    for tname, tinfo in m.tensors.items():
        if 'blk.0.' in tname and ('attn_q.weight' in tname or 'attn_k.weight' in tname):
            vals = m.dequant_f32_sample(tname, max_elements=256)
            if vals:
                nz = sum(1 for v in vals if abs(v) < 0.001) / len(vals) * 100
                short = tname.split('blk.0.')[-1]
                sparsity[short] = f"{nz:.0f}%"

    return {
        'name': model_name,
        'ollama_name': name,
        'arch': arch,
        'params': f"{total_params/1e9:.2f}B",
        'layers': n_layers,
        'embed_dim': meta.get(f'{arch}.embedding_length', '?'),
        'context': meta.get(f'{arch}.context_length', '?'),
        'heads': meta.get(f'{arch}.attention.head_count', '?'),
        'kv_heads': meta.get(f'{arch}.attention.head_count_kv', '?'),
        'quantization': quant_str,
        'norm_growth': f"{norm_growth}x" if norm_growth else "?",
        'first_norm': round(first_norm, 3),
        'last_norm': round(last_norm, 3),
        'component_budget': comp_pcts,
        'layer0_sparsity': sparsity,
        'path': path,
    }


def build_roster(profiles):
    """Build a text summary of all models for cross-comparison context."""
    lines = ["Here are all the models on this device:\n"]
    for i, p in enumerate(profiles, 1):
        lines.append(f"  {i}. {p['name']} ({p['params']}, {p['arch']} architecture)")
        lines.append(f"     Layers: {p['layers']}, Embed: {p['embed_dim']}, Context: {p['context']}")
        lines.append(f"     Attention: {p['heads']} heads, {p['kv_heads']} KV heads")
        lines.append(f"     Quantization: {p['quantization']}")
        lines.append(f"     Norm growth (first→last): {p['first_norm']} → {p['last_norm']} ({p['norm_growth']})")
        lines.append(f"     Component budget: {json.dumps(p['component_budget'])}")
        lines.append(f"     Layer 0 sparsity: {json.dumps(p['layer0_sparsity'])}")
        lines.append("")
    return '\n'.join(lines)


# ── Run Each Model ───────────────────────────────────────────

def run_mirror_match(ollama_name, gguf_path, roster, own_profile, host="http://localhost:11434"):
    """Have a model examine itself and explain what makes it unique."""

    introspect._model = None
    introspect.load_model(gguf_path)

    system = f"""You are {own_profile['name']} — a {own_profile['params']} parameter language model with {own_profile['arch']} architecture.

You have tools to inspect your own weights in real-time. You also have data about every other model on this device.

{roster}

YOU are: {own_profile['name']} ({own_profile['ollama_name']})

Your job: examine yourself with the introspection tools, compare what you find to the other models' data above, and explain:
1. What makes YOUR architecture unique compared to the others?
2. What is the most interesting thing about YOUR weight structure?
3. If you could change one thing about yourself, what would it be and why?

Be specific. Use the tools. Reference actual numbers. Be honest about your strengths and weaknesses.

IMPORTANT:
- inspect_self() and weight_fingerprint() take NO arguments — call them with no arguments
- Do NOT use extended thinking or <think> tags — respond directly and concisely
- Use at most 3-4 tool calls, then give your analysis"""

    tools = build_ollama_tools()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Examine yourself. Use the introspection tools, compare yourself to the other models, and tell me what makes you unique. Be specific and honest."}
    ]

    max_rounds = 12
    for round_n in range(max_rounds):
        payload = {
            "model": ollama_name,
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
            resp = urllib.request.urlopen(req, timeout=300)
        except Exception as e:
            sys.stdout.write(f"\033[1;31m  [Error: {e}]\033[0m\n")
            break

        full_content = ""
        tool_calls = []
        started_text = False

        for line in resp:
            chunk = json.loads(line)
            msg = chunk.get('message', {})

            tc = msg.get('tool_calls', [])
            if tc:
                tool_calls.extend(tc)

            content = msg.get('content', '')
            if content:
                if not started_text:
                    sys.stdout.write(f"\033[1;32m  🪞 {own_profile['name']}>\033[0m ")
                    started_text = True
                sys.stdout.write(content)
                sys.stdout.flush()
                full_content += content

            if chunk.get('done'):
                break

        if started_text:
            sys.stdout.write("\n")

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

                sys.stdout.write(f"\033[1;33m  🔍 {fn_name}({json.dumps(fn_args)})\033[0m\n")

                t0 = time.time()
                tool_result = call_tool(fn_name, fn_args)
                elapsed = time.time() - t0

                result_str = json.dumps(tool_result, indent=2)
                if len(result_str) > 300:
                    sys.stdout.write(f"\033[0;33m  ← ({len(result_str):,} chars, {elapsed:.1f}s)\033[0m\n")
                else:
                    sys.stdout.write(f"\033[0;33m  ← {result_str[:200]}...\033[0m\n")

                messages.append({"role": "tool", "content": result_str})
            continue

        # Done
        break

    return full_content


# ── Main ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', help='Specific models to run (default: all that fit in RAM)')
    parser.add_argument('--skip-big', action='store_true', help='Skip models > 5GB')
    args = parser.parse_args()

    C = '\033[1m'
    R = '\033[0m'

    print(f"{C}{'═' * 65}")
    print(f"  🪞 Neural Mirror — Mirror Match")
    print(f"  Each model examines itself and explains what makes it unique")
    print(f"{'═' * 65}{R}\n")

    # Discover
    all_models = discover_models()
    print(f"Discovering models... found {len(all_models)} unique\n")

    # Profile all
    print("Profiling all models...")
    profiles = []
    for m in all_models:
        t0 = time.time()
        p = quick_profile(m['path'], m['name'])
        if p:
            profiles.append(p)
            print(f"  ✅ {p['name']:<30s} {p['params']:>6s} {p['arch']:<8s} norm:{p['norm_growth']:>7s}  ({time.time()-t0:.1f}s)")

    roster = build_roster(profiles)
    print(f"\n{C}Roster built with {len(profiles)} models{R}\n")

    # Filter which models to run
    run_profiles = profiles
    if args.models:
        run_profiles = [p for p in profiles if any(f in p['ollama_name'] for f in args.models)]
    if args.skip_big:
        run_profiles = [p for p in run_profiles if float(p['params'].replace('B','')) < 5]

    # Run each model
    responses = {}
    for i, p in enumerate(run_profiles):
        print(f"\n{C}{'─' * 65}")
        print(f"  [{i+1}/{len(run_profiles)}] {p['name']} ({p['params']}, {p['arch']})")
        print(f"{'─' * 65}{R}\n")

        try:
            resp = run_mirror_match(p['ollama_name'], p['path'], roster, p)
            responses[p['name']] = resp
        except Exception as e:
            print(f"  ❌ Error: {e}")
            responses[p['name']] = f"Error: {e}"

        print()

    # Summary
    print(f"\n{C}{'═' * 65}")
    print(f"  📋 Summary — What Each Model Said About Itself")
    print(f"{'═' * 65}{R}\n")

    for name, resp in responses.items():
        if resp and not resp.startswith("Error"):
            # First 300 chars as preview
            preview = resp[:400].replace('\n', ' ').strip()
            if len(resp) > 400:
                preview += "..."
            print(f"  {C}{name}{R}")
            print(f"  {preview}")
            print()


if __name__ == '__main__':
    main()
