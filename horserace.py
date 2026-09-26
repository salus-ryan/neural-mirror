#!/usr/bin/env python3
"""
Neural Mirror — Horserace 🏇

Multiple models examine their own weights simultaneously.
Responses stream in real-time, braided and color-coded.
"""

import sys, os, json, time, threading, queue, struct, math, textwrap
from collections import defaultdict
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from introspect import GGUFModel, load_model, tool_inspect_self, tool_inspect_layer, tool_compare_layers, tool_weight_fingerprint, tool_inspect_tensor, call_tool, build_ollama_tools, TOOLS
import introspect

# ── Colors ────────────────────────────────────────────────────

COLORS = [
    '\033[1;36m',  # cyan
    '\033[1;33m',  # yellow
    '\033[1;35m',  # magenta
    '\033[1;32m',  # green
    '\033[1;31m',  # red
    '\033[1;34m',  # blue
    '\033[1;97m',  # white
    '\033[1;91m',  # bright red
]
C_DIM    = '\033[2m'
C_BOLD   = '\033[1m'
C_RESET  = '\033[0m'
C_TOOL   = '\033[0;33m'

# ── Model Discovery ──────────────────────────────────────────

def discover_models():
    import subprocess
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    models = []
    seen = set()
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
        if path and os.path.isfile(path) and path not in seen:
            seen.add(path)
            models.append({'name': name, 'path': path})
    return models


def quick_stats(path):
    """Quick GGUF stats for context."""
    try:
        m = GGUFModel(path)
        meta = m.metadata
        arch = meta.get('general.architecture', '?')
        name = meta.get('general.name', '?')
        params = sum(t['n_elements'] for t in m.tensors.values())
        layers = set()
        for t in m.tensors:
            if 'blk.' in t:
                parts = t.split('.')
                for i, p in enumerate(parts):
                    if p == 'blk' and i+1 < len(parts):
                        try: layers.add(int(parts[i+1]))
                        except: pass
        return {
            'model_name': name,
            'arch': arch,
            'params': f"{params/1e9:.2f}B",
            'n_layers': len(layers),
            'max_layer': max(layers) if layers else 0,
        }
    except:
        return None


# ── Streaming Worker ─────────────────────────────────────────

def model_worker(model_name, gguf_path, stats, prompt, color, output_q, host="http://localhost:11434"):
    """Worker thread: parse GGUF, call tools, stream response."""

    tag = f"{color}{stats['model_name']}{C_RESET}"
    short_tag = stats['model_name'][:20]

    # Load GGUF for this thread
    local_model = GGUFModel(gguf_path)

    # Build system prompt with self-knowledge
    system = f"""You are {stats['model_name']} — a {stats['params']} parameter {stats['arch']} model with {stats['n_layers']} layers.

You have tools to inspect your own weights. Call them and explain what you find.
- inspect_self() — no arguments, returns architecture overview
- inspect_layer(layer_num) — 0-indexed, your last layer is {stats['max_layer']}
- compare_layers(layer_a, layer_b) — compare two layers
- weight_fingerprint() — no arguments, per-layer stats
- inspect_tensor(tensor_name) — deep dive into a named tensor

IMPORTANT: inspect_self and weight_fingerprint take NO arguments.
Respond directly — no extended thinking or <think> tags. Be concise (3-4 sentences per point)."""

    tools = build_ollama_tools()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    # Temporarily set global model for tool calls
    old_model = introspect._model
    introspect._model = local_model

    max_rounds = 8
    for round_n in range(max_rounds):
        payload = {
            "model": model_name,
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
            output_q.put(('error', model_name, short_tag, color, f"[Error: {e}]"))
            break

        full_content = ""
        tool_calls = []
        token_buf = ""

        for line in resp:
            chunk = json.loads(line)
            msg = chunk.get('message', {})

            tc = msg.get('tool_calls', [])
            if tc:
                tool_calls.extend(tc)

            content = msg.get('content', '')
            if content:
                full_content += content
                token_buf += content
                # Emit word-by-word for smoother braiding
                while ' ' in token_buf or '\n' in token_buf:
                    # Find first break
                    space_idx = token_buf.find(' ')
                    nl_idx = token_buf.find('\n')
                    if nl_idx >= 0 and (space_idx < 0 or nl_idx < space_idx):
                        word = token_buf[:nl_idx+1]
                        token_buf = token_buf[nl_idx+1:]
                    elif space_idx >= 0:
                        word = token_buf[:space_idx+1]
                        token_buf = token_buf[space_idx+1:]
                    else:
                        break
                    output_q.put(('token', model_name, short_tag, color, word))

            if chunk.get('done'):
                break

        # Flush remaining buffer
        if token_buf:
            output_q.put(('token', model_name, short_tag, color, token_buf))

        # Build message for history
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

                output_q.put(('tool', model_name, short_tag, color, f"🔍 {fn_name}({json.dumps(fn_args)})"))

                # Temporarily set correct model for tool
                introspect._model = local_model
                tool_result = call_tool(fn_name, fn_args)
                result_str = json.dumps(tool_result)

                size = len(result_str)
                output_q.put(('tool_result', model_name, short_tag, color, f"← {size:,} chars"))

                messages.append({"role": "tool", "content": result_str})
            continue

        break

    output_q.put(('done', model_name, short_tag, color, ''))
    introspect._model = old_model


# ── Display Engine ───────────────────────────────────────────

def display_braided(output_q, model_names, colors_map, total_models):
    """Read from queue and display braided output."""

    active = set(model_names)
    last_model = None
    line_width = 70

    while active:
        try:
            event_type, model, short_tag, color, data = output_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if event_type == 'token':
            if model != last_model:
                if last_model is not None:
                    sys.stdout.write('\n')
                sys.stdout.write(f"  {color}{short_tag:>20s}{C_RESET} │ ")
                last_model = model

            # Clean up think tags
            clean = data.replace('<think>', '').replace('</think>', '')
            if clean:
                sys.stdout.write(clean)
                sys.stdout.flush()

        elif event_type == 'tool':
            if last_model is not None:
                sys.stdout.write('\n')
            sys.stdout.write(f"  {color}{short_tag:>20s}{C_RESET} │ {C_TOOL}{data}{C_RESET}\n")
            last_model = None

        elif event_type == 'tool_result':
            sys.stdout.write(f"  {' ':>20s} │ {C_DIM}{data}{C_RESET}\n")
            last_model = None

        elif event_type == 'error':
            if last_model is not None:
                sys.stdout.write('\n')
            sys.stdout.write(f"  {color}{short_tag:>20s}{C_RESET} │ \033[1;31m{data}{C_RESET}\n")
            active.discard(model)
            last_model = None

        elif event_type == 'done':
            if last_model == model:
                sys.stdout.write('\n')
            sys.stdout.write(f"  {color}{short_tag:>20s}{C_RESET} │ {C_DIM}✓ finished{C_RESET}\n")
            active.discard(model)
            last_model = None

    sys.stdout.write('\n')


# ── Main ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Neural Mirror Horserace')
    parser.add_argument('--models', nargs='*', help='Model names to race')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt')
    parser.add_argument('--max', type=int, default=4, help='Max models to race')
    args = parser.parse_args()

    # Header
    print(f"\n{C_BOLD}{'━' * 65}")
    print(f"  🏇 Neural Mirror — Horserace")
    print(f"  Models examine their own weights simultaneously")
    print(f"{'━' * 65}{C_RESET}\n")

    # Discover
    all_models = discover_models()

    if args.models:
        selected = [m for m in all_models if any(f in m['name'] for f in args.models)]
    else:
        # Pick interesting diverse set
        preferred = ['qwen3:1.7b', 'qwen2.5:0.5b', 'gemma4', 'qwen2.5-coder', 'qwen3.5:4b']
        selected = []
        for pref in preferred:
            for m in all_models:
                if pref in m['name'] and m not in selected:
                    selected.append(m)
                    break
        if not selected:
            selected = all_models[:args.max]

    selected = selected[:args.max]

    # Get stats for each
    print(f"  {C_DIM}Profiling {len(selected)} models...{C_RESET}")
    race_entries = []
    for i, m in enumerate(selected):
        stats = quick_stats(m['path'])
        if stats:
            color = COLORS[i % len(COLORS)]
            race_entries.append((m['name'], m['path'], stats, color))
            print(f"  {color}█{C_RESET} {stats['model_name']:<25s} {stats['params']:>6s}  {stats['arch']:<8s}  {stats['n_layers']} layers")

    if not race_entries:
        print("No valid models found!")
        return

    prompt = args.prompt or (
        "Examine yourself: call inspect_self, then compare your first and last layer. "
        "In 2-3 concise paragraphs, what is the most interesting thing about your own weight structure? "
        "Be specific with numbers."
    )

    print(f"\n  {C_DIM}Prompt: {prompt[:80]}{'...' if len(prompt)>80 else ''}{C_RESET}")
    print(f"\n{C_BOLD}{'━' * 65}")
    print(f"  🏁 GO!")
    print(f"{'━' * 65}{C_RESET}\n")

    # Launch all threads
    output_q = queue.Queue()
    threads = []
    model_names = []
    colors_map = {}

    for model_name, gguf_path, stats, color in race_entries:
        model_names.append(model_name)
        colors_map[model_name] = color

        t = threading.Thread(
            target=model_worker,
            args=(model_name, gguf_path, stats, prompt, color, output_q),
            daemon=True,
        )
        threads.append(t)

    # Start all threads simultaneously
    t0 = time.time()
    for t in threads:
        t.start()

    # Display braided output
    display_braided(output_q, model_names, colors_map, len(race_entries))

    # Wait for all threads
    for t in threads:
        t.join(timeout=5)

    elapsed = time.time() - t0

    # Footer
    print(f"{C_BOLD}{'━' * 65}")
    print(f"  🏁 Race complete in {elapsed:.1f}s")
    print(f"{'━' * 65}{C_RESET}\n")


if __name__ == '__main__':
    main()
