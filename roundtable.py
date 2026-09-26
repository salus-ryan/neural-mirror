#!/usr/bin/env python3
"""
Neural Mirror — Roundtable 🪞🔄

Multiple models examine themselves AND each other's findings.
Each round, every model sees what the others discovered and builds on it.
Co-learning through shared introspection.

Round 1: Each model inspects itself (parallel)
Round 2: Each model sees all Round 1 findings, reacts + digs deeper
Round 3: Each model synthesizes — what did we learn together?
"""

import sys, os, json, time, threading, queue, subprocess
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from introspect import GGUFModel, call_tool, build_ollama_tools, TOOLS
import introspect

# ── Colors ────────────────────────────────────────────────────

PALETTE = [
    ('\033[1;36m', '🔵'),  # cyan
    ('\033[1;33m', '🟡'),  # yellow
    ('\033[1;35m', '🟣'),  # magenta
    ('\033[1;32m', '🟢'),  # green
    ('\033[1;31m', '🔴'),  # red
    ('\033[1;34m', '🔷'),  # blue
]
C = '\033[0m'      # reset
CD = '\033[2m'     # dim
CB = '\033[1m'     # bold
CT = '\033[0;33m'  # tool yellow


# ── Model Discovery ──────────────────────────────────────────

def discover_models():
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


def profile_model(path):
    try:
        m = GGUFModel(path)
        meta = m.metadata
        arch = meta.get('general.architecture', '?')
        name = meta.get('general.name', '?')
        params = sum(t['n_elements'] for t in m.tensors.values())
        layers = set()
        for t in m.tensors:
            if 'blk.' in t:
                ps = t.split('.')
                for i, p in enumerate(ps):
                    if p == 'blk' and i+1 < len(ps):
                        try: layers.add(int(ps[i+1]))
                        except: pass
        return {
            'model_name': name, 'arch': arch,
            'params': f"{params/1e9:.2f}B",
            'n_layers': len(layers),
            'max_layer': max(layers) if layers else 0,
            'n_tensors': len(m.tensors),
        }
    except:
        return None


# ── Single Model Turn ────────────────────────────────────────

def run_turn(model_name, gguf_path, system_prompt, user_prompt, color, emoji, output_q, host="http://localhost:11434"):
    """One model's turn: tool calls + streaming response."""

    local_model = GGUFModel(gguf_path)
    tools = build_ollama_tools()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    old = introspect._model
    introspect._model = local_model

    full_response = ""
    tool_log = []

    max_rounds = 6
    for _ in range(max_rounds):
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
            output_q.put(('error', model_name, color, emoji, f"[Error: {e}]"))
            break

        content = ""
        tool_calls = []

        for line in resp:
            chunk = json.loads(line)
            msg = chunk.get('message', {})
            tc = msg.get('tool_calls', [])
            if tc:
                tool_calls.extend(tc)
            c = msg.get('content', '')
            if c:
                clean = c.replace('<think>', '').replace('</think>', '')
                if clean:
                    content += clean
                    output_q.put(('token', model_name, color, emoji, clean))
            if chunk.get('done'):
                break

        full_msg = {"role": "assistant"}
        if content:
            full_msg["content"] = content
            full_response += content
        if tool_calls:
            full_msg["tool_calls"] = tool_calls
        messages.append(full_msg)

        if tool_calls:
            for tc in tool_calls:
                fn = tc['function']['name']
                args = tc['function'].get('arguments', {})
                output_q.put(('tool', model_name, color, emoji, f"{fn}({json.dumps(args)})"))

                introspect._model = local_model
                result = call_tool(fn, args)
                result_str = json.dumps(result)
                tool_log.append(f"{fn}: {result_str[:200]}")

                messages.append({"role": "tool", "content": result_str})
            continue
        break

    output_q.put(('turn_done', model_name, color, emoji, ''))
    introspect._model = old
    return full_response, tool_log


def stream_display(output_q, participants, until_done_count):
    """Display streamed tokens from queue, prefixed by model color."""
    done = 0
    last_model = None
    responses = defaultdict(str)

    while done < until_done_count:
        try:
            etype, model, color, emoji, data = output_q.get(timeout=0.15)
        except queue.Empty:
            continue

        if etype == 'token':
            if model != last_model:
                if last_model is not None:
                    sys.stdout.write('\n')
                sys.stdout.write(f"  {color}{emoji} ")
                last_model = model
            sys.stdout.write(data)
            sys.stdout.flush()
            responses[model] += data

        elif etype == 'tool':
            if last_model is not None:
                sys.stdout.write('\n')
            sys.stdout.write(f"  {color}{emoji} {CT}🔍 {data}{C}\n")
            last_model = None

        elif etype == 'error':
            if last_model is not None:
                sys.stdout.write('\n')
            sys.stdout.write(f"  {color}{emoji} \033[1;31m{data}{C}\n")
            done += 1
            last_model = None

        elif etype == 'turn_done':
            if last_model == model:
                sys.stdout.write('\n')
            done += 1
            last_model = None

    if last_model is not None:
        sys.stdout.write('\n')

    return dict(responses)


# ── Main Roundtable ──────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--max', type=int, default=3)
    args = parser.parse_args()

    print(f"\n{CB}{'━' * 65}")
    print(f"  🪞 Neural Mirror — Roundtable")
    print(f"  Models co-learn by examining themselves and each other")
    print(f"{'━' * 65}{C}\n")

    # Discover
    all_models = discover_models()

    if args.models:
        selected = [m for m in all_models if any(f in m['name'] for f in args.models)]
    else:
        preferred = ['qwen3:1.7b', 'gemma4', 'qwen2.5-coder']
        selected = []
        for p in preferred:
            for m in all_models:
                if p in m['name'] and m not in selected:
                    selected.append(m)
                    break
        if not selected:
            selected = all_models[:args.max]

    selected = selected[:args.max]

    # Profile
    participants = []
    for i, m in enumerate(selected):
        stats = profile_model(m['path'])
        if stats:
            color, emoji = PALETTE[i % len(PALETTE)]
            participants.append({
                'ollama_name': m['name'],
                'path': m['path'],
                'stats': stats,
                'color': color,
                'emoji': emoji,
            })
            print(f"  {color}{emoji} {stats['model_name']:<25s}{C} {stats['params']:>6s}  {stats['arch']:<8s}  {stats['n_layers']} layers")

    if not participants:
        print("No valid models!")
        return

    # Build roster
    roster = "Models at this roundtable:\n"
    for p in participants:
        s = p['stats']
        roster += f"  - {s['model_name']} ({s['params']}, {s['arch']}, {s['n_layers']} layers)\n"

    # State: accumulated findings per round
    round_findings = {}  # round -> {model_name: response}

    t_start = time.time()

    for round_num in range(1, args.rounds + 1):
        print(f"\n{CB}{'━' * 65}")
        if round_num == 1:
            print(f"  Round {round_num}: Self-Examination")
            print(f"  Each model inspects its own weights")
        elif round_num == 2:
            print(f"  Round {round_num}: Cross-Pollination")
            print(f"  Each model reads the others' findings and digs deeper")
        else:
            print(f"  Round {round_num}: Synthesis")
            print(f"  What did we learn together that none could alone?")
        print(f"{'━' * 65}{C}\n")

        output_q = queue.Queue()
        threads = []

        for p in participants:
            s = p['stats']

            base_system = f"""You are {s['model_name']} ({s['params']}, {s['arch']} architecture, {s['n_layers']} layers).

You have introspection tools to examine your own weights:
- inspect_self() — no arguments
- inspect_layer(layer_num) — 0 to {s['max_layer']}
- compare_layers(layer_a, layer_b)
- weight_fingerprint() — no arguments
- inspect_tensor(tensor_name)
- list_tensors(filter_str)

{roster}

RULES:
- inspect_self and weight_fingerprint take NO arguments
- No <think> tags — respond directly
- Be concise: 2-4 focused paragraphs
- Reference specific numbers from your tools
- When discussing others' findings, say their name"""

            if round_num == 1:
                prompt = (
                    "Examine yourself. Use inspect_self and compare_layers with 0 and your last layer. "
                    "Report your key findings: architecture, norm growth pattern, anything surprising about your weights. "
                    "The other models will read your findings next round."
                )
            elif round_num == 2:
                # Include round 1 findings from all models
                others_findings = ""
                for name, finding in round_findings.get(1, {}).items():
                    if name != p['ollama_name']:
                        others_findings += f"\n--- {name} reported ---\n{finding[:600]}\n"
                    else:
                        others_findings += f"\n--- Your own Round 1 findings ---\n{finding[:400]}\n"

                prompt = (
                    f"Here is what every model found in Round 1:\n{others_findings}\n\n"
                    "Now dig deeper. Pick something another model found that's DIFFERENT from you. "
                    "Use your tools to investigate: why is your architecture different? "
                    "What does the contrast reveal about how different models process information? "
                    "Reference specific numbers."
                )
            else:
                # Include all prior rounds
                all_prior = ""
                for r in range(1, round_num):
                    all_prior += f"\n=== ROUND {r} ===\n"
                    for name, finding in round_findings.get(r, {}).items():
                        all_prior += f"\n[{name}]: {finding[:400]}\n"

                prompt = (
                    f"Here is everything discovered so far:\n{all_prior}\n\n"
                    "Final synthesis: What patterns emerged across ALL models that no single model could see alone? "
                    "What is universal about transformer weight structure vs. what is unique to each architecture? "
                    "What is the most surprising cross-model discovery? "
                    "Be specific and reference the actual numbers from the discussion."
                )

            t = threading.Thread(
                target=run_turn,
                args=(p['ollama_name'], p['path'], base_system, prompt,
                      p['color'], p['emoji'], output_q),
                daemon=True,
            )
            threads.append((p['ollama_name'], t))

        # Start all
        for _, t in threads:
            t.start()

        # Stream display
        responses = stream_display(output_q, participants, len(participants))

        # Save round findings
        round_findings[round_num] = responses

        # Wait for threads
        for _, t in threads:
            t.join(timeout=5)

        # Brief summary
        print(f"\n  {CD}Round {round_num} complete — {len(responses)} models responded{C}")

    elapsed = time.time() - t_start

    # Final summary
    print(f"\n{CB}{'━' * 65}")
    print(f"  🪞 Roundtable Complete — {elapsed:.0f}s across {args.rounds} rounds")
    print(f"{'━' * 65}{C}\n")

    # Save transcript
    transcript = {
        'participants': [
            {'name': p['ollama_name'], 'model_name': p['stats']['model_name'],
             'params': p['stats']['params'], 'arch': p['stats']['arch']}
            for p in participants
        ],
        'rounds': {str(r): findings for r, findings in round_findings.items()},
        'elapsed_seconds': round(elapsed, 1),
    }

    out_path = os.path.expanduser('~/neural-mirror/roundtable_transcript.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"  Transcript saved to {out_path}")
    except:
        pass


if __name__ == '__main__':
    main()
