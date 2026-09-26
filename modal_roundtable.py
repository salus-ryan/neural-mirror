"""
Neural Mirror — Roundtable on Modal 🪞🔄

Co-learning between models, running on cloud GPUs.
Each round, models see what others found and build on it.

Usage:
  modal run modal_roundtable.py
  modal run modal_roundtable.py --rounds 4
"""

import modal
import json, time, os, sys

app = modal.App("neural-mirror-roundtable")

model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "procps")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("gguf")
    .add_local_file("introspect.py", "/app/introspect.py")
)

# Models for the roundtable — diverse architectures
ROUNDTABLE_MODELS = [
    "qwen3:1.7b",
    "gemma3:4b",
    "llama3.2:3b",
    "phi4-mini",
    "mistral:7b",
]


def start_ollama():
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    proc = subprocess.Popen(["ollama", "serve"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return proc
        except:
            time.sleep(1)
    raise RuntimeError("Ollama failed to start")


def pull_model(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    r = subprocess.run(["ollama", "pull", name], env=env,
                      capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"Pull failed: {r.stderr}")


def find_gguf(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    r = subprocess.run(["ollama", "show", name, "--modelfile"],
                      env=env, capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if line.startswith('FROM /'):
            path = line[5:].strip()
            if os.path.isfile(path):
                return path
    return None


def profile_gguf(path):
    sys.path.insert(0, "/app")
    from introspect import GGUFModel
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
    # Quick norm growth
    first_norms, last_norms = [], []
    max_layer = max(layers) if layers else 0
    for tname, tinfo in m.tensors.items():
        if 'blk.0.' in tname and 'norm' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=128)
            if vals: first_norms.append(sum(vals)/len(vals))
        if f'blk.{max_layer}.' in tname and 'norm' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=128)
            if vals: last_norms.append(sum(vals)/len(vals))
    fn = sum(first_norms)/len(first_norms) if first_norms else 0
    ln = sum(last_norms)/len(last_norms) if last_norms else 0
    ng = round(ln/fn, 2) if fn else None

    return {
        'model_name': name, 'arch': arch,
        'params': f"{params/1e9:.2f}B",
        'n_layers': len(layers), 'max_layer': max_layer,
        'norm_growth': ng,
        'first_norm': round(fn, 4), 'last_norm': round(ln, 4),
    }


def run_model_turn(model_name, gguf_path, system_prompt, user_prompt, host="http://localhost:11434"):
    """One model's turn with tool calling."""
    sys.path.insert(0, "/app")
    from introspect import GGUFModel, call_tool, build_ollama_tools
    import introspect

    local_model = GGUFModel(gguf_path)
    introspect._model = local_model
    tools = build_ollama_tools()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tool_log = []
    full_response = ""

    for _ in range(6):
        payload = {
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        import urllib.request
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())

        msg = result.get('message', {})
        messages.append(msg)

        tool_calls = msg.get('tool_calls', [])
        content = msg.get('content', '').replace('<think>', '').replace('</think>', '').strip()

        if content:
            full_response += content

        if tool_calls:
            for tc in tool_calls:
                fn = tc['function']['name']
                args = tc['function'].get('arguments', {})
                tool_log.append(f"{fn}({json.dumps(args)})")

                introspect._model = local_model
                tr = call_tool(fn, args)
                messages.append({"role": "tool", "content": json.dumps(tr)})
            continue
        break

    return full_response, tool_log


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
    memory=16384,
    cpu=4,
)
def run_round_for_model(
    model_name: str,
    round_num: int,
    profile: dict,
    roster: str,
    prior_findings: dict,
) -> dict:
    """Run one model's turn in one round."""

    proc = start_ollama()

    # Pull model
    pull_model(model_name)
    gguf_path = find_gguf(model_name)
    if not gguf_path:
        return {"model": model_name, "error": "No GGUF found"}

    # Warm up
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
    except:
        pass

    s = profile
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
- Be concise: 2-3 focused paragraphs with specific numbers
- When discussing others' findings, name them"""

    if round_num == 1:
        prompt = (
            "Examine yourself. Use inspect_self then compare_layers 0 and your last layer. "
            "Report your key findings: norm growth, weight distributions, anything surprising. "
            "The other models will read this next round."
        )
    elif round_num == 2:
        others = ""
        for name, finding in prior_findings.items():
            label = "YOUR Round 1" if name == model_name else f"{name}'s Round 1"
            others += f"\n--- {label} ---\n{finding[:800]}\n"
        prompt = (
            f"Round 1 findings from all models:\n{others}\n\n"
            "Now dig deeper. Pick the most interesting CONTRAST between your weights and another model's. "
            "Use your tools to investigate WHY. What does the difference reveal? "
            "Reference specific numbers from both your tools and their findings."
        )
    else:
        all_prior = ""
        for r, findings in sorted(prior_findings.items()):
            if isinstance(r, int) or r.isdigit():
                all_prior += f"\n=== ROUND {r} ===\n"
                if isinstance(findings, dict):
                    for name, f in findings.items():
                        all_prior += f"\n[{name}]: {f[:500]}\n"
                else:
                    all_prior += str(findings)[:1000]

        prompt = (
            f"Everything discovered:\n{all_prior}\n\n"
            "Final synthesis: What universal patterns exist across ALL these different architectures? "
            "What is unique to each? What is the most surprising cross-model finding? "
            "What would you tell a human researcher about what these weight comparisons reveal?"
        )

    response, tools_used = run_model_turn(model_name, gguf_path, base_system, prompt)

    proc.terminate()

    return {
        "model": model_name,
        "model_name": s['model_name'],
        "round": round_num,
        "response": response,
        "tools_used": tools_used,
    }


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=120,
    memory=8192,
)
def profile_model_remote(model_name: str) -> dict:
    """Pull and profile a model."""
    proc = start_ollama()
    pull_model(model_name)
    path = find_gguf(model_name)
    if not path:
        proc.terminate()
        return {"model": model_name, "error": "No GGUF"}
    p = profile_gguf(path)
    p['ollama_name'] = model_name
    proc.terminate()
    return p


@app.local_entrypoint()
def main(rounds: int = 3, models: str = None):
    """Run the full roundtable on Modal."""

    MODELS = models.split(',') if models else ROUNDTABLE_MODELS

    print("━" * 65)
    print("  🪞 Neural Mirror — Roundtable on Modal")
    print("  Models co-learn by examining themselves and each other")
    print("━" * 65)
    print()

    # Phase 0: Profile all models in parallel
    print("📋 Profiling models...")
    profile_futures = [profile_model_remote.spawn(m) for m in MODELS]
    profiles = {}
    for m, fut in zip(MODELS, profile_futures):
        try:
            p = fut.get()
            if 'error' not in p:
                profiles[m] = p
                ng = p.get('norm_growth', '?')
                print(f"  ✅ {p['model_name']:<25s} {p['params']:>6s} {p['arch']:<8s} {p['n_layers']}L  norm:{ng}x")
            else:
                print(f"  ❌ {m}: {p['error']}")
        except Exception as e:
            print(f"  ❌ {m}: {e}")

    active_models = list(profiles.keys())
    if len(active_models) < 2:
        print("Need at least 2 models!")
        return

    # Build roster
    roster = "Models at this roundtable:\n"
    for m, p in profiles.items():
        roster += f"  - {p['model_name']} ({p['params']}, {p['arch']}, {p['n_layers']}L, norm growth: {p.get('norm_growth','?')}x)\n"

    print(f"\n  Roster: {len(active_models)} models ready\n")

    round_findings = {}  # round_num -> {model_name: response}
    t_start = time.time()

    for round_num in range(1, rounds + 1):
        print("━" * 65)
        labels = {1: "Self-Examination", 2: "Cross-Pollination", 3: "Synthesis"}
        label = labels.get(round_num, f"Round {round_num}")
        print(f"  Round {round_num}: {label}")
        print("━" * 65)
        print()

        # Build prior findings for this round
        if round_num == 1:
            prior = {}
        elif round_num == 2:
            prior = round_findings.get(1, {})
        else:
            # Flatten all prior rounds
            prior = {}
            for r in range(1, round_num):
                prior[str(r)] = round_findings.get(r, {})

        # Launch all models in parallel
        futures = {}
        for m in active_models:
            fut = run_round_for_model.spawn(
                m, round_num, profiles[m], roster, prior
            )
            futures[m] = fut

        # Collect results
        this_round = {}
        for m, fut in futures.items():
            p = profiles[m]
            try:
                result = fut.get()
                response = result.get('response', '')
                tools = result.get('tools_used', [])
                this_round[m] = response

                print(f"  {'─' * 60}")
                mname = p['model_name']
                print(f"  🪞 {mname} ({p['params']}, {p['arch']})")
                if tools:
                    print(f"     🔍 {', '.join(tools[:5])}")
                print()
                # Print response with wrapping
                for line in response.split('\n'):
                    line = line.strip()
                    if line:
                        wrapped = line[:120]
                        print(f"     {wrapped}")
                        if len(line) > 120:
                            for i in range(120, len(line), 120):
                                print(f"     {line[i:i+120]}")
                print()

            except Exception as e:
                print(f"  ❌ {m}: {e}")
                this_round[m] = f"Error: {e}"

        round_findings[round_num] = this_round
        print(f"  ✅ Round {round_num} complete\n")

    elapsed = time.time() - t_start

    # Final summary
    print("━" * 65)
    print(f"  🪞 Roundtable Complete — {elapsed:.0f}s, {rounds} rounds, {len(active_models)} models")
    print("━" * 65)
    print()

    # Print the synthesis round highlights
    if rounds >= 3 and 3 in round_findings:
        print("  💡 Synthesis Highlights:")
        print()
        for m, resp in round_findings[3].items():
            p = profiles.get(m, {})
            print(f"  {p.get('model_name', m)}:")
            preview = resp[:300].replace('\n', ' ').strip()
            print(f"  {preview}...")
            print()

    # Save transcript
    transcript = {
        'participants': {m: profiles[m] for m in active_models},
        'rounds': {str(r): f for r, f in round_findings.items()},
        'elapsed': round(elapsed, 1),
    }
    out = os.path.expanduser('~/neural-mirror/roundtable_transcript.json')
    try:
        with open(out, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"  Transcript: {out}")
    except:
        print(json.dumps(transcript, indent=2)[:2000])
