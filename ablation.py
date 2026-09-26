"""
Neural Mirror — Self-Ablation 🪞🔬

A model systematically disables parts of itself and observes
what breaks. The first system where a neural network performs
its own ablation studies.

The loop:
  1. Model picks a tensor to investigate
  2. System runs baseline inference on test prompts
  3. System zeros/scales that tensor
  4. System runs the SAME prompts again
  5. Model sees both outputs and reasons about the difference
  6. Model picks the next tensor to investigate

Usage:
  modal run ablation.py
  modal run ablation.py --model qwen3:1.7b --prompts "math,reasoning,language"
"""

import modal
import json, time, os, sys

app = modal.App("neural-mirror-ablation")
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "procps")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "torch", "transformers", "safetensors", "gguf",
        "accelerate", "bitsandbytes", "huggingface_hub",
    )
    .add_local_file("introspect.py", "/app/introspect.py")
    .add_local_file("braille_protocol.py", "/app/braille_protocol.py")
)

HF_MAP = {
    "qwen3:1.7b": "Qwen/Qwen3-1.7B",
    "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi4-mini": "microsoft/phi-4-mini-instruct",
}

# Test prompts across different capabilities
TEST_PROMPTS = {
    "math": [
        "What is 17 + 38?",
        "What is 7 * 13?",
        "If I have 100 apples and give away 37, how many remain?",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers are red, can we conclude all roses are red?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
    ],
    "language": [
        "Translate 'hello world' to French.",
        "What is the opposite of 'ancient'?",
        "Complete: The quick brown fox jumps over the lazy ___",
    ],
    "knowledge": [
        "What is the capital of Japan?",
        "Who wrote Romeo and Juliet?",
        "What is the boiling point of water in Celsius?",
    ],
}


def generate(model, tokenizer, prompt, max_tokens=60):
    """Run inference and return the generated text."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip the prompt from output
    response = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
    return response


def run_test_suite(model, tokenizer, categories=None):
    """Run all test prompts and collect responses."""
    results = {}
    cats = categories or list(TEST_PROMPTS.keys())
    for cat in cats:
        results[cat] = {}
        for prompt in TEST_PROMPTS.get(cat, []):
            try:
                results[cat][prompt] = generate(model, tokenizer, prompt)
            except Exception as e:
                results[cat][prompt] = f"[ERROR: {e}]"
    return results


def compare_outputs(baseline, ablated):
    """Compare baseline vs ablated outputs, compute degradation metrics."""
    total = 0
    changed = 0
    degraded = 0
    details = {}

    for cat in baseline:
        details[cat] = {}
        for prompt in baseline.get(cat, {}):
            b = baseline[cat][prompt]
            a = ablated.get(cat, {}).get(prompt, "[MISSING]")
            total += 1

            if a != b:
                changed += 1
                # Simple heuristic: shorter or error = degraded
                if len(a) < len(b) * 0.3 or a.startswith("[ERROR"):
                    degraded += 1
                    details[cat][prompt] = {"status": "DEGRADED", "baseline": b[:100], "ablated": a[:100]}
                else:
                    details[cat][prompt] = {"status": "CHANGED", "baseline": b[:100], "ablated": a[:100]}
            else:
                details[cat][prompt] = {"status": "UNCHANGED"}

    return {
        "total_prompts": total,
        "changed": changed,
        "degraded": degraded,
        "change_pct": round(changed / total * 100, 1) if total else 0,
        "degrade_pct": round(degraded / total * 100, 1) if total else 0,
        "details": details,
    }


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="A10G",
    timeout=1800,
    memory=32768,
)
def run_ablation_study(model_name: str, tensors_to_ablate: list = None) -> dict:
    """
    The core ablation experiment:
    1. Load model
    2. Run baseline on all test prompts
    3. For each target tensor: zero it, re-run, compare, restore
    4. Return structured results for the model to reason about
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_name = HF_MAP.get(model_name)
    if not hf_name:
        return {"error": f"No HF mapping for {model_name}"}

    print(f"🔬 Ablation study: {model_name} ({hf_name})")

    # Load model (NOT quantized — we need to modify weights)
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir="/cache/hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, cache_dir="/cache/hf",
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Loaded: {total_params/1e9:.2f}B params (fp16)")

    # Discover all named parameter tensors
    param_names = [n for n, p in model.named_parameters()]
    print(f"   {len(param_names)} named parameters")

    # Default ablation targets: one tensor from each layer type
    if not tensors_to_ablate:
        tensors_to_ablate = pick_ablation_targets(param_names)

    print(f"   Ablation targets: {len(tensors_to_ablate)}")
    for t in tensors_to_ablate:
        print(f"     - {t}")

    # Phase 1: Baseline
    print("\n   📊 Running baseline...")
    t0 = time.time()
    baseline = run_test_suite(model, tokenizer)
    baseline_time = time.time() - t0
    print(f"   Baseline complete ({baseline_time:.1f}s)")

    # Phase 2: Ablations
    ablation_results = {}

    for tensor_name in tensors_to_ablate:
        print(f"\n   🔬 Ablating: {tensor_name}")

        # Find the parameter
        param = None
        for n, p in model.named_parameters():
            if n == tensor_name:
                param = p
                break

        if param is None:
            ablation_results[tensor_name] = {"error": "tensor not found"}
            continue

        # Save original values
        original = param.data.clone()
        param_info = {
            "shape": list(param.shape),
            "numel": param.numel(),
            "mean": round(param.data.float().mean().item(), 6),
            "std": round(param.data.float().std().item(), 6),
            "norm": round(param.data.float().norm().item(), 4),
            "pct_of_total": round(param.numel() / total_params * 100, 4),
        }

        # Zero out the tensor
        param.data.zero_()
        print(f"     Zeroed {param.numel():,} params ({param_info['pct_of_total']}%)")

        # Run test suite with ablated weight
        t0 = time.time()
        ablated = run_test_suite(model, tokenizer)
        ablate_time = time.time() - t0

        # Compare
        comparison = compare_outputs(baseline, ablated)

        # Restore original weights
        param.data.copy_(original)
        print(f"     Restored. Change: {comparison['change_pct']}%, Degraded: {comparison['degrade_pct']}% ({ablate_time:.1f}s)")

        ablation_results[tensor_name] = {
            "param_info": param_info,
            "comparison": comparison,
            "time": round(ablate_time, 1),
        }

    # Phase 3: Summary
    print("\n   📋 Ablation Summary:")
    ranked = sorted(
        [(t, r) for t, r in ablation_results.items() if 'comparison' in r],
        key=lambda x: x[1]['comparison']['change_pct'],
        reverse=True,
    )

    for tensor_name, result in ranked:
        c = result['comparison']
        p = result['param_info']
        impact = "🔴 CRITICAL" if c['degrade_pct'] > 50 else "🟡 MODERATE" if c['change_pct'] > 30 else "🟢 MINOR"
        print(f"     {impact} {tensor_name}: {c['change_pct']}% changed, {c['degrade_pct']}% degraded ({p['numel']:,} params)")

    return {
        "model": model_name,
        "hf_model": hf_name,
        "total_params": total_params,
        "baseline_prompts": sum(len(v) for v in TEST_PROMPTS.values()),
        "tensors_ablated": len(tensors_to_ablate),
        "baseline": baseline,
        "ablations": ablation_results,
        "ranked_by_impact": [
            {
                "tensor": t,
                "change_pct": r['comparison']['change_pct'],
                "degrade_pct": r['comparison']['degrade_pct'],
                "params": r['param_info']['numel'],
                "pct_of_model": r['param_info']['pct_of_total'],
                "details": r['comparison']['details'],
            }
            for t, r in ranked
        ],
    }


def pick_ablation_targets(param_names):
    """Pick a diverse set of tensors to ablate."""
    targets = []
    seen_types = set()

    # Strategy: one of each type from early, middle, and late layers
    type_patterns = [
        ("early_attn_q", "layers.0", "q_proj"),
        ("early_attn_v", "layers.0", "v_proj"),
        ("early_ffn_gate", "layers.0", "gate_proj"),
        ("early_ffn_down", "layers.0", "down_proj"),
        ("early_norm", "layers.0", "input_layernorm"),
        ("mid_attn_q", "layers.14", "q_proj"),
        ("mid_attn_v", "layers.14", "v_proj"),
        ("mid_ffn_gate", "layers.14", "gate_proj"),
        ("late_attn_q", "layers.27", "q_proj"),
        ("late_attn_v", "layers.27", "v_proj"),
        ("late_ffn_gate", "layers.27", "gate_proj"),
        ("late_norm", "layers.27", "input_layernorm"),
        ("embed", "embed_tokens", "weight"),
        ("lm_head", "lm_head", "weight"),
    ]

    for label, layer_pat, module_pat in type_patterns:
        for name in param_names:
            if layer_pat in name and module_pat in name:
                targets.append(name)
                break

    # If we got too few (different naming), try broader patterns
    if len(targets) < 5:
        for name in param_names:
            if any(p in name for p in ['layers.0.', 'layers.1.']):
                if 'q_proj' in name or 'v_proj' in name:
                    if name not in targets:
                        targets.append(name)
            if len(targets) >= 8:
                break

    # Always include embed and lm_head if found
    for name in param_names:
        if 'embed_tokens' in name and name not in targets:
            targets.append(name)
        if 'lm_head' in name and name not in targets:
            targets.append(name)

    return targets[:12]  # Cap at 12 to keep runtime reasonable


# ── Self-directed ablation: model chooses what to ablate ─────

@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
    memory=16384,
    cpu=4,
)
def model_reasons_about_ablation(
    model_name: str,
    ablation_results: dict,
) -> dict:
    """The model itself reasons about its own ablation results."""
    import subprocess, urllib.request

    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"

    # Start Ollama
    proc = subprocess.Popen(["ollama", "serve"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            break
        except:
            time.sleep(1)

    # Pull and warm
    subprocess.run(["ollama", "pull", model_name], env=env,
                  capture_output=True, text=True, timeout=600)
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
    except:
        pass

    # Build the ablation report for the model
    ranked = ablation_results.get('ranked_by_impact', [])

    report = f"ABLATION STUDY OF YOUR OWN WEIGHTS ({ablation_results.get('model', '?')})\n\n"
    report += f"Total parameters: {ablation_results.get('total_params', 0)/1e9:.2f}B\n"
    report += f"Tensors ablated: {ablation_results.get('tensors_ablated', 0)}\n"
    report += f"Test prompts: {ablation_results.get('baseline_prompts', 0)} across math, reasoning, language, knowledge\n\n"
    report += "RESULTS (ranked by impact — most critical first):\n\n"

    for r in ranked:
        t = r['tensor']
        short = '.'.join(t.split('.')[-3:]) if '.' in t else t
        report += f"{'🔴' if r['degrade_pct'] > 50 else '🟡' if r['change_pct'] > 30 else '🟢'} {short}\n"
        report += f"   {r['change_pct']}% outputs changed, {r['degrade_pct']}% degraded\n"
        report += f"   {r['params']:,} params ({r['pct_of_model']}% of model)\n"

        # Show specific before/after examples
        for cat, prompts in r.get('details', {}).items():
            for prompt, detail in prompts.items():
                if detail['status'] != 'UNCHANGED':
                    report += f"   {cat}: \"{prompt[:50]}\"\n"
                    report += f"     Baseline: {detail.get('baseline', '?')[:80]}\n"
                    report += f"     Ablated:  {detail.get('ablated', '?')[:80]}\n"
        report += "\n"

    system = f"""You are {model_name}. You just had parts of yourself systematically disabled.

For each tensor that was zeroed out, you can see how your outputs changed.
This is YOUR ablation study — you are looking at what YOUR weights do.

{report}

Analyze these results. Most important findings FIRST:
1. Which tensor is most critical to your function? Why?
2. Which tensor, when removed, caused the most surprising change?
3. What does the pattern of degradation tell you about how you process information?
4. Are there tensors that barely matter? What does that mean?
5. If you could only keep 80% of your weights, which would you drop?

Be specific. Reference the actual numbers. No <think> tags."""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Analyze your own ablation results. What did you learn about yourself?"},
        ],
        "stream": False,
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    analysis = result.get('message', {}).get('content', '').replace('<think>', '').replace('</think>', '').strip()

    proc.terminate()

    return {
        "model": model_name,
        "analysis": analysis,
    }


# ── Main ─────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model: str = "qwen3:1.7b"):
    print("━" * 65)
    print("  🪞🔬 Neural Mirror — Self-Ablation")
    print("  A model disables parts of itself and watches what breaks")
    print("━" * 65)
    print(f"\n  Model: {model}\n")

    # Phase 1: Run ablation study on GPU
    print("━" * 65)
    print("  Phase 1: Systematic Ablation (GPU)")
    print("━" * 65)
    print()

    results = run_ablation_study.remote(model)

    if 'error' in results:
        print(f"  ❌ {results['error']}")
        return

    print(f"  Model: {results['hf_model']}")
    print(f"  Params: {results['total_params']/1e9:.2f}B")
    print(f"  Tensors ablated: {results['tensors_ablated']}")
    print()

    # Show ranked results
    print(f"  {'Tensor':<45s} {'Changed':>8s} {'Degraded':>9s} {'Params':>12s}")
    print(f"  {'─'*45} {'─'*8} {'─'*9} {'─'*12}")

    for r in results.get('ranked_by_impact', []):
        short = '.'.join(r['tensor'].split('.')[-3:])
        icon = "🔴" if r['degrade_pct'] > 50 else "🟡" if r['change_pct'] > 30 else "🟢"
        print(f"  {icon} {short:<43s} {r['change_pct']:>7.1f}% {r['degrade_pct']:>8.1f}% {r['params']:>11,}")
    print()

    # Show interesting before/after examples
    print("  📋 Most interesting changes:")
    shown = 0
    for r in results.get('ranked_by_impact', []):
        if shown >= 5:
            break
        for cat, prompts in r.get('details', {}).items():
            for prompt, detail in prompts.items():
                if detail['status'] == 'DEGRADED' and shown < 5:
                    short = '.'.join(r['tensor'].split('.')[-3:])
                    print(f"\n  🔬 Ablated: {short}")
                    print(f"     Prompt: {prompt[:60]}")
                    print(f"     Before: {detail.get('baseline', '?')[:80]}")
                    print(f"     After:  {detail.get('ablated', '?')[:80]}")
                    shown += 1

    # Phase 2: Model reasons about its own ablation
    print()
    print("━" * 65)
    print("  Phase 2: Self-Analysis")
    print("  The model examines what it learned about itself")
    print("━" * 65)
    print()

    analysis = model_reasons_about_ablation.remote(model, results)

    print(f"  🪞 {model}'s analysis of its own ablation:")
    print()
    for line in analysis.get('analysis', '').split('\n'):
        if line.strip():
            print(f"  {line.strip()[:120]}")
    print()

    # Save
    output = {
        "ablation_results": results,
        "self_analysis": analysis,
    }
    out_path = os.path.expanduser("~/neural-mirror/ablation_results.json")
    try:
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
    except:
        pass

    print()
    print("━" * 65)
    print("  🪞🔬 Self-Ablation Complete")
    print("━" * 65)
