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

# Small deterministic capability probe. Each answer is scored for correctness;
# exact-string change is retained only as a sensitivity measure, never quality.
TEST_CASES = {
    "math": [
        {"prompt": "What is 17 + 38?", "answers": ["55"]},
        {"prompt": "What is 7 * 13?", "answers": ["91"]},
        {"prompt": "If I have 100 apples and give away 37, how many remain?", "answers": ["63"]},
    ],
    "reasoning": [
        {
            "prompt": "If all roses are flowers and some flowers are red, can we conclude all roses are red?",
            "answers": ["no", "cannot conclude", "not necessarily"],
        },
        {
            "prompt": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
            "answers": ["$0.05", "0.05", "5 cents", "five cents"],
        },
    ],
    "language": [
        {"prompt": "Translate 'hello world' to French.", "answers": ["bonjour le monde"]},
        {"prompt": "What is the opposite of 'ancient'?", "answers": ["modern"]},
        {"prompt": "Complete: The quick brown fox jumps over the lazy ___", "answers": ["dog"]},
    ],
    "knowledge": [
        {"prompt": "What is the capital of Japan?", "answers": ["tokyo"]},
        {"prompt": "Who wrote Romeo and Juliet?", "answers": ["shakespeare"]},
        {"prompt": "What is the boiling point of water in Celsius?", "answers": ["100"]},
    ],
}


def answer_is_correct(text, accepted):
    """Conservative answer-key match with numeric word boundaries."""
    import re
    normalized = " ".join(text.lower().replace("’", "'").split())
    for answer in accepted:
        answer = answer.lower()
        if answer.replace(".", "", 1).isdigit():
            if re.search(rf"(?<![\d.]){re.escape(answer)}(?!\d|\.\d)", normalized):
                return True
        elif answer in normalized:
            return True
    return False


def generate(model, tokenizer, prompt, max_tokens=96):
    """Run instruct inference and decode only newly generated tokens."""
    import torch
    messages = [
        {"role": "system", "content": "Answer the question directly and concisely."},
        {"role": "user", "content": prompt},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
    except (TypeError, ValueError):
        try:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
        except (AttributeError, TypeError, ValueError):
            inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


def run_test_suite(model, tokenizer, categories=None):
    """Run capability probes and score each response against its answer key."""
    results = {}
    cats = categories or list(TEST_CASES.keys())
    for cat in cats:
        results[cat] = {}
        for case in TEST_CASES.get(cat, []):
            prompt = case["prompt"]
            try:
                output = generate(model, tokenizer, prompt)
                results[cat][prompt] = {
                    "output": output,
                    "correct": answer_is_correct(output, case["answers"]),
                    "accepted": case["answers"],
                }
            except Exception as e:
                results[cat][prompt] = {
                    "output": f"[ERROR: {e}]", "correct": False,
                    "accepted": case["answers"],
                }
    return results


def compare_outputs(baseline, ablated):
    """Measure answer accuracy separately from output sensitivity."""
    from difflib import SequenceMatcher

    total = changed = degraded = improved = 0
    baseline_correct = ablated_correct = 0
    similarities = []
    details = {}

    for cat in baseline:
        details[cat] = {}
        for prompt, baseline_item in baseline.get(cat, {}).items():
            ablated_item = ablated.get(cat, {}).get(
                prompt, {"output": "[MISSING]", "correct": False}
            )
            b = baseline_item["output"]
            a = ablated_item["output"]
            b_ok = bool(baseline_item["correct"])
            a_ok = bool(ablated_item["correct"])
            total += 1
            baseline_correct += int(b_ok)
            ablated_correct += int(a_ok)

            similarity = SequenceMatcher(None, b, a).ratio()
            similarities.append(similarity)
            is_changed = a != b
            changed += int(is_changed)

            if b_ok and not a_ok:
                status = "DEGRADED"
                degraded += 1
            elif not b_ok and a_ok:
                status = "IMPROVED"
                improved += 1
            elif is_changed:
                status = "CHANGED_CORRECT" if a_ok else "CHANGED_INCORRECT"
            else:
                status = "UNCHANGED"

            details[cat][prompt] = {
                "status": status,
                "baseline_correct": b_ok,
                "ablated_correct": a_ok,
                "similarity": round(similarity, 3),
                "baseline": b[:200],
                "ablated": a[:200],
            }

    baseline_accuracy = baseline_correct / total * 100 if total else 0
    ablated_accuracy = ablated_correct / total * 100 if total else 0
    return {
        "total_prompts": total,
        "changed": changed,
        "degraded": degraded,
        "improved": improved,
        "change_pct": round(changed / total * 100, 1) if total else 0,
        "degrade_pct": round(degraded / total * 100, 1) if total else 0,
        "baseline_accuracy": round(baseline_accuracy, 1),
        "ablated_accuracy": round(ablated_accuracy, 1),
        "accuracy_drop_pp": round(baseline_accuracy - ablated_accuracy, 1),
        "mean_text_similarity": round(sum(similarities) / len(similarities), 3) if similarities else 0,
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
        print(
            f"     Restored. Accuracy: {comparison['baseline_accuracy']}% → "
            f"{comparison['ablated_accuracy']}% "
            f"(Δ {-comparison['accuracy_drop_pp']:+.1f} pp), "
            f"changed: {comparison['change_pct']}% ({ablate_time:.1f}s)"
        )

        ablation_results[tensor_name] = {
            "param_info": param_info,
            "comparison": comparison,
            "time": round(ablate_time, 1),
        }

    # Phase 3: Summary
    print("\n   📋 Ablation Summary:")
    ranked = sorted(
        [(t, r) for t, r in ablation_results.items() if 'comparison' in r],
        key=lambda x: (
            x[1]['comparison']['accuracy_drop_pp'],
            1 - x[1]['comparison']['mean_text_similarity'],
        ),
        reverse=True,
    )

    for tensor_name, result in ranked:
        c = result['comparison']
        p = result['param_info']
        impact = "🔴 CRITICAL" if c['accuracy_drop_pp'] > 50 else "🟡 MODERATE" if c['accuracy_drop_pp'] > 10 else "🟢 MINOR"
        print(
            f"     {impact} {tensor_name}: accuracy {c['baseline_accuracy']}% → "
            f"{c['ablated_accuracy']}% (drop {c['accuracy_drop_pp']} pp), "
            f"{c['change_pct']}% text changed ({p['numel']:,} params)"
        )

    return {
        "model": model_name,
        "hf_model": hf_name,
        "methodology": {
            "version": 2,
            "prompt_format": "native chat template",
            "generation": "greedy",
            "quality_metric": "answer-key accuracy",
            "sensitivity_metric": "exact-string change plus text similarity",
            "warning": "11-prompt pilot; full-tensor zeroing is an extreme intervention",
        },
        "total_params": total_params,
        "baseline_prompts": sum(len(v) for v in TEST_CASES.values()),
        "tensors_ablated": len(tensors_to_ablate),
        "baseline": baseline,
        "ablations": ablation_results,
        "ranked_by_impact": [
            {
                "tensor": t,
                "change_pct": r['comparison']['change_pct'],
                "degrade_pct": r['comparison']['degrade_pct'],
                "baseline_accuracy": r['comparison']['baseline_accuracy'],
                "ablated_accuracy": r['comparison']['ablated_accuracy'],
                "accuracy_drop_pp": r['comparison']['accuracy_drop_pp'],
                "mean_text_similarity": r['comparison']['mean_text_similarity'],
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
    report += f"Test prompts: {ablation_results.get('baseline_prompts', 0)} across math, reasoning, language, knowledge\n"
    report += "Scoring: native chat template, greedy decoding, answer-key correctness\n"
    report += "Caveat: small pilot with extreme full-tensor zeroing; no pruning conclusions\n\n"
    report += "RESULTS (ranked by answer-accuracy drop, most damaging first):\n\n"

    for r in ranked:
        t = r['tensor']
        drop = r.get('accuracy_drop_pp', 0)
        report += f"{'🔴' if drop > 50 else '🟡' if drop > 10 else '🟢'} {t}\n"
        report += (
            f"   answer accuracy: {r.get('baseline_accuracy')}% → "
            f"{r.get('ablated_accuracy')}% (drop {drop} percentage points)\n"
        )
        report += (
            f"   output sensitivity: {r['change_pct']}% exact strings changed; "
            f"mean text similarity {r.get('mean_text_similarity')}\n"
        )
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

Analyze these results. Most important finding FIRST:
1. Which tested intervention caused the largest answer-accuracy drop?
2. Does layer depth predict damage for the same module type?
3. Which changes affected wording while preserving correctness?
4. What controlled follow-up would distinguish unique tensor importance from generic disruption?

Do NOT infer that a tensor is dispensable from this one small test, and do not
recommend pruning. Distinguish measured results from hypotheses. Reference full
tensor names and actual numbers. No <think> tags."""

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
    print(f"  {'Tensor':<48s} {'Accuracy':>9s} {'Drop':>8s} {'Changed':>8s}")
    print(f"  {'─'*48} {'─'*9} {'─'*8} {'─'*8}")

    for r in results.get('ranked_by_impact', []):
        tensor = r['tensor'].replace('model.layers.', 'L')
        drop = r.get('accuracy_drop_pp', 0)
        icon = "🔴" if drop > 50 else "🟡" if drop > 10 else "🟢"
        accuracy = f"{r.get('ablated_accuracy', 0):.1f}%"
        print(f"  {icon} {tensor:<46.46s} {accuracy:>9s} {drop:>7.1f}p {r['change_pct']:>7.1f}%")
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
                    print(f"\n  🔬 Ablated: {r['tensor']}")
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
