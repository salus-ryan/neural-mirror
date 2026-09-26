"""Cross-family causal mechanism study for Neural Mirror.

This study does not ask models to vote a mechanism into existence. It applies
one functionally matched intervention to homologous modules and evaluates the
same examples in every model.

Hypothesis H1 (shared sensitivity): changing pre-attention normalization output
amplitude changes decision margins in every tested family.

Hypothesis H2 (shared depth mechanism): matched changes at the final relative
depth have a larger standardized effect than changes at the first depth.

Raw parameter means are not compared. The intervention is a dimensionless
multiplicative scale on module output; outcomes are accuracy change and choice-
margin change standardized by each model's baseline margin standard deviation.

Usage:
  MODAL_PROFILE=salus modal run mechanism_study.py
  MODAL_PROFILE=salus modal run mechanism_study.py --models 'qwen3:4b,phi4-mini,mistral:7b' --items 32
"""

import json
import math
import os
import random
import statistics
import time
from typing import Any

import modal

app = modal.App("neural-mirror-mechanism-study")
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers==4.57.6",
        "huggingface_hub<1.0",
        "accelerate",
        "datasets",
        "safetensors",
        "sentencepiece",
    )
)

MODEL_SPECS = {
    "qwen3:4b": {
        "family": "Qwen",
        "hf": "Qwen/Qwen3-4B",
    },
    "phi4-mini": {
        "family": "Phi",
        "hf": "microsoft/phi-4-mini-instruct",
    },
    "mistral:7b": {
        "family": "Mistral",
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "granite3.3:8b": {
        "family": "Granite",
        "hf": "ibm-granite/granite-3.3-8b-instruct",
    },
    "olmo2:7b": {
        "family": "OLMo",
        "hf": "allenai/OLMo-2-1124-7B-Instruct",
    },
}

DEFAULT_MODELS = ["qwen3:4b", "phi4-mini", "mistral:7b"]
DEPTH_FRACTIONS = {"early": 0.0, "middle": 0.5, "late": 1.0}
SCALES = [0.5, 0.75, 1.25, 1.5]
BOOTSTRAP_SAMPLES = 2000
STUDY_VERSION = 1


def _transformer_layers(model):
    """Resolve the decoder block list across common Transformers families."""
    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("model.decoder.layers", lambda m: getattr(getattr(getattr(m, "model", None), "decoder", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
    ]
    for path, getter in candidates:
        layers = getter(model)
        if layers is not None and len(layers):
            return path, layers
    raise RuntimeError("Could not locate transformer decoder layers")


def _pre_attention_norm(layer):
    """Resolve a block's normalization immediately before self-attention."""
    for name in (
        "input_layernorm",
        "self_attn_layer_norm",
        "attention_norm",
        "attn_norm",
        "ln_1",
    ):
        module = getattr(layer, name, None)
        if module is not None:
            return name, module
    raise RuntimeError(f"Could not locate pre-attention norm on {type(layer).__name__}")


def _replace_first(output, tensor):
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    return tensor


def _output_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _scaling_hook(scale):
    def hook(_module, _inputs, output):
        tensor = _output_tensor(output)
        return _replace_first(output, tensor * scale)
    return hook


def _rms_collector(storage):
    def hook(_module, inputs, output):
        import torch

        source = inputs[0]
        target = _output_tensor(output)
        with torch.no_grad():
            storage["input_sq_sum"] += source.float().square().sum().item()
            storage["input_count"] += source.numel()
            storage["output_sq_sum"] += target.float().square().sum().item()
            storage["output_count"] += target.numel()
    return hook


def _load_arc(items, seed=42):
    """Load a deterministic four-choice ARC-Challenge evaluation slice."""
    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="validation",
        cache_dir="/cache/datasets",
    )
    rows = []
    for row in dataset:
        choices = row.get("choices", {})
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
        answer = str(row.get("answerKey", ""))
        if len(labels) != 4 or len(texts) != 4 or answer not in labels:
            continue
        rows.append({
            "id": row.get("id"),
            "question": row["question"],
            "labels": labels,
            "choices": texts,
            "answer_index": labels.index(answer),
        })
    random.Random(seed).shuffle(rows)
    if len(rows) < items:
        raise RuntimeError(f"Only {len(rows)} valid ARC rows; requested {items}")
    return rows[:items]


def _prompt_text(tokenizer, item):
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(item["labels"], item["choices"])
    )
    question = (
        f"{item['question']}\n\n{options}\n\n"
        "Answer with only the letter of the correct option."
    )
    messages = [
        {"role": "system", "content": "Answer multiple-choice questions accurately and output one option letter."},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError, ValueError):
            return question + "\nAnswer:"


def _score_item(model, tokenizer, item, device):
    """Score each option label by conditional log likelihood."""
    import torch

    prefix = _prompt_text(tokenizer, item)
    prefix_ids = tokenizer(
        prefix, add_special_tokens=True, truncation=True, max_length=480
    )["input_ids"]
    sequences = []
    answer_lengths = []
    for label in item["labels"]:
        answer_ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            raise RuntimeError(f"Tokenizer produced no IDs for answer label {label!r}")
        sequences.append(prefix_ids + answer_ids)
        answer_lengths.append(len(answer_ids))

    max_len = max(map(len, sequences))
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    for sequence in sequences:
        padding = max_len - len(sequence)
        input_ids.append(sequence + [pad_id] * padding)
        attention_mask.append([1] * len(sequence) + [0] * padding)
    input_ids = torch.tensor(input_ids, device=device)
    attention_mask = torch.tensor(attention_mask, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

    scores = []
    prefix_len = len(prefix_ids)
    for row_index, answer_len in enumerate(answer_lengths):
        score = 0.0
        for offset in range(answer_len):
            target_position = prefix_len + offset
            token_id = int(input_ids[row_index, target_position])
            score += float(log_probs[row_index, target_position - 1, token_id])
        scores.append(score / answer_len)

    predicted = max(range(len(scores)), key=scores.__getitem__)
    correct = item["answer_index"]
    incorrect_best = max(score for index, score in enumerate(scores) if index != correct)
    return {
        "id": item["id"],
        "predicted": predicted,
        "correct_index": correct,
        "is_correct": predicted == correct,
        "scores": [round(value, 6) for value in scores],
        "correct_margin": scores[correct] - incorrect_best,
    }


def _evaluate(model, tokenizer, benchmark, device):
    rows = [_score_item(model, tokenizer, item, device) for item in benchmark]
    return {
        "accuracy": sum(row["is_correct"] for row in rows) / len(rows),
        "mean_margin": statistics.fmean(row["correct_margin"] for row in rows),
        "rows": rows,
    }


def _percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_ci(values, seed, samples=BOOTSTRAP_SAMPLES):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(resample))
    return [round(_percentile(means, 0.025), 6), round(_percentile(means, 0.975), 6)]


def _stratified_model_ci(groups, seed, samples=BOOTSTRAP_SAMPLES):
    """Bootstrap examples within model, then weight model families equally."""
    if not groups or any(not group for group in groups):
        return [None, None]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        model_means = []
        for group in groups:
            resample = [group[rng.randrange(len(group))] for _ in group]
            model_means.append(statistics.fmean(resample))
        estimates.append(statistics.fmean(model_means))
    return [
        round(_percentile(estimates, 0.025), 6),
        round(_percentile(estimates, 0.975), 6),
    ]


def _condition_summary(current, baseline, baseline_sd, seed):
    margin_deltas = [
        current_row["correct_margin"] - baseline_row["correct_margin"]
        for current_row, baseline_row in zip(current["rows"], baseline["rows"])
    ]
    absolute_standardized = [abs(value) / baseline_sd for value in margin_deltas]
    flips = [
        current_row["predicted"] != baseline_row["predicted"]
        for current_row, baseline_row in zip(current["rows"], baseline["rows"])
    ]
    return {
        "accuracy": round(current["accuracy"], 6),
        "accuracy_change_pp": round((current["accuracy"] - baseline["accuracy"]) * 100, 3),
        "mean_margin": round(current["mean_margin"], 6),
        "mean_margin_change": round(statistics.fmean(margin_deltas), 6),
        "mean_abs_standardized_margin_change": round(statistics.fmean(absolute_standardized), 6),
        "abs_standardized_margin_change_ci95": _bootstrap_mean_ci(absolute_standardized, seed),
        "prediction_flip_pct": round(sum(flips) / len(flips) * 100, 3),
        "per_example_abs_standardized_margin_change": [round(value, 6) for value in absolute_standardized],
    }


def _depth_summary(conditions, item_count):
    result = {}
    for depth in DEPTH_FRACTIONS:
        matching = [
            condition for condition in conditions
            if condition["depth"] == depth and condition["scale"] != 1.0
        ]
        per_example = []
        for item_index in range(item_count):
            per_example.append(statistics.fmean(
                condition["metrics"]["per_example_abs_standardized_margin_change"][item_index]
                for condition in matching
            ))
        result[depth] = {
            "mean_abs_standardized_margin_change": round(statistics.fmean(per_example), 6),
            "per_example": [round(value, 6) for value in per_example],
        }
    differences = [
        late - early
        for early, late in zip(result["early"]["per_example"], result["late"]["per_example"])
    ]
    result["late_minus_early"] = {
        "mean": round(statistics.fmean(differences), 6),
        "ci95": _bootstrap_mean_ci(differences, seed=9917),
        "per_example": [round(value, 6) for value in differences],
    }
    return result


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="A100",
    cpu=8,
    memory=32768,
    timeout=3600,
)
def run_model_study(model_name: str, items: int = 32) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS.get(model_name)
    if not spec:
        return {"model": model_name, "error": "No Hugging Face mapping"}

    started = time.time()
    benchmark = _load_arc(items)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["hf"], cache_dir="/cache/hf", trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf"],
        cache_dir="/cache/hf",
        dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device

    layer_path, layers = _transformer_layers(model)
    selected = {}
    for label, fraction in DEPTH_FRACTIONS.items():
        index = round((len(layers) - 1) * fraction)
        norm_name, norm = _pre_attention_norm(layers[index])
        selected[label] = {"index": index, "name": norm_name, "module": norm}

    # Baseline also measures functional RMS gain on real benchmark activations.
    collectors = {}
    handles = []
    for depth, target in selected.items():
        storage = {
            "input_sq_sum": 0.0,
            "input_count": 0,
            "output_sq_sum": 0.0,
            "output_count": 0,
        }
        collectors[depth] = storage
        handles.append(target["module"].register_forward_hook(_rms_collector(storage)))
    baseline = _evaluate(model, tokenizer, benchmark, device)
    for handle in handles:
        handle.remove()

    baseline_margins = [row["correct_margin"] for row in baseline["rows"]]
    baseline_sd = statistics.pstdev(baseline_margins)
    if baseline_sd < 1e-8:
        raise RuntimeError("Baseline choice-margin variance is too small to standardize")

    functional_gain = {}
    for depth, storage in collectors.items():
        input_rms = math.sqrt(storage["input_sq_sum"] / storage["input_count"])
        output_rms = math.sqrt(storage["output_sq_sum"] / storage["output_count"])
        functional_gain[depth] = {
            "layer": selected[depth]["index"],
            "module": selected[depth]["name"],
            "input_rms": round(input_rms, 6),
            "output_rms": round(output_rms, 6),
            "output_to_input_rms": round(output_rms / input_rms, 6),
        }

    # A scale-1 sham verifies that hook installation itself changes nothing.
    sham_target = selected["early"]["module"]
    sham_handle = sham_target.register_forward_hook(_scaling_hook(1.0))
    sham = _evaluate(model, tokenizer, benchmark, device)
    sham_handle.remove()
    max_sham_margin_error = max(
        abs(a["correct_margin"] - b["correct_margin"])
        for a, b in zip(sham["rows"], baseline["rows"])
    )
    sham_prediction_changes = sum(
        a["predicted"] != b["predicted"]
        for a, b in zip(sham["rows"], baseline["rows"])
    )
    if sham_prediction_changes or max_sham_margin_error > 1e-5:
        raise RuntimeError(
            f"Scale-1 sham failed: {sham_prediction_changes} prediction changes, "
            f"max margin error {max_sham_margin_error}"
        )

    conditions = []
    condition_index = 0
    for depth, target in selected.items():
        for scale in SCALES:
            condition_index += 1
            handle = target["module"].register_forward_hook(_scaling_hook(scale))
            current = _evaluate(model, tokenizer, benchmark, device)
            handle.remove()
            metrics = _condition_summary(
                current, baseline, baseline_sd, seed=1000 + condition_index
            )
            conditions.append({
                "id": f"{depth}.scale_{scale}",
                "depth": depth,
                "layer": target["index"],
                "module": target["name"],
                "scale": scale,
                "metrics": metrics,
            })

    depth_summary = _depth_summary(conditions, len(benchmark))
    return {
        "study_version": STUDY_VERSION,
        "model": model_name,
        "family": spec["family"],
        "hf_model": spec["hf"],
        "architecture": getattr(model.config, "model_type", type(model).__name__),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "layer_path": layer_path,
        "layer_count": len(layers),
        "benchmark": {
            "name": "allenai/ai2_arc ARC-Challenge validation",
            "seed": 42,
            "items": len(benchmark),
            "ids": [item["id"] for item in benchmark],
            "scoring": "conditional option-label log likelihood",
        },
        "baseline": {
            "accuracy": round(baseline["accuracy"], 6),
            "mean_margin": round(baseline["mean_margin"], 6),
            "margin_sd": round(baseline_sd, 6),
        },
        "sham_control": {
            "scale": 1.0,
            "prediction_changes": sham_prediction_changes,
            "max_margin_error": max_sham_margin_error,
            "passed": True,
        },
        "functional_gain": functional_gain,
        "conditions": conditions,
        "depth_summary": depth_summary,
        "elapsed_seconds": round(time.time() - started, 1),
    }


def _cross_family_verdict(results):
    successful = [result for result in results if "error" not in result]
    model_tests = []
    model_difference_groups = []
    for result in successful:
        depth = result["depth_summary"]
        difference = depth["late_minus_early"]
        model_difference_groups.append(difference["per_example"])
        model_tests.append({
            "model": result["model"],
            "family": result["family"],
            "early_sensitivity": depth["early"]["mean_abs_standardized_margin_change"],
            "middle_sensitivity": depth["middle"]["mean_abs_standardized_margin_change"],
            "late_sensitivity": depth["late"]["mean_abs_standardized_margin_change"],
            "late_minus_early": difference["mean"],
            "late_minus_early_ci95": difference["ci95"],
            "direction_supports_h2": difference["mean"] > 0,
            "individually_significant": difference["ci95"][0] > 0,
        })

    pooled_ci = _stratified_model_ci(model_difference_groups, seed=271828)
    enough_families = len({result["family"] for result in successful}) >= 3
    all_directional = bool(model_tests) and all(
        test["direction_supports_h2"] for test in model_tests
    )
    significant_models = sum(test["individually_significant"] for test in model_tests)
    pooled_positive = pooled_ci[0] is not None and pooled_ci[0] > 0
    h2_established = (
        enough_families
        and all_directional
        and significant_models >= math.ceil(len(model_tests) * 2 / 3)
        and pooled_positive
    )

    # H1 uses one preregistered condition, not a post-hoc maximum: a 50%
    # reduction at the late pre-attention normalization site.
    h1_by_model = []
    for result in successful:
        condition = next(
            item for item in result["conditions"]
            if item["depth"] == "late" and item["scale"] == 0.5
        )
        metrics = condition["metrics"]
        ci = metrics["abs_standardized_margin_change_ci95"]
        h1_by_model.append({
            "model": result["model"],
            "condition": condition["id"],
            "standardized_margin_response": metrics["mean_abs_standardized_margin_change"],
            "ci95": ci,
            "prediction_flip_pct": metrics["prediction_flip_pct"],
            "supports_h1": ci[0] is not None and ci[0] > 0.05,
        })
    h1_established = enough_families and all(item["supports_h1"] for item in h1_by_model)

    return {
        "preregistered_hypotheses": {
            "H1": "Matched pre-attention norm-output scaling changes decision margins across at least three families.",
            "H2": "Late-depth scaling has greater standardized margin sensitivity than early-depth scaling across families.",
        },
        "criteria": {
            "minimum_distinct_families": 3,
            "H1_preregistered_condition": "late.scale_0.5",
            "H1_per_model_bootstrap_ci_lower_gt": 0.05,
            "H2_all_models_same_positive_direction": True,
            "H2_individual_ci_positive_fraction": "at least 2/3",
            "H2_pooled_bootstrap_ci_lower_gt": 0,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "successful_families": len({result["family"] for result in successful}),
        "failed_models": [result for result in results if "error" in result],
        "H1": {
            "established": h1_established,
            "models": h1_by_model,
        },
        "H2": {
            "established": h2_established,
            "models": model_tests,
            "pooled_late_minus_early_ci95": pooled_ci,
        },
        "overall": (
            "shared_depth_mechanism_established"
            if h2_established
            else "shared_gain_sensitivity_only"
            if h1_established
            else "not_established"
        ),
        "interpretation_limit": (
            "Even a positive result establishes a shared causal response to matched normalization-output scaling on this "
            "benchmark. It does not establish that raw GGUF means are comparable or identify the training-time origin of the mechanism."
        ),
    }


@app.local_entrypoint()
def main(
    models: str = ",".join(DEFAULT_MODELS),
    items: int = 32,
):
    selected = [name.strip() for name in models.split(",") if name.strip()]
    print("━" * 72)
    print("  🪞🔬 Neural Mirror — Cross-Family Mechanism Study")
    print(f"  Models: {', '.join(selected)}")
    print(f"  Benchmark: {items} deterministic ARC-Challenge questions")
    print(f"  Intervention: pre-attention norm output × {SCALES}")
    print("━" * 72)

    futures = {name: run_model_study.spawn(name, items) for name in selected}
    results = []
    for name, future in futures.items():
        try:
            result = future.get()
        except Exception as error:
            result = {"model": name, "error": str(error)}
        results.append(result)
        if "error" in result:
            print(f"  ❌ {name}: {result['error']}")
            continue
        depth = result["depth_summary"]
        print(
            f"  ✅ {name:<14s} baseline={result['baseline']['accuracy'] * 100:5.1f}%  "
            f"sensitivity early/mid/late="
            f"{depth['early']['mean_abs_standardized_margin_change']:.3f}/"
            f"{depth['middle']['mean_abs_standardized_margin_change']:.3f}/"
            f"{depth['late']['mean_abs_standardized_margin_change']:.3f}  "
            f"late−early={depth['late_minus_early']['mean']:+.3f}"
        )

    verdict = _cross_family_verdict(results)
    print("\n  Preregistered verdict:")
    print(f"  H1 shared gain sensitivity: {'ESTABLISHED' if verdict['H1']['established'] else 'NOT ESTABLISHED'}")
    print(f"  H2 shared late-depth mechanism: {'ESTABLISHED' if verdict['H2']['established'] else 'NOT ESTABLISHED'}")
    print(f"  Overall: {verdict['overall']}")

    output = {
        "study_version": STUDY_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": {
            "models": selected,
            "depth_fractions": DEPTH_FRACTIONS,
            "scales": SCALES,
            "benchmark_items": items,
            "intervention": "multiply homologous pre-attention normalization module output",
            "primary_effect": "absolute choice-margin change divided by each model's baseline margin SD",
            "warning": "causal for the intervention; not evidence that raw parameter means share a scale",
        },
        "results": results,
        "verdict": verdict,
    }
    path = os.path.expanduser("~/neural-mirror/mechanism_results.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(temporary, path)
    print(f"\n  Results: {path}")


if __name__ == "__main__":
    # Modal uses the decorated local entrypoint. This block documents that the
    # file is not intended for direct local GPU execution.
    pass
