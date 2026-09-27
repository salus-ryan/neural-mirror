"""Analyze the preregistered Mistral adapter scale/randomization study."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "mistral_adapter_mechanism_results.json"
MANIFEST = ROOT / "mistral_adapter_mechanism_manifest.json"
OUTPUT = ROOT / "mistral_adapter_mechanism_summary.json"
BOOTSTRAPS = 20_000
SUITES = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]


def _percentile(values, p):
    return round(float(np.quantile(values, p)), 3)


def _condition_rows(condition):
    nested = {}
    for suite in SUITES:
        suite_data = condition["suites"][suite]
        nested[suite] = {}
        for task, rows in suite_data["samples"].items():
            selected = {}
            for row in rows:
                selected.setdefault(row["doc_id"], row)
            if suite == "gsm8k":
                expected = suite_data["results"][task]["exact_match,strict-match"]
                observed = statistics.fmean(row["score"] for row in selected.values())
                if abs(expected - observed) > 1e-12:
                    raise RuntimeError(f"GSM8K strict metric mismatch in {condition['id']}")
            elif len(selected) != len(rows):
                raise RuntimeError(f"unexpected duplicate rows in {condition['id']} {task}")
            nested[suite][task] = selected
    return nested


def _paired_groups(base, current, condition_id):
    groups = {}
    for suite in SUITES:
        if set(base[suite]) != set(current[suite]):
            raise RuntimeError(f"task mismatch in {condition_id}/{suite}")
        groups[suite] = {}
        for task in sorted(base[suite]):
            if set(base[suite][task]) != set(current[suite][task]):
                raise RuntimeError(f"document mismatch in {condition_id}/{task}")
            pairs = []
            for doc_id in sorted(base[suite][task]):
                before, after = base[suite][task][doc_id], current[suite][task][doc_id]
                for field in ("doc_hash", "prompt_hash", "target_hash"):
                    if before[field] != after[field]:
                        raise RuntimeError(f"{field} mismatch in {condition_id}/{task}/{doc_id}")
                pairs.append((before["score"], after["score"]))
            groups[suite][task] = pairs
    return groups


def _macro(nested):
    suites = {}
    for suite, tasks in nested.items():
        suites[suite] = statistics.fmean(
            statistics.fmean(row["score"] for row in rows.values()) for rows in tasks.values()
        )
    return suites, statistics.fmean(suites.values())


def _bootstrap(groups, seed):
    rng = np.random.default_rng(seed)
    suite_draws = []
    for suite in SUITES:
        task_draws = []
        for pairs in groups[suite].values():
            delta = np.asarray([after - before for before, after in pairs], dtype=np.float64) * 100
            indexes = rng.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))
            task_draws.append(delta[indexes].mean(axis=1))
        suite_draws.append(np.stack(task_draws).mean(axis=0))
    values = np.stack(suite_draws).mean(axis=0)
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _mcnemar(groups):
    pairs = [pair for suite in groups.values() for task in suite.values() for pair in task]
    wins = sum(before == 0 and after == 1 for before, after in pairs)
    losses = sum(before == 1 and after == 0 for before, after in pairs)
    n = wins + losses
    tail = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)
    return {"adapter_wins": wins, "adapter_losses": losses, "exact_two_sided_p": round(tail, 10)}


def _spearman(xs, ys):
    def ranks(values):
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        output = [0] * len(values)
        for rank, index in enumerate(ordered, 1):
            output[index] = rank
        return output
    rx, ry = ranks(xs), ranks(ys)
    return statistics.correlation(rx, ry)


def main():
    manifest = json.loads(MANIFEST.read_text())
    result = json.loads(SOURCE.read_text())
    if result["status"] != "complete" or result["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("result is incomplete or not bound to the preregistered manifest")
    conditions = {item["id"]: item for item in result["conditions"]}
    rows = {name: _condition_rows(item) for name, item in conditions.items()}
    base = rows["base"]
    base_suites, base_macro = _macro(base)
    summaries = []
    groups_by_condition = {}
    for index, condition in enumerate(manifest["conditions"]):
        name = condition["id"]
        current_suites, current_macro = _macro(rows[name])
        groups = _paired_groups(base, rows[name], name)
        groups_by_condition[name] = groups
        protocol = conditions[name]["protocol"]
        summary = {
            "id": name, "kind": condition["kind"], "scale": condition["scale"],
            "suite_accuracy": {suite: round(value, 6) for suite, value in current_suites.items()},
            "suite_change_pp": {suite: round((current_suites[suite] - base_suites[suite]) * 100, 3) for suite in SUITES},
            "macro_accuracy": round(current_macro, 6), "macro_change_from_base_pp": round((current_macro - base_macro) * 100, 3),
            "macro_change_ci95_pp": [0.0, 0.0] if name == "base" else _bootstrap(groups, 8100 + index),
            "mcnemar_vs_base": _mcnemar(groups),
            "protocol_correct": protocol["correct"], "protocol_total": protocol["total"],
            "protocol_preserved": protocol["correct"] == protocol["total"],
        }
        if conditions[name].get("randomization"):
            summary["randomization"] = conditions[name]["randomization"]
        summaries.append(summary)

    trained = rows["trained_scale_1.0"]
    random_control = rows["row_sign_randomized_1.0"]
    trained_vs_random = _paired_groups(random_control, trained, "trained_vs_randomized")
    trained_macro = _macro(trained)[1]
    random_macro = _macro(random_control)[1]
    scale_rows = [item for item in summaries if item["kind"] in {"base", "trained"}]
    preserving = [item for item in scale_rows if item["protocol_preserved"]]
    selected = max(preserving, key=lambda item: item["macro_accuracy"]) if preserving else None

    integrity = []
    for suite, tasks in base.items():
        for task, task_rows in tasks.items():
            for doc_id, row in sorted(task_rows.items()):
                integrity.append((suite, task, doc_id, row["doc_hash"], row["prompt_hash"], row["target_hash"]))
    summary = {
        "provenance": {
            "manifest_sha256": manifest["manifest_sha256"],
            "results_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preregistration_commit": "fbb0f6d966744cfc90600712709995d5ba7a2da5",
            "paired_sample_sha256": hashlib.sha256(json.dumps(integrity, separators=(",", ":")).encode()).hexdigest(),
        },
        "design": {"bootstrap_samples": BOOTSTRAPS, "all_condition_sample_hashes_matched": True,
                   "primary_weighting": "equal suites; equal MMLU subjects within MMLU"},
        "base_macro_accuracy": round(base_macro, 6),
        "conditions": summaries,
        "trained_scale_curve_spearman_rho": round(_spearman([item["scale"] for item in scale_rows], [item["macro_accuracy"] for item in scale_rows]), 6),
        "trained_full_minus_row_sign_randomized_pp": round((trained_macro - random_macro) * 100, 3),
        "trained_full_minus_row_sign_randomized_ci95_pp": _bootstrap(trained_vs_random, 92817),
        "pareto_report": None if selected is None else {"highest_measured_accuracy_protocol_preserving_scale": selected["scale"],
                                                         "macro_accuracy": selected["macro_accuracy"],
                                                         "change_from_base_pp": selected["macro_change_from_base_pp"]},
        "interpretation": [
            "Protocol preservation first appeared at scale 0.5 in the fixed grid.",
            "Row-sign randomization destroyed protocol performance despite preserving every LoRA B Frobenius norm, so protocol skill is not explained by norm alone.",
            "The trained-minus-randomized general-accuracy interval includes zero; this study does not distinguish their general benchmark effects conclusively.",
            "The scale curve is a one-checkpoint causal intervention and does not identify individual harmful layers."],
        "limitations": manifest["limitations"],
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    for item in summaries:
        print(item["id"], item["macro_change_from_base_pp"], item["macro_change_ci95_pp"],
              f"protocol={item['protocol_correct']}/{item['protocol_total']}")
    print("trained-minus-randomized", summary["trained_full_minus_row_sign_randomized_pp"],
          summary["trained_full_minus_row_sign_randomized_ci95_pp"])
    print("pareto", summary["pareto_report"])


if __name__ == "__main__":
    main()
