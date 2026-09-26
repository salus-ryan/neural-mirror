"""Analyze the preregistered canonical paired adapter replication."""

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "canonical_adapter_replication_results.json"
MANIFEST = ROOT / "canonical_adapter_replication_manifest.json"
OUTPUT = ROOT / "canonical_adapter_replication_summary.json"
BOOTSTRAPS = 20_000
SUITES = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _ci(values):
    return [round(_percentile(values, 0.025), 3), round(_percentile(values, 0.975), 3)]


def _task_rows(result, condition, suite):
    output = {}
    suite_data = result["conditions"][condition][suite]
    for task, rows in suite_data["samples"].items():
        selected = {}
        for row in rows:
            # lm-eval logs GSM8K strict-match rows followed by flexible-extract
            # rows under the same task. The preregistered sample metric selects
            # strict-match, so retain the first row for each document.
            selected.setdefault(row["doc_id"], row)
        if suite == "gsm8k":
            aggregate = suite_data["results"][task]["exact_match,strict-match"]
            observed = statistics.fmean(row["score"] for row in selected.values())
            if abs(aggregate - observed) > 1e-12:
                raise RuntimeError(f"GSM8K strict extraction mismatch for {result['model']} {condition}")
        elif len(selected) != len(rows):
            raise RuntimeError(f"unexpected duplicate documents in {suite}/{task}")
        output[task] = selected
    return output


def _paired_groups(result):
    groups = {}
    integrity = []
    for suite in SUITES:
        base = _task_rows(result, "base", suite)
        adapter = _task_rows(result, "adapter", suite)
        if set(base) != set(adapter):
            raise RuntimeError(f"task mismatch for {result['model']} {suite}")
        groups[suite] = {}
        for task in sorted(base):
            if set(base[task]) != set(adapter[task]):
                raise RuntimeError(f"doc ID mismatch for {result['model']} {task}")
            pairs = []
            for doc_id in sorted(base[task]):
                before, after = base[task][doc_id], adapter[task][doc_id]
                for field in ("doc_hash", "prompt_hash", "target_hash"):
                    if before[field] != after[field]:
                        raise RuntimeError(f"{field} mismatch for {result['model']} {task}/{doc_id}")
                pairs.append((before["score"], after["score"]))
                integrity.append((task, doc_id, before["doc_hash"], before["prompt_hash"], before["target_hash"]))
            groups[suite][task] = pairs
    digest = hashlib.sha256(json.dumps(integrity, separators=(",", ":")).encode()).hexdigest()
    return groups, digest


def _suite_stats(tasks):
    base = statistics.fmean(statistics.fmean(x for x, _ in pairs) for pairs in tasks.values())
    adapter = statistics.fmean(statistics.fmean(y for _, y in pairs) for pairs in tasks.values())
    return base, adapter


def _bootstrap(groups, seed):
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAPS):
        suite_effects = []
        for suite in SUITES:
            task_effects = []
            for pairs in groups[suite].values():
                sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                task_effects.append(statistics.fmean((after - before) * 100 for before, after in sampled))
            suite_effects.append(statistics.fmean(task_effects))
        estimates.append(statistics.fmean(suite_effects))
    return _ci(estimates)


def _bootstrap_checkpoint_macro(group_sets, seed=44017):
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAPS):
        checkpoint_effects = []
        for groups in group_sets:
            suite_effects = []
            for suite in SUITES:
                task_effects = []
                for pairs in groups[suite].values():
                    sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                    task_effects.append(statistics.fmean((after - before) * 100 for before, after in sampled))
                suite_effects.append(statistics.fmean(task_effects))
            checkpoint_effects.append(statistics.fmean(suite_effects))
        estimates.append(statistics.fmean(checkpoint_effects))
    return _ci(estimates)


def _mcnemar(groups):
    pairs = [pair for suite in groups.values() for task in suite.values() for pair in task]
    wins = sum(before == 0 and after == 1 for before, after in pairs)
    losses = sum(before == 1 and after == 0 for before, after in pairs)
    n = wins + losses
    if not n:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1)) / (2 ** n)
        p_value = min(1.0, 2 * tail)
    return {"adapter_wins": wins, "adapter_losses": losses, "discordant": n,
            "exact_two_sided_p": p_value, "pooled_items": len(pairs)}


def _holm(entries):
    ordered = sorted(entries, key=lambda item: item[1])
    running = 0.0
    adjusted = {}
    count = len(ordered)
    for index, (model, p_value) in enumerate(ordered):
        running = max(running, min(1.0, p_value * (count - index)))
        adjusted[model] = running
    return adjusted


def main():
    manifest = json.loads(MANIFEST.read_text())
    results = json.loads(SOURCE.read_text())
    if len(results) != len(manifest["models"]):
        raise RuntimeError("incomplete checkpoint set")
    analyzed = []
    internal = {}
    for model_index, result in enumerate(results):
        if result["status"] != "complete" or result["smoke"]:
            raise RuntimeError(f"non-complete result: {result['model']}")
        if result["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError(f"manifest mismatch: {result['model']}")
        groups, sample_hash = _paired_groups(result)
        suite_results = {}
        for suite, tasks in groups.items():
            base, adapter = _suite_stats(tasks)
            suite_results[suite] = {
                "base_accuracy": round(base, 6),
                "adapter_accuracy": round(adapter, 6),
                "change_pp": round((adapter - base) * 100, 3),
                "tasks": len(tasks),
                "unique_items": sum(len(pairs) for pairs in tasks.values()),
            }
        base_macro = statistics.fmean(item["base_accuracy"] for item in suite_results.values())
        adapter_macro = statistics.fmean(item["adapter_accuracy"] for item in suite_results.values())
        test = _mcnemar(groups)
        record = {
            "model": result["model"],
            "replication_hypothesis": manifest["models"][result["model"]]["replication_hypothesis"],
            "condition_order": result["condition_order"],
            "paired_sample_sha256": sample_hash,
            "suites": suite_results,
            "base_macro_accuracy": round(base_macro, 6),
            "adapter_macro_accuracy": round(adapter_macro, 6),
            "macro_change_pp": round((adapter_macro - base_macro) * 100, 3),
            "macro_change_ci95_pp": _bootstrap(groups, 19001 + model_index),
            "mcnemar": test,
        }
        analyzed.append(record)
        internal[result["model"]] = groups

    adjusted = _holm([(item["model"], item["mcnemar"]["exact_two_sided_p"]) for item in analyzed])
    for item in analyzed:
        p_adjusted = adjusted[item["model"]]
        item["mcnemar"]["exact_two_sided_p"] = round(item["mcnemar"]["exact_two_sided_p"], 10)
        item["mcnemar"]["holm_adjusted_p"] = round(p_adjusted, 10)
        direction = item["replication_hypothesis"]
        sign_matches = ((direction == "positive" and item["macro_change_pp"] > 0) or
                        (direction == "negative" and item["macro_change_pp"] < 0))
        if direction == "neutral_control":
            item["replication_verdict"] = "no_detected_change" if p_adjusted >= 0.05 else "detected_change"
        else:
            item["replication_verdict"] = "replicated" if sign_matches and p_adjusted < 0.05 else "not_replicated"

    macro_change = statistics.fmean(item["macro_change_pp"] for item in analyzed)
    summary = {
        "provenance": {
            "manifest_sha256": manifest["manifest_sha256"],
            "results_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preregistration_commit": "bb9aee61eb646b67fc213407ff5b038cf940d6ce",
        },
        "design": {
            "checkpoints": len(analyzed),
            "bootstrap_samples": BOOTSTRAPS,
            "primary_weighting": "equal suites; equal MMLU subjects within the MMLU suite",
            "all_sample_hashes_matched": True,
            "holm_familywise_alpha": 0.05,
        },
        "checkpoint_results": analyzed,
        "four_checkpoint_base_macro_accuracy": round(statistics.fmean(item["base_macro_accuracy"] for item in analyzed), 6),
        "four_checkpoint_adapter_macro_accuracy": round(statistics.fmean(item["adapter_macro_accuracy"] for item in analyzed), 6),
        "four_checkpoint_macro_change_pp": round(macro_change, 3),
        "four_checkpoint_macro_change_ci95_pp": _bootstrap_checkpoint_macro([internal[item["model"]] for item in analyzed]),
        "interpretation_policy": "Replication requires Holm-adjusted p<0.05 and the preregistered sign. Failure to reject is not proof of equivalence.",
        "limitations": manifest["scope_limitations"] + [
            "McNemar tests pool binary items and therefore weight GSM8K less than larger suites; macro effect estimates weight suites equally.",
            "GSM8K uses the preregistered strict-match metric; flexible extraction is not used in the primary analysis.",
        ],
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    for item in analyzed:
        print(item["model"], item["macro_change_pp"], item["macro_change_ci95_pp"],
              item["mcnemar"]["holm_adjusted_p"], item["replication_verdict"])
    print("macro", summary["four_checkpoint_macro_change_pp"])


if __name__ == "__main__":
    main()
