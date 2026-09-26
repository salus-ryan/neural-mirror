#!/usr/bin/env python3
"""Build a deterministic, provenance-preserving introspection evidence registry."""

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _record_hash(record):
    payload = {key: value for key, value in record.items() if key not in {"evidence_id", "record_sha256"}}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _load(name):
    path = ROOT / name
    with path.open() as handle:
        return json.load(handle), _file_hash(path)


def build_records():
    certification, certification_hash = _load("braille_certified_admission.json")
    certified = {identity.split("+", 1)[0]: identity for identity in certification["certified"]}
    records = []

    def add(key, owner, level, metric, value, payload, limitations, source_file, source_hash):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"non-finite evidence value for {key}")
        if not limitations:
            raise ValueError(f"evidence must declare limitations: {key}")
        records.append({
            "key": key,
            "owner_model": owner,
            "owner_identity": certified.get(owner),
            "certified_owner": owner in certified,
            "level": level,
            "metric": metric,
            "value": value,
            "payload": payload,
            "limitations": limitations,
            "source_file": source_file,
            "source_sha256": source_hash,
        })

    # Fresh one-shot certification is evidence of protocol competence only.
    for result in certification["results"]:
        if result["status"] != "certified":
            continue
        add(
            f"certification:{result['model']}:{result['adapter']}",
            result["model"], "certification", "exact_protocol_accuracy", 1.0,
            {"correct": result["correct"], "total": result["total"], "seeds": result["seeds"],
             "task_sha256": result["task_sha256"], "adapter": result["adapter"]},
            ["protocol competence only", "not evidence of semantic truth or general capability"],
            "braille_certified_admission.json", certification_hash,
        )

    # Structured base-GGUF profiles from the completed five-family conference.
    roundtable, roundtable_hash = _load("roundtable_core5_fast_consensus.json")
    for model, profile in roundtable["participants"].items():
        add(
            f"static:{model}:norm_growth:first_to_last",
            model, "static", "first_to_last_norm_growth", float(profile["norm_growth"]),
            {"architecture": profile["arch"], "parameters": profile["params"],
             "layers": profile["n_layers"], "first_norm": profile["first_norm"],
             "last_norm": profile["last_norm"]},
            ["host-selected GGUF statistic", "quantization-sensitive", "architecture scales not directly comparable", "not causal"],
            "roundtable_core5_fast_consensus.json", roundtable_hash,
        )

    # Adapter A/B measurements. Include only the exact frozen certified adapter.
    fleet, fleet_hash = _load("braille_fleet_results.json")
    for result in fleet:
        model = result["model"]
        expected_identity = certified.get(model)
        if not expected_identity or expected_identity.split("+", 1)[1] != result.get("adapter"):
            continue
        packet = result.get("adapter_introspection", {}).get("evidence_packet", {})
        for label, payload in packet.get("evidence", {}).items():
            if "delta_rms_upper_bound" in payload:
                metric = "delta_rms_upper_bound"
                value = float(payload["delta_rms_upper_bound"])
                limits = ["Frobenius-derived upper bound", "not exact B@A norm", "not causal", "within-adapter ranking only"]
            else:
                metric = "adapter_parameter_count"
                value = float(payload["adapter_parameters"])
                limits = ["parameter count is not effect size", "not causal"]
            add(
                f"adapter:{model}:{result['adapter']}:{label}", model, "adapter", metric, value,
                payload, limits, "braille_fleet_results.json", fleet_hash,
            )

    # Qwen was trained before fleet-wide profile persistence; ingest its standalone profile.
    qwen_profile, qwen_hash = _load("qwen3-4b-braille-literacy-v3-profile.json")
    qwen_adapter = qwen_profile["adapter"]
    if certified.get("qwen3:4b", "").endswith("+" + qwen_adapter):
        for label, payload in qwen_profile["evidence_packet"]["evidence"].items():
            if "delta_rms_upper_bound" in payload:
                metric, value = "delta_rms_upper_bound", float(payload["delta_rms_upper_bound"])
                limits = ["Frobenius-derived upper bound", "not exact B@A norm", "not causal", "within-adapter ranking only"]
            else:
                metric, value = "adapter_parameter_count", float(payload["adapter_parameters"])
                limits = ["parameter count is not effect size", "not causal"]
            add(
                f"adapter:qwen3:4b:{qwen_adapter}:{label}", "qwen3:4b", "adapter", metric, value,
                payload, limits, "qwen3-4b-braille-literacy-v3-profile.json", qwen_hash,
            )

    # Controlled functional interventions are causal only for the stated hook.
    mechanism, mechanism_hash = _load("mechanism_results.json")
    for result in mechanism["results"]:
        condition = next(item for item in result["conditions"] if item["id"] == "late.scale_0.5")
        metrics = condition["metrics"]
        add(
            f"causal:{result['model']}:late_norm_scale_0.5:standardized_margin_response",
            result["model"], "causal", "mean_abs_standardized_margin_change",
            float(metrics["mean_abs_standardized_margin_change"]),
            {"condition": condition["id"], "layer": condition["layer"], "module": condition["module"],
             "scale": condition["scale"], "ci95": metrics["abs_standardized_margin_change_ci95"],
             "prediction_flip_pct": metrics["prediction_flip_pct"], "benchmark_items": result["benchmark"]["items"],
             "sham_passed": result["sham_control"]["passed"]},
            ["causal only for this activation intervention", "single checkpoint", "single 32-item benchmark", "not evidence of identical algorithms"],
            "mechanism_results.json", mechanism_hash,
        )

    # Assign deterministic uint16 IDs after sorting canonical keys. Fail on duplicate keys.
    records.sort(key=lambda item: item["key"])
    if len({item["key"] for item in records}) != len(records):
        raise ValueError("duplicate evidence keys")
    occupied = set()
    for record in records:
        digest = _record_hash(record)
        candidate = int(digest[:4], 16) or 1
        while candidate in occupied:
            candidate = 1 if candidate == 65535 else candidate + 1
        occupied.add(candidate)
        record["evidence_id"] = candidate
        record["record_sha256"] = digest
    return records


def validate_registry(registry):
    records = registry["records"]
    ids = [item["evidence_id"] for item in records]
    if len(ids) != len(set(ids)) or any(not 1 <= value <= 65535 for value in ids):
        raise ValueError("evidence IDs must be unique uint16 values")
    for record in records:
        if _record_hash(record) != record["record_sha256"]:
            raise ValueError(f"record hash mismatch: {record['key']}")
        if not record["limitations"]:
            raise ValueError(f"missing limitations: {record['key']}")
    expected = hashlib.sha256(_canonical(records).encode()).hexdigest()
    if registry["registry_sha256"] != expected:
        raise ValueError("registry manifest hash mismatch")
    return True


def build_registry():
    records = build_records()
    registry = {
        "version": 1,
        "id_type": "uint16",
        "policy": "Models may interpret registered records but may not create or mutate measurements.",
        "records": records,
        "registry_sha256": hashlib.sha256(_canonical(records).encode()).hexdigest(),
    }
    validate_registry(registry)
    return registry


def main():
    registry = build_registry()
    path = ROOT / "introspection_evidence_registry.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    counts = {}
    for record in registry["records"]:
        counts[record["level"]] = counts.get(record["level"], 0) + 1
    print(json.dumps({"records": len(registry["records"]), "levels": counts,
                      "registry_sha256": registry["registry_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
