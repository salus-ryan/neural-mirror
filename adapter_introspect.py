#!/usr/bin/env python3
"""Inspect PEFT LoRA adapters without confusing them with base GGUF weights.

The reported `delta_rms_upper_bound` is a scale-normalized upper bound derived
from ||B||_F ||A||_F, not the exact norm of B@A. This avoids materializing every
full dense update and keeps the evidence comparable within one adapter.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict


_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),
]
_LORA_PATTERN = re.compile(r"^(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$")


def _layer_number(name):
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _module_name(prefix):
    return prefix.rsplit(".", 1)[-1]


def _tensor_stats(tensor):
    import torch

    values = tensor.detach().float()
    count = values.numel()
    if not count:
        return {"elements": 0}
    return {
        "shape": list(values.shape),
        "elements": count,
        "mean": round(float(values.mean()), 9),
        "std": round(float(values.std(unbiased=False)), 9),
        "abs_mean": round(float(values.abs().mean()), 9),
        "min": round(float(values.min()), 9),
        "max": round(float(values.max()), 9),
        "frobenius_norm": round(float(torch.linalg.vector_norm(values)), 7),
        "near_zero_pct": round(float((values.abs() < 1e-8).float().mean() * 100), 4),
    }


class LoRAAdapter:
    """Read PEFT config and safetensors, then expose bounded inspection tools."""

    def __init__(self, adapter_dir):
        from safetensors import safe_open

        self.adapter_dir = os.path.abspath(adapter_dir)
        config_path = os.path.join(self.adapter_dir, "adapter_config.json")
        tensor_path = os.path.join(self.adapter_dir, "adapter_model.safetensors")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"missing {config_path}")
        if not os.path.isfile(tensor_path):
            raise FileNotFoundError(f"missing {tensor_path}")
        with open(config_path) as handle:
            self.config = json.load(handle)
        self.tensor_path = tensor_path
        self._profile_cache = None
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            self.tensor_names = sorted(handle.keys())
            self.metadata = handle.metadata() or {}

    def _tensor(self, name):
        from safetensors import safe_open

        with safe_open(self.tensor_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def _pairs(self):
        pairs = defaultdict(dict)
        for name in self.tensor_names:
            match = _LORA_PATTERN.match(name)
            if match:
                pairs[match.group(1)][match.group(2)] = name
        return {prefix: pair for prefix, pair in pairs.items() if set(pair) == {"A", "B"}}

    def inspect_self(self):
        pairs = self._pairs()
        layers = sorted({layer for layer in (_layer_number(name) for name in pairs) if layer is not None})
        modules = sorted({_module_name(name) for name in pairs})
        elements = 0
        dtypes = defaultdict(int)
        for name in self.tensor_names:
            tensor = self._tensor(name)
            elements += tensor.numel()
            dtypes[str(tensor.dtype).replace("torch.", "")] += tensor.numel()
        return {
            "identity": {
                "format": "PEFT LoRA safetensors",
                "adapter_directory": self.adapter_dir,
                "base_model": self.config.get("base_model_name_or_path"),
                "peft_type": self.config.get("peft_type"),
                "task_type": self.config.get("task_type"),
            },
            "configuration": {
                "rank": self.config.get("r"),
                "alpha": self.config.get("lora_alpha"),
                "dropout": self.config.get("lora_dropout"),
                "scaling": self._scaling(),
                "target_modules": self.config.get("target_modules"),
                "bias": self.config.get("bias"),
            },
            "scale": {
                "adapter_parameters": elements,
                "adapter_parameters_human": f"{elements / 1e6:.2f}M",
                "tensor_count": len(self.tensor_names),
                "paired_updates": len(pairs),
                "adapted_layer_count": len(layers),
                "adapted_layers": layers,
                "adapted_module_types": modules,
                "dtypes": dict(dtypes),
                "file_bytes": os.path.getsize(self.tensor_path),
            },
        }

    def _scaling(self):
        rank = self.config.get("r") or 0
        alpha = self.config.get("lora_alpha") or 0
        return alpha / rank if rank else None

    def inspect_tensor(self, name):
        if name not in self.tensor_names:
            matches = [candidate for candidate in self.tensor_names if name in candidate]
            return {"error": f"tensor not found: {name}", "matches": matches[:20]}
        return {"name": name, **_tensor_stats(self._tensor(name))}

    def list_tensors(self, filter_str=None):
        names = self.tensor_names
        if filter_str:
            names = [name for name in names if filter_str in name]
        return {
            "total": len(names),
            "tensors": [self.inspect_tensor(name) for name in names],
        }

    def update_profile(self):
        """Profile each paired LoRA update without constructing full B@A."""
        if self._profile_cache is not None:
            return self._profile_cache
        scaling = self._scaling() or 1.0
        updates = []
        for prefix, pair in self._pairs().items():
            a = self._tensor(pair["A"])
            b = self._tensor(pair["B"])
            a_stats = _tensor_stats(a)
            b_stats = _tensor_stats(b)
            out_features = b.shape[0]
            in_features = a.shape[1]
            product_bound = scaling * a_stats["frobenius_norm"] * b_stats["frobenius_norm"]
            rms_bound = product_bound / math.sqrt(out_features * in_features)
            updates.append({
                "prefix": prefix,
                "layer": _layer_number(prefix),
                "module": _module_name(prefix),
                "rank": int(a.shape[0]),
                "input_features": int(in_features),
                "output_features": int(out_features),
                "a_frobenius_norm": a_stats["frobenius_norm"],
                "b_frobenius_norm": b_stats["frobenius_norm"],
                "delta_frobenius_upper_bound": round(product_bound, 7),
                "delta_rms_upper_bound": round(rms_bound, 10),
                "a_near_zero_pct": a_stats["near_zero_pct"],
                "b_near_zero_pct": b_stats["near_zero_pct"],
            })
        updates.sort(key=lambda item: item["delta_rms_upper_bound"], reverse=True)
        self._profile_cache = {
            "measurement": "scale * ||A||_F * ||B||_F / sqrt(out_features * input_features)",
            "warning": "This is an upper bound on update RMS, not exact ||B@A|| and not a causal effect.",
            "updates": updates,
        }
        return self._profile_cache

    def layer_fingerprint(self):
        updates = self.update_profile()["updates"]
        grouped = defaultdict(list)
        for item in updates:
            grouped[item["layer"]].append(item["delta_rms_upper_bound"])
        fingerprint = {}
        for layer, values in sorted(grouped.items(), key=lambda item: (-1 if item[0] is None else item[0])):
            key = "non_layer" if layer is None else f"layer_{layer}"
            fingerprint[key] = {
                "update_count": len(values),
                "mean_delta_rms_upper_bound": round(sum(values) / len(values), 10),
                "max_delta_rms_upper_bound": round(max(values), 10),
            }
        return {"fingerprint": fingerprint, "layer_count": sum(key != "non_layer" for key in fingerprint)}

    def evidence_packet(self, top_k=8):
        overview = self.inspect_self()
        profile = self.update_profile()
        evidence = {
            "ADAPTER.1": {
                "claim": "adapter identity and scale",
                "base_model": overview["identity"]["base_model"],
                "rank": overview["configuration"]["rank"],
                "alpha": overview["configuration"]["alpha"],
                "adapter_parameters": overview["scale"]["adapter_parameters"],
                "paired_updates": overview["scale"]["paired_updates"],
            }
        }
        for index, update in enumerate(profile["updates"][:top_k], start=2):
            evidence[f"ADAPTER.{index}"] = {
                "layer": update["layer"],
                "module": update["module"],
                "delta_rms_upper_bound": update["delta_rms_upper_bound"],
                "rank": update["rank"],
            }
        return {
            "source": "host-executed PEFT safetensors inspection",
            "measurement_warning": profile["warning"],
            "evidence": evidence,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir")
    parser.add_argument("--output")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    adapter = LoRAAdapter(args.adapter_dir)
    result = {
        "overview": adapter.inspect_self(),
        "update_profile": adapter.update_profile(),
        "layer_fingerprint": adapter.layer_fingerprint(),
        "evidence_packet": adapter.evidence_packet(args.top_k),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
