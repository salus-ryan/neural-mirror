"""Preregistered canonical lm-eval replication of exploratory LoRA effects.

The manifest must be committed before this script is run with GPU inference.
"""

import hashlib
import json
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "canonical_adapter_replication_manifest.json"

app = modal.App("neural-mirror-canonical-adapter-replication")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "huggingface_hub<1.0", "accelerate",
        "peft", "safetensors", "sentencepiece", "protobuf", "tiktoken",
        "datasets==3.6.0", "lm-eval==0.4.9.2",
    )
    .add_local_file(str(MANIFEST_PATH), "/root/canonical_adapter_replication_manifest.json")
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    claimed = manifest["manifest_sha256"]
    actual = hashlib.sha256(_canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"}).encode()).hexdigest()
    if claimed != actual:
        raise RuntimeError(f"manifest hash mismatch: {claimed} != {actual}")
    return manifest


def _tree_hash(directory):
    files = {}
    for root, _, names in os.walk(directory):
        for name in sorted(names):
            path = os.path.join(root, name)
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            files[os.path.relpath(path, directory)] = digest.hexdigest()
    canonical = "".join(name + "\0" + files[name] + "\n" for name in sorted(files)).encode()
    return hashlib.sha256(canonical).hexdigest(), files


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metric_from_sample(task, row, requested):
    candidates = [requested, requested + ",none", requested + ",strict-match", requested + ",flexible-extract"]
    for key in candidates:
        if key in row:
            return float(row[key])
    for key, value in row.items():
        if key.split(",", 1)[0] == requested and isinstance(value, (int, float, bool)):
            return float(value)
    raise RuntimeError(f"metric {requested!r} absent from {task} sample; keys={sorted(row)}")


def _sanitize_run(raw, metric):
    compact = {"results": _jsonable(raw.get("results", {})), "samples": {}}
    for task, rows in raw.get("samples", {}).items():
        compact["samples"][task] = [
            {
                "doc_id": row.get("doc_id"),
                "doc_hash": row.get("doc_hash"),
                "prompt_hash": row.get("prompt_hash"),
                "target_hash": row.get("target_hash"),
                "score": _metric_from_sample(task, row, metric),
            }
            for row in rows
        ]
    return compact


def _verify_upstream(manifest):
    from huggingface_hub import HfApi

    api = HfApi()
    observed = {}
    for repo, expected in manifest["dataset_revisions"].items():
        actual = api.dataset_info(repo_id=repo, revision=expected).sha
        if actual != expected:
            raise RuntimeError(f"dataset revision mismatch for {repo}: {actual} != {expected}")
        observed[repo] = actual
    return observed


def _warmup(model, tokenizer, enabled):
    import torch

    context = nullcontext() if enabled else model.disable_adapter()
    with context, torch.inference_mode():
        inputs = tokenizer("A deterministic warmup prompt.", return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        model(**inputs, use_cache=False)
    torch.cuda.synchronize()


def _run_suite(model, tokenizer, suite):
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device="cuda",
        batch_size=suite["batch_size"],
        max_batch_size=suite["batch_size"],
        logits_cache=False,
    )
    raw = evaluator.simple_evaluate(
        model=wrapped,
        tasks=suite["tasks"],
        num_fewshot=suite.get("num_fewshot"),
        limit=suite["limit_per_task"],
        log_samples=True,
        random_seed=1234,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    return _sanitize_run(raw, suite["sample_metric"])


@app.function(
    image=image,
    volumes={"/cache": volume},
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=21600,
    max_containers=4,
)
def evaluate_model(model_name: str, smoke: bool = False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = _manifest()
    spec = manifest["models"][model_name]
    started = time.time()
    result = {
        "model": model_name,
        "manifest_sha256": manifest["manifest_sha256"],
        "status": "failed",
        "smoke": smoke,
    }
    try:
        result["observed_dataset_revisions"] = _verify_upstream(manifest)
        adapter_dir = "/cache/braille-adapters/" + spec["adapter"]
        tree_hash, file_hashes = _tree_hash(adapter_dir)
        if tree_hash != spec["adapter_tree_sha256"]:
            raise RuntimeError(f"adapter tree mismatch: {tree_hash} != {spec['adapter_tree_sha256']}")
        result["adapter_tree_sha256"] = tree_hash
        result["adapter_model_sha256"] = file_hashes["adapter_model.safetensors"]

        tokenizer = AutoTokenizer.from_pretrained(
            spec["hf_model"], revision=spec["model_revision"], cache_dir="/cache/hf"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            spec["hf_model"], revision=spec["model_revision"], cache_dir="/cache/hf",
            dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda")
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        model.config.use_cache = True

        # Warm both paths before either measured condition. Warmup data are excluded.
        _warmup(model, tokenizer, False)
        _warmup(model, tokenizer, True)

        suites = manifest["suites"]
        if smoke:
            suites = [{**suite, "limit_per_task": 2, "tasks": suite["tasks"][:1]} for suite in suites[:3]]
        conditions = {}
        for condition in spec["condition_order"]:
            enabled = condition == "adapter"
            context = nullcontext() if enabled else model.disable_adapter()
            condition_started = time.time()
            with context:
                conditions[condition] = {}
                for suite in suites:
                    suite_started = time.time()
                    conditions[condition][suite["name"]] = _run_suite(model, tokenizer, suite)
                    conditions[condition][suite["name"]]["elapsed_seconds"] = round(time.time() - suite_started, 3)
            conditions[condition]["elapsed_seconds"] = round(time.time() - condition_started, 3)

        result.update({
            "status": "complete",
            "hf_model": spec["hf_model"],
            "model_revision": spec["model_revision"],
            "adapter": spec["adapter"],
            "condition_order": spec["condition_order"],
            "conditions": conditions,
            "elapsed_seconds": round(time.time() - started, 3),
        })
    except Exception as error:
        result.update({
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-12000:],
            "elapsed_seconds": round(time.time() - started, 3),
        })

    suffix = "-smoke" if smoke else ""
    directory = "/cache/canonical-adapter-replication"
    os.makedirs(directory, exist_ok=True)
    path = f"{directory}/{model_name.replace(':', '-')}{suffix}.json"
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    volume.commit()
    return {key: result.get(key) for key in ("model", "status", "smoke", "error", "elapsed_seconds")}


@app.function(image=image, volumes={"/cache": volume}, timeout=600)
def collect_results(smoke: bool = False):
    suffix = "-smoke.json" if smoke else ".json"
    directory = "/cache/canonical-adapter-replication"
    results = []
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith(suffix) and (("-smoke" in name) == smoke):
                with open(os.path.join(directory, name)) as handle:
                    results.append(json.load(handle))
    return results


@app.local_entrypoint()
def main(collect: bool = False, smoke: bool = False, models: str = ""):
    manifest = _manifest()
    selected = [item.strip() for item in models.split(",") if item.strip()] or list(manifest["models"])
    if collect:
        results = collect_results.remote(smoke)
        destination = ROOT / ("canonical_adapter_replication_smoke.json" if smoke else "canonical_adapter_replication_results.json")
        destination.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        for result in results:
            print(result["model"], result["status"], result.get("error", ""))
        print(destination)
        return
    for name in selected:
        if name not in manifest["models"]:
            raise ValueError(f"model not preregistered: {name}")
        call = evaluate_model.spawn(name, smoke)
        print(name, call.object_id)
