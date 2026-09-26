"""Preregistered Mistral LoRA scaling and norm-preserving randomization study."""

import hashlib
import json
import os
import random
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import modal

from braille_literacy import TASKS_PER_SEED, _build_tasks, _literacy_prompt, _score
from canonical_adapter_replication import _canonical, _run_suite, _tree_hash, _verify_upstream

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "mistral_adapter_mechanism_manifest.json"

app = modal.App("neural-mirror-mistral-adapter-mechanism")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "huggingface_hub<1.0", "accelerate",
        "peft", "safetensors", "sentencepiece", "protobuf", "tiktoken",
        "datasets==3.6.0", "lm-eval==0.4.9.2",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("braille_literacy.py", "/root/braille_literacy.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
    .add_local_file("canonical_adapter_replication.py", "/root/canonical_adapter_replication.py")
    .add_local_file("canonical_adapter_replication_manifest.json", "/root/canonical_adapter_replication_manifest.json")
    .add_local_file(str(MANIFEST_PATH), "/root/mistral_adapter_mechanism_manifest.json")
)


def _manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if manifest["manifest_sha256"] != actual:
        raise RuntimeError("mechanism manifest hash mismatch")
    return manifest


def _lora_scalings(model):
    entries = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter_name, value in scaling.items():
                entries.append((name, module, adapter_name, float(value)))
    if not entries:
        raise RuntimeError("no PEFT LoRA scaling entries found")
    return entries


def _set_scale(entries, factor):
    for _, module, adapter_name, original in entries:
        module.scaling[adapter_name] = original * factor


def _toggle_row_sign_randomization(model, seed):
    import torch

    sign_digest = hashlib.sha256()
    tensors = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            lora_b = getattr(module, "lora_B", None)
            if lora_b is None:
                continue
            for adapter_name, projection in lora_b.items():
                weight = projection.weight
                local_seed = int(hashlib.sha256(f"{seed}:{name}:{adapter_name}".encode()).hexdigest()[:16], 16)
                rng = random.Random(local_seed)
                signs_list = [1.0 if rng.getrandbits(1) else -1.0 for _ in range(weight.shape[0])]
                signs = torch.tensor(signs_list, device=weight.device, dtype=weight.dtype).unsqueeze(1)
                weight.mul_(signs)
                sign_digest.update(name.encode() + b"\0" + bytes(1 if value > 0 else 0 for value in signs_list))
                tensors += 1
    if not tensors:
        raise RuntimeError("no LoRA B tensors randomized")
    return {"lora_b_tensors": tensors, "sign_pattern_sha256": sign_digest.hexdigest()}


def _chat_prompt(tokenizer, tasks):
    messages = [
        {"role": "system", "content": "Exact protocol conformance test. Output only schema-valid JSON."},
        {"role": "user", "content": "/no_think\n" + _literacy_prompt(tasks)},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _protocol_eval(model, tokenizer, seeds):
    import torch

    outcomes = []
    for seed in seeds:
        tasks = _build_tasks(seed)
        prompt = _chat_prompt(tokenizer, tasks)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=2200, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        try:
            scored = _score(tasks, json.loads(raw))
        except Exception as error:
            scored = {"passed": False, "correct": 0, "total": TASKS_PER_SEED, "error": str(error)}
        scored.update({"seed": seed, "raw_response": raw})
        outcomes.append(scored)
    return {"correct": sum(item["correct"] for item in outcomes),
            "total": TASKS_PER_SEED * len(seeds), "outcomes": outcomes}


@app.function(image=image, volumes={"/cache": volume}, gpu="H100", cpu=8, memory=65536, timeout=21600)
def run_study(smoke: bool = False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = _manifest()
    spec = manifest["model"]
    started = time.time()
    result = {"status": "failed", "smoke": smoke, "manifest_sha256": manifest["manifest_sha256"]}
    try:
        result["observed_dataset_revisions"] = _verify_upstream(manifest)
        adapter_dir = "/cache/braille-adapters/" + spec["adapter"]
        adapter_hash, files = _tree_hash(adapter_dir)
        if adapter_hash != spec["adapter_tree_sha256"]:
            raise RuntimeError("adapter tree mismatch")
        tokenizer = AutoTokenizer.from_pretrained(spec["hf_model"], revision=spec["model_revision"], cache_dir="/cache/hf")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            spec["hf_model"], revision=spec["model_revision"], cache_dir="/cache/hf",
            dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda")
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        entries = _lora_scalings(model)
        suites = manifest["suites"]
        seeds = manifest["protocol_seeds"]
        conditions = manifest["conditions"]
        if smoke:
            suites = [{**suites[0], "tasks": suites[0]["tasks"][:1], "limit_per_task": 2}]
            seeds = []
            conditions = conditions[:3]

        outputs = []
        for condition in conditions:
            condition_started = time.time()
            factor = condition.get("scale", 1.0)
            _set_scale(entries, factor)
            randomization = None
            if condition["kind"] == "row_sign_randomized":
                randomization = _toggle_row_sign_randomization(model, manifest["randomization_seed"])
            context = model.disable_adapter() if condition["kind"] == "base" else nullcontext()
            with context:
                suite_outputs = {}
                for suite in suites:
                    suite_outputs[suite["name"]] = _run_suite(model, tokenizer, suite)
                protocol = _protocol_eval(model, tokenizer, seeds) if seeds else None
            if condition["kind"] == "row_sign_randomized":
                restored = _toggle_row_sign_randomization(model, manifest["randomization_seed"])
                if restored != randomization:
                    raise RuntimeError("randomization restoration digest mismatch")
            outputs.append({"id": condition["id"], "kind": condition["kind"], "scale": factor,
                            "randomization": randomization, "suites": suite_outputs, "protocol": protocol,
                            "elapsed_seconds": round(time.time() - condition_started, 3)})

        _set_scale(entries, 1.0)
        result.update({"status": "complete", "model": spec["name"], "adapter": spec["adapter"],
                       "adapter_tree_sha256": adapter_hash, "adapter_model_sha256": files["adapter_model.safetensors"],
                       "conditions": outputs, "elapsed_seconds": round(time.time() - started, 3)})
    except Exception as error:
        result.update({"error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()[-12000:], "elapsed_seconds": round(time.time() - started, 3)})
    directory = "/cache/mistral-adapter-mechanism"
    os.makedirs(directory, exist_ok=True)
    path = directory + ("/smoke.json" if smoke else "/results.json")
    with open(path + ".tmp", "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    volume.commit()
    return {key: result.get(key) for key in ("status", "smoke", "error", "elapsed_seconds")}


@app.function(image=image, volumes={"/cache": volume}, timeout=600)
def collect(smoke: bool = False):
    path = "/cache/mistral-adapter-mechanism/" + ("smoke.json" if smoke else "results.json")
    with open(path) as handle:
        return json.load(handle)


@app.local_entrypoint()
def main(collect_only: bool = False, smoke: bool = False):
    if collect_only:
        result = collect.remote(smoke)
        path = ROOT / ("mistral_adapter_mechanism_smoke.json" if smoke else "mistral_adapter_mechanism_results.json")
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(path, result["status"], result.get("error", ""))
    else:
        call = run_study.spawn(smoke)
        print(call.object_id)
