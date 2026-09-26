"""One-shot fresh certification for frozen braille-literacy adapters.

The certification seeds and task hash are fixed in
`braille_certification_manifest.json`. Results never trigger training on these
seeds; failed identities remain uncertified.
"""

import hashlib
import json
import os
import time
import traceback

import modal

from braille_literacy import TASKS_PER_SEED, _build_tasks, _literacy_prompt, _score

app = modal.App("neural-mirror-braille-certification")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "huggingface_hub<1.0", "accelerate",
        "peft", "safetensors", "sentencepiece", "protobuf", "tiktoken",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("braille_literacy.py", "/root/braille_literacy.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
)

CERTIFICATION_SEEDS = [17017, 29029]
MODELS = {
    "qwen3:4b": {"hf": "Qwen/Qwen3-4B", "adapter": "qwen3-4b-braille-literacy-v3"},
    "llama3.2:3b": {"hf": "unsloth/Llama-3.2-3B-Instruct", "adapter": "llama3.2-3b-braille-literacy-v1"},
    "granite3.3:8b": {"hf": "ibm-granite/granite-3.3-8b-instruct", "adapter": "granite3.3-8b-braille-literacy-v1"},
    "glm4:9b": {"hf": "THUDM/glm-4-9b-chat-hf", "adapter": "glm4-9b-braille-literacy-v1", "trust_remote_code": True},
    "falcon3:7b": {"hf": "tiiuae/Falcon3-7B-Instruct", "adapter": "falcon3-7b-braille-literacy-v2"},
    "mistral:7b": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "adapter": "mistral-7b-braille-literacy-v2"},
    "phi4-mini": {"hf": "microsoft/phi-4-mini-instruct", "adapter": "phi4-mini-braille-literacy-v2"},
    "solar:10.7b": {"hf": "upstage/SOLAR-10.7B-Instruct-v1.0", "adapter": "solar-10.7b-braille-literacy-v2"},
    "gemma3:4b": {"hf": "unsloth/gemma-3-4b-it", "adapter": "gemma3-4b-braille-literacy-v2", "model_class": "image_text"},
    "olmo2:7b": {"hf": "allenai/OLMo-2-1124-7B-Instruct", "adapter": "olmo2-7b-braille-literacy-v2"},
    "yi:6b": {"hf": "01-ai/Yi-6B-Chat", "adapter": "yi-6b-braille-literacy-v2"},
}


def canonical_task_hash():
    tasks = [_build_tasks(seed) for seed in CERTIFICATION_SEEDS]
    encoded = json.dumps(tasks, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prefix(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "Exact protocol conformance test. Output only schema-valid JSON."},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@app.function(image=image, volumes={"/cache": volume}, gpu="H100", cpu=8, memory=65536, timeout=2400, max_containers=12)
def certify(model_name: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    spec = MODELS[model_name]
    started = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec["hf"], cache_dir="/cache/hf", trust_remote_code=spec.get("trust_remote_code", False)
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        loader = AutoModelForImageTextToText if spec.get("model_class") == "image_text" else AutoModelForCausalLM
        base = loader.from_pretrained(
            spec["hf"], cache_dir="/cache/hf", dtype=torch.bfloat16,
            trust_remote_code=spec.get("trust_remote_code", False), low_cpu_mem_usage=True,
        ).to("cuda")
        model = PeftModel.from_pretrained(base, f"/cache/braille-adapters/{spec['adapter']}")
        model.eval()
        outcomes = []
        for seed in CERTIFICATION_SEEDS:
            tasks = _build_tasks(seed)
            inputs = tokenizer(_prefix(tokenizer, _literacy_prompt(tasks)), return_tensors="pt")
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            length = inputs["input_ids"].shape[-1]
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=2200, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(generated[0, length:], skip_special_tokens=True).strip()
                score = _score(tasks, json.loads(raw))
                score.update({"seed": seed, "raw_response": raw})
            except Exception as error:
                score = {"seed": seed, "passed": False, "correct": 0, "total": TASKS_PER_SEED,
                         "error": f"{type(error).__name__}: {error}"}
            outcomes.append(score)
        passed = all(item["passed"] for item in outcomes)
        result = {
            "model": model_name, "adapter": spec["adapter"], "passed": passed,
            "status": "certified" if passed else "not_certified",
            "correct": sum(item["correct"] for item in outcomes), "total": 28,
            "seeds": CERTIFICATION_SEEDS, "task_sha256": canonical_task_hash(),
            "elapsed_seconds": round(time.time() - started, 1), "outcomes": outcomes,
        }
    except Exception as error:
        result = {
            "model": model_name, "adapter": spec["adapter"], "passed": False,
            "status": "inconclusive", "correct": 0, "total": 28,
            "seeds": CERTIFICATION_SEEDS, "task_sha256": canonical_task_hash(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-6000:],
        }
    path = f"/cache/braille-certification/{model_name.replace(':', '-')}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle: json.dump(result, handle, indent=2, ensure_ascii=False)
    volume.commit()
    return result


@app.function(image=image, volumes={"/cache": volume}, timeout=300)
def collect_results():
    directory = "/cache/braille-certification"
    results=[]
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith('.json'):
                with open(os.path.join(directory,name)) as handle: results.append(json.load(handle))
    return results


@app.local_entrypoint()
def main(collect: bool=False):
    if collect:
        results=collect_results.remote()
        path=os.path.expanduser('~/neural-mirror/braille_certification.json')
        with open(path,'w') as handle: json.dump(results,handle,indent=2,ensure_ascii=False)
        for r in results: print(r['model'],r['status'],f"{r['correct']}/{r['total']}")
        return
    for name in MODELS:
        call=certify.spawn(name); print(name,call.object_id)
