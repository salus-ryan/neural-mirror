"""Train and retest a computer-braille literacy LoRA.

Training examples are generated from seeds disjoint from the immutable literacy
exam seeds (17 and 29). The adapter is admitted only if it scores 28/28 on the
same exact scorer and task generator used for the unadapted models.

Usage:
  MODAL_PROFILE=salus modal run braille_curriculum.py
"""

import json
import os
import random
import time

import modal

from braille_literacy import (
    SEEDS as EXAM_SEEDS,
    TASKS_PER_SEED,
    _build_tasks,
    _literacy_prompt,
    _score,
    expected_answer,
)

app = modal.App("neural-mirror-braille-curriculum")
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers==4.57.6",
        "huggingface_hub<1.0",
        "accelerate",
        "peft",
        "safetensors",
        "sentencepiece",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
    .add_local_file("braille_literacy.py", "/root/braille_literacy.py")
)

MODEL_MAP = {
    "qwen3:4b": "Qwen/Qwen3-4B",
}
TRAIN_SEEDS = list(range(1256, 1512))
TRAIN_EPOCHS = 1
GRADIENT_ACCUMULATION = 4
PARENT_ADAPTER = "/cache/braille-adapters/qwen3-4b-braille-literacy-v2"
ADAPTER_NAME = "qwen3-4b-braille-literacy-v3"


def _chat_prefix(tokenizer, prompt):
    messages = [
        {
            "role": "system",
            "content": "Exact protocol conformance test. Output only schema-valid JSON.",
        },
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _training_example(tokenizer, seed, max_length=8192):
    tasks = _build_tasks(seed)
    prompt = _literacy_prompt(tasks)
    answer = json.dumps(
        expected_answer(tasks),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prefix_ids = tokenizer(
        _chat_prefix(tokenizer, prompt),
        add_special_tokens=True,
    )["input_ids"]
    suffix = answer + (tokenizer.eos_token or "")
    answer_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    if len(prefix_ids) + len(answer_ids) > max_length:
        raise RuntimeError(
            f"seed {seed} tokenized to {len(prefix_ids) + len(answer_ids)} tokens; "
            f"limit is {max_length}"
        )
    return {
        "seed": seed,
        "input_ids": prefix_ids + answer_ids,
        "labels": [-100] * len(prefix_ids) + answer_ids,
        "prompt_tokens": len(prefix_ids),
        "answer_tokens": len(answer_ids),
    }


def _evaluate_exam(model, tokenizer, device):
    import torch

    model.eval()
    results = []
    for seed in EXAM_SEEDS:
        tasks = _build_tasks(seed)
        prompt = _literacy_prompt(tasks)
        prefix = _chat_prefix(tokenizer, prompt)
        encoded = tokenizer(prefix, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]
        started = time.time()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=2200,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            content = tokenizer.decode(
                generated[0, input_length:], skip_special_tokens=True
            ).strip()
            answer = json.loads(content)
            score = _score(tasks, answer)
            score.update({
                "seed": seed,
                "elapsed_seconds": round(time.time() - started, 1),
                "generated_tokens": int(generated.shape[-1] - input_length),
                "raw_response": content,
            })
        except Exception as error:
            score = {
                "seed": seed,
                "passed": False,
                "correct": 0,
                "total": TASKS_PER_SEED,
                "error": str(error),
                "elapsed_seconds": round(time.time() - started, 1),
            }
        results.append(score)
    return results


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    cpu=8,
    memory=49152,
    timeout=3600,
)
def train_and_test(model_name: str = "qwen3:4b") -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if set(TRAIN_SEEDS) & set(EXAM_SEEDS):
        raise RuntimeError("training seed leakage into immutable exam")
    hf_name = MODEL_MAP.get(model_name)
    if not hf_name:
        return {"model": model_name, "error": "No curriculum mapping"}

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name, cache_dir="/cache/hf", trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = [_training_example(tokenizer, seed) for seed in TRAIN_SEEDS]
    token_stats = {
        "minimum": min(len(item["input_ids"]) for item in examples),
        "maximum": max(len(item["input_ids"]) for item in examples),
        "mean": round(sum(len(item["input_ids"]) for item in examples) / len(examples), 1),
    }

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        cache_dir="/cache/hf",
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if not os.path.exists(os.path.join(PARENT_ADAPTER, "adapter_config.json")):
        raise RuntimeError(f"parent adapter is missing: {PARENT_ADAPTER}")
    model = PeftModel.from_pretrained(
        model,
        PARENT_ADAPTER,
        is_trainable=True,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-5,
        weight_decay=0.01,
    )

    rng = random.Random(4242)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    started = time.time()
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for position, example_index in enumerate(order):
            example = examples[example_index]
            input_ids = torch.tensor([example["input_ids"]], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor([example["labels"]], device="cuda")
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = output.loss / GRADIENT_ACCUMULATION
            loss.backward()
            losses.append(float(output.loss.detach()))
            micro_step += 1
            final_micro_step = epoch == TRAIN_EPOCHS - 1 and position == len(order) - 1
            if micro_step % GRADIENT_ACCUMULATION == 0 or final_micro_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    adapter_path = f"/cache/braille-adapters/{ADAPTER_NAME}"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    model_cache.commit()

    model.config.use_cache = True
    exam = _evaluate_exam(model, tokenizer, torch.device("cuda"))
    passed = all(item["passed"] for item in exam)
    result = {
        "model": model_name,
        "adapter": ADAPTER_NAME,
        "hf_model": hf_name,
        "passed": passed,
        "correct": sum(item["correct"] for item in exam),
        "total": sum(item["total"] for item in exam),
        "training": {
            "train_seeds": TRAIN_SEEDS,
            "exam_seeds": EXAM_SEEDS,
            "seed_overlap": [],
            "examples": len(examples),
            "epochs": TRAIN_EPOCHS,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "optimizer_steps": (micro_step + GRADIENT_ACCUMULATION - 1) // GRADIENT_ACCUMULATION,
            "learning_rate": 5e-5,
            "parent_adapter": PARENT_ADAPTER,
            "first_loss": round(losses[0], 6),
            "last_loss": round(losses[-1], 6),
            "mean_last_8_loss": round(sum(losses[-8:]) / min(8, len(losses)), 6),
            "token_lengths": token_stats,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "elapsed_seconds": round(time.time() - started, 1),
            "adapter_path": adapter_path,
        },
        "exam": exam,
    }

    # Persist inside the Modal Volume before returning. A detached training job
    # therefore retains its exam and audit evidence even if the local client
    # disconnects while the function is running.
    remote_result_path = f"/cache/braille-results/{ADAPTER_NAME}.json"
    os.makedirs(os.path.dirname(remote_result_path), exist_ok=True)
    temporary = remote_result_path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, remote_result_path)
    model_cache.commit()
    result["remote_result_path"] = remote_result_path
    return result


@app.local_entrypoint()
def main(model: str = "qwen3:4b"):
    print("━" * 72)
    print("  ⠿ Neural Mirror — Braille Literacy Curriculum")
    print(f"  Model: {model}")
    print(f"  Training seeds: {TRAIN_SEEDS[0]}..{TRAIN_SEEDS[-1]} ({len(TRAIN_SEEDS)})")
    print(f"  Immutable exam seeds: {EXAM_SEEDS}")
    print("━" * 72)

    try:
        result = train_and_test.remote(model)
    except Exception as error:
        result = {"model": model, "passed": False, "correct": 0, "total": 28, "error": str(error)}

    if "training" in result:
        training = result["training"]
        print(
            f"  Training: loss {training['first_loss']:.4f} → {training['last_loss']:.4f}; "
            f"{training['optimizer_steps']} optimizer steps; {training['elapsed_seconds']:.0f}s"
        )
    print(
        f"  {'✅ ADMIT' if result.get('passed') else '❌ EXCLUDE'} "
        f"{result.get('adapter', model)}: {result.get('correct', 0)}/{result.get('total', 28)}"
    )
    for seed_result in result.get("exam", []):
        print(
            f"     seed {seed_result['seed']}: "
            f"{seed_result['correct']}/{seed_result['total']}"
            + (f" error={seed_result.get('error')}" if seed_result.get('error') else "")
        )

    path = os.path.expanduser("~/neural-mirror/braille_curriculum_results.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    print(f"  Results: {path}")

    # Admission is adapter-specific. Never replace the failed base-model entry.
    admission_path = os.path.expanduser("~/neural-mirror/braille_admission.json")
    if result.get("passed") and os.path.exists(admission_path):
        with open(admission_path) as handle:
            admission = json.load(handle)
        identity = f"{model}+{result['adapter']}"
        if identity not in admission["admitted"]:
            admission["admitted"].append(identity)
        temporary = admission_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(admission, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, admission_path)
        print(f"  Admission roster updated: {identity}")
