"""Train checkpoint-specific braille-literacy adapters across model families.

Each model receives the same fixed curriculum before the immutable 28-task exam.
Adapters are never shared between checkpoints. Every remote worker persists its
own result to the Modal Volume, including technical or access failures.

Launch a detached batch:
  MODAL_PROFILE=salus modal run --detach braille_fleet.py \
    --models 'phi4-mini,mistral:7b,llama3.2:3b'

Collect completed results and update the local admission roster:
  MODAL_PROFILE=salus modal run braille_fleet.py --collect
"""

import json
import os
import random
import re
import time
import traceback

import modal

from braille_literacy import (
    SEEDS as EXAM_SEEDS,
    TASKS_PER_SEED,
    _build_tasks,
    _literacy_prompt,
    _score,
    expected_answer,
)

app = modal.App("neural-mirror-braille-fleet")
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
        "protobuf",
        "tiktoken",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
    .add_local_file("braille_literacy.py", "/root/braille_literacy.py")
    .add_local_file("adapter_introspect.py", "/root/adapter_introspect.py")
)

# One exact upstream checkpoint per Ollama identity. `access` records known
# license gates; workers still attempt loading and preserve the actual error.
MODEL_SPECS = {
    "phi4-mini": {"hf": "microsoft/phi-4-mini-instruct"},
    "mistral:7b": {"hf": "mistralai/Mistral-7B-Instruct-v0.3"},
    "gemma3:4b": {
        "hf": "unsloth/gemma-3-4b-it",
        "model_class": "image_text",
        "access": "public_mirror",
    },
    "llama3.2:3b": {"hf": "unsloth/Llama-3.2-3B-Instruct"},
    "deepseek-v2:16b": {
        "hf": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "trust_remote_code": True,
        "rank": 16,
        "target_modules": ["q_proj", "o_proj"],
    },
    "granite3.3:8b": {"hf": "ibm-granite/granite-3.3-8b-instruct"},
    "command-r7b:7b": {"hf": "CohereForAI/c4ai-command-r7b-12-2024", "access": "auto"},
    "olmo2:7b": {"hf": "allenai/OLMo-2-1124-7B-Instruct"},
    "falcon3:7b": {"hf": "tiiuae/Falcon3-7B-Instruct"},
    "glm4:9b": {"hf": "THUDM/glm-4-9b-chat-hf", "trust_remote_code": True},
    "aya-expanse:8b": {"hf": "CohereForAI/aya-expanse-8b", "access": "auto"},
    "exaone3.5:7.8b": {
        "hf": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "trust_remote_code": True,
        "revision": "0ff6b5ec7c13",
    },
    "stablelm2:1.6b": {"hf": "stabilityai/stablelm-2-1_6b-chat"},
    "yi:6b": {"hf": "01-ai/Yi-6B-Chat"},
    "internlm2:7b": {
        "hf": "internlm/internlm2-chat-7b",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "smollm2:1.7b": {"hf": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
    "solar:10.7b": {"hf": "upstage/SOLAR-10.7B-Instruct-v1.0"},
    "starcoder2:7b": {"hf": "bigcode/starcoder2-7b", "base_model": True},
    "mixtral:latest": {"hf": "mistralai/Mixtral-8x7B-Instruct-v0.1", "large": True},
    "granite4:3b": {"hf": "ibm-granite/granite-4.0-h-micro", "trust_remote_code": True},
    "nemotron-3-nano:4b": {
        "hf": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "trust_remote_code": True,
        "large": True,
    },
    "gpt-oss:120b": {"hf": "openai/gpt-oss-120b", "large": True, "research_lead": True},
}

CORE_BATCH = ["phi4-mini", "mistral:7b", "gemma3:4b", "llama3.2:3b"]
WIDE_DENSE_BATCH = [
    "deepseek-v2:16b", "granite3.3:8b", "command-r7b:7b", "olmo2:7b",
    "falcon3:7b", "glm4:9b", "aya-expanse:8b", "exaone3.5:7.8b",
    "stablelm2:1.6b", "yi:6b", "internlm2:7b", "smollm2:1.7b",
    "solar:10.7b", "starcoder2:7b", "granite4:3b",
]
LARGE_BATCH = ["mixtral:latest", "nemotron-3-nano:4b", "gpt-oss:120b"]
REFINE_MODELS = [
    "phi4-mini", "mistral:7b", "falcon3:7b", "solar:10.7b", "gemma3:4b",
    "olmo2:7b", "yi:6b", "starcoder2:7b", "smollm2:1.7b", "stablelm2:1.6b",
    "granite4:3b",
]
REFINE_SEEDS = list(range(1512, 1768))

# Fixed in advance. Exam seeds never appear here. No exam-driven continuation.
STAGES = [
    {"seeds": list(range(1000, 1256)), "epochs": 2, "learning_rate": 1e-4},
    {"seeds": list(range(1256, 1512)), "epochs": 1, "learning_rate": 5e-5},
]
GRADIENT_ACCUMULATION = 4
PREFERRED_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "gate_up_proj", "query_key_value", "dense_h_to_4h", "dense_4h_to_h",
    "c_attn", "c_proj", "fc1", "fc2", "Wqkv", "out_proj",
]


def _slug(model_name):
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-")


def _chat_prefix(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "Exact protocol conformance test. Output only schema-valid JSON."},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "System: Exact protocol conformance test. Output only schema-valid JSON.\n"
        "User: /no_think\n" + prompt + "\nAssistant:"
    )


def _example(tokenizer, seed, context_limit):
    tasks = _build_tasks(seed)
    prompt = _literacy_prompt(tasks)
    answer = json.dumps(expected_answer(tasks), ensure_ascii=False, separators=(",", ":"))
    prefix_ids = tokenizer(_chat_prefix(tokenizer, prompt), add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    length = len(prefix_ids) + len(answer_ids)
    if length > context_limit:
        raise RuntimeError(f"CONTEXT: seed {seed} requires {length} tokens but model limit is {context_limit}")
    return {"input_ids": prefix_ids + answer_ids, "labels": [-100] * len(prefix_ids) + answer_ids}


def _context_limit(model, tokenizer):
    config = getattr(model.config, "text_config", model.config)
    model_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and 512 <= model_limit < 1_000_000:
        return min(8192, model_limit)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 512 <= tokenizer_limit < 1_000_000:
        return min(8192, tokenizer_limit)
    return 8192


def _target_modules(model):
    import torch

    leaves = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in {"lm_head", "output", "classifier"}:
                leaves.add(leaf)
    targets = [name for name in PREFERRED_TARGETS if name in leaves]
    if not targets:
        raise RuntimeError(f"TARGET_MODULES: no supported linear projection names; found {sorted(leaves)[:80]}")
    return targets


def _evaluate(model, tokenizer):
    import torch

    model.eval()
    outcomes = []
    for seed in EXAM_SEEDS:
        tasks = _build_tasks(seed)
        encoded = tokenizer(_chat_prefix(tokenizer, _literacy_prompt(tasks)), return_tensors="pt")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
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
                    use_cache=False,
                )
            raw = tokenizer.decode(generated[0, input_length:], skip_special_tokens=True).strip()
            score = _score(tasks, json.loads(raw))
            score.update({
                "seed": seed,
                "elapsed_seconds": round(time.time() - started, 1),
                "generated_tokens": int(generated.shape[-1] - input_length),
                "raw_response": raw,
            })
        except Exception as error:
            score = {
                "seed": seed,
                "passed": False,
                "correct": 0,
                "total": TASKS_PER_SEED,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.time() - started, 1),
            }
        outcomes.append(score)
    return outcomes


def _persist(result):
    path = f"/cache/braille-fleet-results/{_slug(result['model'])}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    model_cache.commit()
    result["remote_result_path"] = path
    return result


def _train_one(model_name):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    spec = MODEL_SPECS.get(model_name)
    if not spec:
        raise RuntimeError("REGISTRY: no Hugging Face checkpoint mapping")
    if spec.get("large"):
        raise RuntimeError("LARGE_MODEL: requires a separately validated quantized training path")
    if any(set(stage["seeds"]) & set(EXAM_SEEDS) for stage in STAGES):
        raise RuntimeError("SEED_LEAKAGE: curriculum overlaps immutable exam")

    trust = spec.get("trust_remote_code", False)
    revision = spec.get("revision")
    if model_name == "internlm2:7b":
        from huggingface_hub import hf_hub_download
        from transformers import LlamaTokenizer
        vocab_file = hf_hub_download(spec["hf"], "tokenizer.model", cache_dir="/cache/hf")
        tokenizer = LlamaTokenizer(vocab_file=vocab_file, legacy=True)
        tokenizer.chat_template = (
            "{% for message in messages %}{{ '<s>' + message['role'] + '\\n' + message['content'] + '</s>\\n' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<s>assistant\\n' }}{% endif %}"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            spec["hf"], cache_dir="/cache/hf", trust_remote_code=trust,
            use_fast=spec.get("use_fast", True), revision=revision,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_loader = (
        AutoModelForImageTextToText
        if spec.get("model_class") == "image_text"
        else AutoModelForCausalLM
    )
    base = model_loader.from_pretrained(
        spec["hf"],
        cache_dir="/cache/hf",
        dtype=torch.bfloat16,
        trust_remote_code=trust,
        low_cpu_mem_usage=True,
        revision=revision,
    ).to("cuda")
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    targets = _target_modules(base)
    if spec.get("target_modules"):
        targets = [name for name in spec["target_modules"] if name in targets]
        if not targets:
            raise RuntimeError("TARGET_MODULES: configured projections are absent")
    rank = spec.get("rank", 32)
    lora = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(base, lora)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    context_limit = _context_limit(base, tokenizer)
    stage_records = []
    all_losses = []
    started = time.time()

    for stage_index, stage in enumerate(STAGES, start=1):
        examples = [_example(tokenizer, seed, context_limit) for seed in stage["seeds"]]
        optimizer = torch.optim.AdamW(parameters, lr=stage["learning_rate"], weight_decay=0.01)
        optimizer.zero_grad(set_to_none=True)
        rng = random.Random(4242 + stage_index)
        stage_losses = []
        micro_step = 0
        model.train()
        for epoch in range(stage["epochs"]):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for position, example_index in enumerate(order):
                example = examples[example_index]
                input_ids = torch.tensor([example["input_ids"]], device="cuda")
                labels = torch.tensor([example["labels"]], device="cuda")
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    labels=labels,
                    use_cache=False,
                )
                (output.loss / GRADIENT_ACCUMULATION).backward()
                value = float(output.loss.detach())
                stage_losses.append(value)
                all_losses.append(value)
                micro_step += 1
                final = epoch == stage["epochs"] - 1 and position == len(order) - 1
                if micro_step % GRADIENT_ACCUMULATION == 0 or final:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        stage_records.append({
            "stage": stage_index,
            "seed_start": stage["seeds"][0],
            "seed_end": stage["seeds"][-1],
            "examples": len(examples),
            "epochs": stage["epochs"],
            "learning_rate": stage["learning_rate"],
            "first_loss": round(stage_losses[0], 6),
            "last_loss": round(stage_losses[-1], 6),
            "mean_last_8_loss": round(sum(stage_losses[-8:]) / min(8, len(stage_losses)), 6),
        })
        del examples, optimizer
        torch.cuda.empty_cache()

    adapter_name = f"{_slug(model_name)}-braille-literacy-v1"
    adapter_path = f"/cache/braille-adapters/{adapter_name}"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    model_cache.commit()

    model.config.use_cache = True
    exam = _evaluate(model, tokenizer)

    # The adapter is a first-class introspection target, separate from GGUF.
    # These bounded host measurements can be supplied to the adapted model as
    # citation-addressable evidence without asking it to invent tensor facts.
    from adapter_introspect import LoRAAdapter
    adapter_reader = LoRAAdapter(adapter_path)
    adapter_profile = {
        "overview": adapter_reader.inspect_self(),
        "layer_fingerprint": adapter_reader.layer_fingerprint(),
        "evidence_packet": adapter_reader.evidence_packet(),
    }
    return {
        "model": model_name,
        "hf_model": spec["hf"],
        "adapter": adapter_name,
        "adapter_path": adapter_path,
        "status": "admitted" if all(item["passed"] for item in exam) else "excluded",
        "passed": all(item["passed"] for item in exam),
        "correct": sum(item["correct"] for item in exam),
        "total": sum(item["total"] for item in exam),
        "exam_seeds": EXAM_SEEDS,
        "target_modules": targets,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "context_limit": context_limit,
        "stages": stage_records,
        "elapsed_seconds": round(time.time() - started, 1),
        "exam": exam,
        "adapter_introspection": adapter_profile,
    }


def _refine_one(model_name):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if model_name not in REFINE_MODELS:
        raise RuntimeError(f"REFINE_REGISTRY: unsupported model {model_name}")
    if set(REFINE_SEEDS) & set(EXAM_SEEDS):
        raise RuntimeError("SEED_LEAKAGE: refinement overlaps immutable exam")
    spec = MODEL_SPECS[model_name]
    trust = spec.get("trust_remote_code", False)
    parent_name = f"{_slug(model_name)}-braille-literacy-v1"
    parent_path = f"/cache/braille-adapters/{parent_name}"
    if not os.path.isfile(os.path.join(parent_path, "adapter_config.json")):
        raise RuntimeError(f"PARENT_ADAPTER: missing {parent_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        spec["hf"], cache_dir="/cache/hf", trust_remote_code=trust,
        use_fast=spec.get("use_fast", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = AutoModelForImageTextToText if spec.get("model_class") == "image_text" else AutoModelForCausalLM
    base = loader.from_pretrained(
        spec["hf"], cache_dir="/cache/hf", dtype=torch.bfloat16,
        trust_remote_code=trust, low_cpu_mem_usage=True,
    ).to("cuda")
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, parent_path, is_trainable=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    examples = [_example(tokenizer, seed, _context_limit(base, tokenizer)) for seed in REFINE_SEEDS]
    optimizer = torch.optim.AdamW(parameters, lr=2.5e-5, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    rng = random.Random(5252)
    order = list(range(len(examples)))
    rng.shuffle(order)
    started = time.time()
    model.train()
    for position, example_index in enumerate(order):
        example = examples[example_index]
        input_ids = torch.tensor([example["input_ids"]], device="cuda")
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=torch.tensor([example["labels"]], device="cuda"),
            use_cache=False,
        )
        (output.loss / GRADIENT_ACCUMULATION).backward()
        losses.append(float(output.loss.detach()))
        final = position == len(order) - 1
        if (position + 1) % GRADIENT_ACCUMULATION == 0 or final:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    adapter_name = f"{_slug(model_name)}-braille-literacy-v2"
    adapter_path = f"/cache/braille-adapters/{adapter_name}"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    model_cache.commit()
    model.config.use_cache = True
    exam = _evaluate(model, tokenizer)

    from adapter_introspect import LoRAAdapter
    reader = LoRAAdapter(adapter_path)
    passed = all(item["passed"] for item in exam)
    return {
        "model": model_name,
        "hf_model": spec["hf"],
        "adapter": adapter_name,
        "adapter_path": adapter_path,
        "parent_adapter": parent_name,
        "status": "admitted" if passed else "excluded",
        "passed": passed,
        "correct": sum(item["correct"] for item in exam),
        "total": sum(item["total"] for item in exam),
        "exam_seeds": EXAM_SEEDS,
        "refinement": {
            "seed_start": REFINE_SEEDS[0],
            "seed_end": REFINE_SEEDS[-1],
            "examples": len(examples),
            "epochs": 1,
            "learning_rate": 2.5e-5,
            "first_loss": round(losses[0], 6),
            "last_loss": round(losses[-1], 6),
            "mean_last_8_loss": round(sum(losses[-8:]) / min(8, len(losses)), 6),
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "exam": exam,
        "adapter_introspection": {
            "overview": reader.inspect_self(),
            "layer_fingerprint": reader.layer_fingerprint(),
            "evidence_packet": reader.evidence_packet(),
        },
    }


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=7200,
    max_containers=15,
)
def train_model(model_name: str) -> dict:
    started = time.time()
    try:
        result = _train_one(model_name)
    except Exception as error:
        result = {
            "model": model_name,
            "status": "blocked",
            "passed": False,
            "correct": 0,
            "total": 28,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-8000:],
            "elapsed_seconds": round(time.time() - started, 1),
        }
    return _persist(result)


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=7200,
    max_containers=8,
)
def refine_model(model_name: str) -> dict:
    started = time.time()
    try:
        result = _refine_one(model_name)
    except Exception as error:
        result = {
            "model": model_name,
            "status": "blocked",
            "passed": False,
            "correct": 0,
            "total": 28,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc()[-8000:],
            "elapsed_seconds": round(time.time() - started, 1),
        }
    return _persist(result)


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=3600,
    max_containers=4,
)
def reevaluate_model(model_name: str, adapter_name: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    spec = MODEL_SPECS[model_name]
    trust = spec.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["hf"], cache_dir="/cache/hf", trust_remote_code=trust,
        use_fast=spec.get("use_fast", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = AutoModelForImageTextToText if spec.get("model_class") == "image_text" else AutoModelForCausalLM
    base = loader.from_pretrained(
        spec["hf"], cache_dir="/cache/hf", dtype=torch.bfloat16,
        trust_remote_code=trust, low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, f"/cache/braille-adapters/{adapter_name}")
    model.config.use_cache = False
    exam = _evaluate(model, tokenizer)
    result_path = f"/cache/braille-fleet-results/{_slug(model_name)}.json"
    with open(result_path) as handle:
        result = json.load(handle)
    result["exam"] = exam
    result["correct"] = sum(item["correct"] for item in exam)
    result["total"] = sum(item["total"] for item in exam)
    result["passed"] = all(item["passed"] for item in exam)
    result["status"] = "admitted" if result["passed"] else "excluded"
    result["reevaluated_without_kv_cache"] = True
    return _persist(result)


@app.function(image=image, volumes={"/cache": model_cache}, cpu=4, memory=16384, timeout=900)
def inspect_adapter(adapter_name: str) -> dict:
    from adapter_introspect import LoRAAdapter

    path = f"/cache/braille-adapters/{adapter_name}"
    reader = LoRAAdapter(path)
    result = {
        "adapter": adapter_name,
        "overview": reader.inspect_self(),
        "layer_fingerprint": reader.layer_fingerprint(),
        "evidence_packet": reader.evidence_packet(),
    }
    output = f"/cache/braille-adapter-profiles/{adapter_name}.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output + ".tmp", "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(output + ".tmp", output)
    model_cache.commit()
    return result


@app.function(image=image, volumes={"/cache": model_cache}, timeout=300)
def read_results() -> list:
    directory = "/cache/braille-fleet-results"
    if not os.path.isdir(directory):
        return []
    results = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename)) as handle:
                results.append(json.load(handle))
    return results


def _update_admission(results):
    path = os.path.expanduser("~/neural-mirror/braille_admission.json")
    with open(path) as handle:
        admission = json.load(handle)
    for result in results:
        model = result["model"]
        if result.get("passed"):
            identity = f"{model}+{result['adapter']}"
            if identity not in admission["admitted"]:
                admission["admitted"].append(identity)
        elif result.get("status") == "blocked":
            identity = f"{model}: {result.get('error', 'blocked')}"
            if identity not in admission["inconclusive"]:
                admission["inconclusive"].append(identity)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(admission, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _selection(models, preset):
    if models:
        return [item.strip() for item in models.split(",") if item.strip()]
    if preset == "core":
        return CORE_BATCH
    if preset == "wide":
        return WIDE_DENSE_BATCH
    if preset == "large":
        return LARGE_BATCH
    raise ValueError("provide --models or choose --preset core|wide|large")


@app.local_entrypoint()
def main(
    models: str = "",
    preset: str = "core",
    collect: bool = False,
    adapter: str = "",
    refine: bool = False,
    reevaluate: str = "",
):
    if reevaluate:
        if not adapter:
            raise ValueError("--reevaluate requires --adapter")
        result = reevaluate_model.remote(reevaluate, adapter)
        print(f"Reevaluated {reevaluate}: {result['correct']}/{result['total']} {result['status']}")
        return

    if adapter:
        result = inspect_adapter.remote(adapter)
        output = os.path.expanduser(f"~/neural-mirror/{adapter}-profile.json")
        with open(output, "w") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Adapter profile: {output}")
        print(json.dumps(result["evidence_packet"], indent=2, ensure_ascii=False))
        return

    if collect:
        results = read_results.remote()
        _update_admission(results)
        output = os.path.expanduser("~/neural-mirror/braille_fleet_results.json")
        with open(output + ".tmp", "w") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        os.replace(output + ".tmp", output)
        print(f"Collected {len(results)} result(s) into {output}")
        for result in results:
            print(f"  {result['model']}: {result['status']} {result.get('correct', 0)}/{result.get('total', 28)}")
        return

    selected = REFINE_MODELS if refine and not models else _selection(models, preset)
    unknown = [model for model in selected if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unmapped models: {unknown}")
    worker = refine_model if refine else train_model
    mode = "refinement" if refine else "curriculum"
    print(f"Launching {len(selected)} detached {mode} workers")
    for model in selected:
        call = worker.spawn(model)
        print(f"  {model}: {call.object_id}")
    print("Workers persist results under /cache/braille-fleet-results; use --collect later.")
