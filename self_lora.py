"""
Neural Mirror — Self-LoRA 🪞🧬

A model inspects its own weights, proposes LoRA modifications,
trains them, and evaluates whether it improved.

The recursive loop:
  1. Model examines its own structure (introspection tools)
  2. Model proposes a LoRA config (which layers, what rank, what to target)
  3. System trains the LoRA on a task the model chose
  4. Model compares before/after weight stats
  5. Model evaluates: did it work? What next?

Usage:
  modal run self_lora.py
  modal run self_lora.py --base-model qwen3:1.7b --target "improve reasoning"
"""

import modal
import json, time, os, sys

app = modal.App("neural-mirror-self-lora")
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "procps", "git")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "torch", "transformers", "peft", "datasets", "accelerate",
        "bitsandbytes", "safetensors", "gguf", "trl", "huggingface_hub",
    )
    .add_local_file("introspect.py", "/app/introspect.py")
)


# ── Helpers ──────────────────────────────────────────────────

def start_ollama():
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    proc = subprocess.Popen(["ollama", "serve"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return proc
        except:
            time.sleep(1)
    raise RuntimeError("Ollama failed to start")


def pull_model(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    subprocess.run(["ollama", "pull", name], env=env,
                  capture_output=True, text=True, timeout=600)


def find_gguf(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    r = subprocess.run(["ollama", "show", name, "--modelfile"],
                      env=env, capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if line.startswith('FROM /'):
            p = line[5:].strip()
            if os.path.isfile(p):
                return p
    return None


def ask_model(model_name, system, prompt, host="http://localhost:11434"):
    """Simple non-tool query to the model."""
    import urllib.request
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    content = result.get('message', {}).get('content', '')
    return content.replace('<think>', '').replace('</think>', '').strip()


# ── Phase 1: Introspect & Propose ────────────────────────────

@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
    memory=16384,
    cpu=4,
)
def introspect_and_propose(model_name: str, target_task: str) -> dict:
    """Model examines itself and proposes a LoRA configuration."""
    sys.path.insert(0, "/app")
    from introspect import GGUFModel, load_model, tool_inspect_self, tool_weight_fingerprint, tool_compare_layers
    import introspect

    proc = start_ollama()
    pull_model(model_name)
    gguf_path = find_gguf(model_name)

    # Parse GGUF
    introspect._model = None
    m = load_model(gguf_path)
    self_info = tool_inspect_self()
    fingerprint = tool_weight_fingerprint()

    meta = m.metadata
    arch = meta.get('general.architecture', '?')
    n_layers = meta.get(f'{arch}.block_count', 28)
    embed_dim = meta.get(f'{arch}.embedding_length', 2048)
    n_heads = meta.get(f'{arch}.attention.head_count', 16)
    n_kv = meta.get(f'{arch}.attention.head_count_kv', 8)

    # Get norm curve for context
    curve = fingerprint.get('fingerprint', {})
    curve_summary = []
    for k, v in sorted(curve.items(), key=lambda x: int(x[0].split('_')[1])):
        ln = int(k.split('_')[1])
        curve_summary.append(f"L{ln}: abs_mean={v['abs_mean']:.3f}")

    # Warm model
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
    except:
        pass

    # Ask the model to propose a LoRA config
    system = f"""You are {self_info['identity']['name']}, a {self_info['scale']['total_parameters_human']} model.
You are about to receive LoRA fine-tuning to improve at: {target_task}

Your architecture:
- {n_layers} transformer blocks, {embed_dim}-dim embeddings
- {n_heads} attention heads, {n_kv} KV heads (GQA)
- Quantization: {json.dumps(self_info.get('quantization', {}))}

Your norm curve (how weight magnitudes change with depth):
{chr(10).join(curve_summary)}

You must propose a LoRA configuration as JSON. Think about:
- Which layers to target (early=syntax, middle=semantics, late=generation)
- Which modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- What rank (4=minimal, 16=moderate, 64=heavy)
- What alpha (scaling factor, typically 2x rank)
- Whether to use dropout

Respond with ONLY a JSON object like:
{{
  "target_modules": ["q_proj", "v_proj"],
  "layers_to_target": "all",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "reasoning": "explanation of why these choices"
}}

Do NOT use <think> tags. Respond with JSON only."""

    proposal_text = ask_model(model_name, system,
        f"Propose your LoRA configuration for improving at: {target_task}")

    # Parse the JSON from response
    try:
        # Find JSON in response
        start = proposal_text.find('{')
        end = proposal_text.rfind('}') + 1
        if start >= 0 and end > start:
            proposal = json.loads(proposal_text[start:end])
        else:
            proposal = {
                "target_modules": ["q_proj", "v_proj"],
                "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                "reasoning": "default config (model didn't return valid JSON)"
            }
    except:
        proposal = {
            "target_modules": ["q_proj", "v_proj"],
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
            "reasoning": f"parse error, raw: {proposal_text[:200]}"
        }

    proc.terminate()

    return {
        "model": model_name,
        "architecture": self_info,
        "fingerprint_sample": {k: v for k, v in list(curve.items())[:3] + list(curve.items())[-3:]},
        "proposal": proposal,
        "target_task": target_task,
    }


# ── Phase 2: Train the LoRA ──────────────────────────────────

@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="A10G",
    timeout=1200,
    memory=32768,
)
def train_lora(model_name: str, proposal: dict, target_task: str) -> dict:
    """Actually train a LoRA adapter based on the model's own proposal."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    print(f"🧬 Training LoRA for {model_name}")
    print(f"   Task: {target_task}")
    print(f"   Config: {json.dumps(proposal, indent=2)}")

    # Map Ollama model names to HuggingFace
    HF_MAP = {
        "qwen3:1.7b": "Qwen/Qwen3-1.7B",
        "qwen3:0.6b": "Qwen/Qwen3-0.6B",
        "qwen3:4b": "Qwen/Qwen3-4B",
        "qwen3:8b": "Qwen/Qwen3-8B",
        "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
        "phi4-mini": "microsoft/phi-4-mini-instruct",
        "gemma3:4b": "google/gemma-3-4b-it",
    }

    hf_name = HF_MAP.get(model_name)
    if not hf_name:
        return {"error": f"No HF mapping for {model_name}"}

    print(f"   HF model: {hf_name}")

    # Load model in 4-bit
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir="/cache/hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        cache_dir="/cache/hf",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )

    print(f"   Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Snapshot pre-LoRA weight stats
    pre_stats = {}
    for name, param in model.named_parameters():
        if 'layers.0.' in name and param.requires_grad:
            pre_stats[name] = {
                'mean': param.data.float().mean().item(),
                'std': param.data.float().std().item(),
            }

    # Configure LoRA from proposal
    target_modules = proposal.get('target_modules', ['q_proj', 'v_proj'])
    r = proposal.get('r', 16)
    alpha = proposal.get('lora_alpha', 32)
    dropout = proposal.get('lora_dropout', 0.05)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   LoRA applied: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total ({trainable/total*100:.2f}%)")

    # Generate training data based on target task
    train_examples = generate_training_data(target_task, tokenizer)
    print(f"   Training examples: {len(train_examples)}")

    # Train
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    training_args = TrainingArguments(
        output_dir="/cache/lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("   Training...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.0f}s. Loss: {result.training_loss:.4f}")

    # Post-LoRA weight stats
    post_stats = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            post_stats[name] = {
                'mean': param.data.float().mean().item(),
                'std': param.data.float().std().item(),
                'norm': param.data.float().norm().item(),
            }

    # Save adapter
    adapter_path = "/cache/lora_output/adapter"
    model.save_pretrained(adapter_path)
    print(f"   Adapter saved to {adapter_path}")

    # Quick eval: generate before/after on some test prompts
    eval_prompts = generate_eval_prompts(target_task)
    pre_responses = []
    post_responses = []

    model.eval()
    for p in eval_prompts[:3]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        post_responses.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return {
        "model": model_name,
        "hf_model": hf_name,
        "lora_config": {
            "r": r, "alpha": alpha, "dropout": dropout,
            "target_modules": target_modules,
        },
        "trainable_params": f"{trainable/1e6:.1f}M",
        "trainable_pct": f"{trainable/total*100:.2f}%",
        "training_loss": result.training_loss,
        "training_time": f"{elapsed:.0f}s",
        "lora_weight_stats": post_stats,
        "sample_outputs": post_responses[:3],
    }


def generate_training_data(task, tokenizer, n=50):
    """Generate simple training examples for a task."""
    from datasets import Dataset

    if 'reason' in task.lower() or 'math' in task.lower():
        examples = [
            {"text": f"Question: What is {a} + {b}?\nAnswer: {a} + {b} = {a+b}. The answer is {a+b}."}
            for a in range(1, 26) for b in range(1, 3)
        ][:n]
    elif 'code' in task.lower():
        examples = [
            {"text": "Question: Write a function to add two numbers.\nAnswer: def add(a, b): return a + b"},
            {"text": "Question: Write a function to find the maximum.\nAnswer: def find_max(lst): return max(lst)"},
        ] * (n // 2)
    else:
        examples = [
            {"text": f"Instruction: {task}\nResponse: I will follow this instruction carefully and provide a helpful response."}
        ] * n

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=128, padding="max_length")

    dataset = Dataset.from_list(examples[:n])
    dataset = dataset.map(tokenize, remove_columns=["text"])
    return dataset


def generate_eval_prompts(task):
    if 'reason' in task.lower() or 'math' in task.lower():
        return [
            "What is 17 + 28?",
            "If I have 5 apples and give away 2, how many do I have?",
            "What is 100 - 37?",
        ]
    elif 'code' in task.lower():
        return [
            "Write a function to reverse a string.",
            "Write a function to check if a number is prime.",
        ]
    return [
        f"Please help me with: {task}",
        "Explain your approach.",
    ]


# ── Phase 3: Self-Evaluate ───────────────────────────────────

@app.function(
    image=image,
    volumes={"/cache": model_cache},
    timeout=600,
    memory=16384,
    cpu=4,
)
def self_evaluate(model_name: str, proposal: dict, training_result: dict, introspection: dict) -> dict:
    """Model evaluates its own LoRA training results."""

    proc = start_ollama()
    pull_model(model_name)

    # Warm up
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
    except:
        pass

    system = f"""You are {model_name}. You just proposed and trained a LoRA adapter on yourself.

Your proposal was:
{json.dumps(proposal, indent=2)}

Training results:
- Loss: {training_result.get('training_loss', '?')}
- Training time: {training_result.get('training_time', '?')}
- Trainable params: {training_result.get('trainable_params', '?')} ({training_result.get('trainable_pct', '?')})

LoRA weight statistics (the new weights added to your existing ones):
{json.dumps(training_result.get('lora_weight_stats', {}), indent=2)[:1500]}

Sample outputs after training:
{json.dumps(training_result.get('sample_outputs', []), indent=2)[:500]}

Evaluate honestly:
1. Did the training loss converge? Was it too high/low?
2. Look at the LoRA weight norms — are they reasonable?
3. If you could redo this, what would you change?
4. Propose your next iteration: different rank? different layers? different task?

Be specific and concise. No <think> tags."""

    evaluation = ask_model(model_name, system,
        "Evaluate your LoRA training results. Was your proposal good? What would you change?")

    proc.terminate()

    return {
        "model": model_name,
        "evaluation": evaluation,
    }


# ── Main Loop ────────────────────────────────────────────────

@app.local_entrypoint()
def main(base_model: str = "qwen3:1.7b", target: str = "improve mathematical reasoning"):
    print("━" * 65)
    print("  🪞🧬 Neural Mirror — Self-LoRA")
    print("  A model proposes, trains, and evaluates its own adapter")
    print("━" * 65)
    print()
    print(f"  Base model: {base_model}")
    print(f"  Target: {target}")
    print()

    # Phase 1: Introspect and propose
    print("━" * 65)
    print("  Phase 1: Self-Examination & LoRA Proposal")
    print("━" * 65)
    print()

    introspection = introspect_and_propose.remote(base_model, target)
    proposal = introspection['proposal']

    print(f"  Model: {introspection.get('architecture', {}).get('identity', {}).get('name', '?')}")
    print(f"  Proposed LoRA config:")
    print(f"    Target modules: {proposal.get('target_modules', '?')}")
    print(f"    Rank: {proposal.get('r', '?')}")
    print(f"    Alpha: {proposal.get('lora_alpha', '?')}")
    print(f"    Dropout: {proposal.get('lora_dropout', '?')}")
    print(f"    Reasoning: {proposal.get('reasoning', '?')[:200]}")
    print()

    # Phase 2: Train
    print("━" * 65)
    print("  Phase 2: Training the Self-Proposed LoRA")
    print("━" * 65)
    print()

    training_result = train_lora.remote(base_model, proposal, target)

    if 'error' in training_result:
        print(f"  ❌ Training error: {training_result['error']}")
        return

    print(f"  ✅ Training complete!")
    print(f"    Loss: {training_result.get('training_loss', '?')}")
    print(f"    Time: {training_result.get('training_time', '?')}")
    print(f"    Trainable: {training_result.get('trainable_params', '?')} ({training_result.get('trainable_pct', '?')})")
    print()

    # Show LoRA weight stats
    lora_stats = training_result.get('lora_weight_stats', {})
    if lora_stats:
        print(f"  LoRA adapter weight norms (first 5):")
        for name, stats in list(lora_stats.items())[:5]:
            short = name.split('.')[-3:]
            print(f"    {'.'.join(short)}: norm={stats.get('norm', '?'):.4f}")
    print()

    # Phase 3: Self-evaluate
    print("━" * 65)
    print("  Phase 3: Self-Evaluation")
    print("━" * 65)
    print()

    eval_result = self_evaluate.remote(base_model, proposal, training_result, introspection)

    print(f"  🪞 {base_model}'s evaluation of its own LoRA:")
    print()
    for line in eval_result.get('evaluation', '').split('\n'):
        if line.strip():
            print(f"    {line.strip()}")
    print()

    # Summary
    print("━" * 65)
    print("  🧬 Self-LoRA Complete")
    print("━" * 65)
    print(f"  Model: {base_model}")
    print(f"  Task: {target}")
    print(f"  Self-proposed: r={proposal.get('r')}, modules={proposal.get('target_modules')}")
    print(f"  Loss: {training_result.get('training_loss', '?')}")
    print(f"  The model has examined its own weights, proposed modifications,")
    print(f"  trained them, and evaluated the result.")
    print()
