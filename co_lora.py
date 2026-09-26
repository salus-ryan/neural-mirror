"""
Neural Mirror — Co-LoRA 🪞🧬🤝

Multiple models collaboratively design LoRA adapters for each other.

Round 1: Each model inspects itself, proposes LoRA for ITSELF
Round 2: Each model sees all proposals, proposes LoRA for ANOTHER model
         ("I think Mistral should target X because I noticed Y in its weights")
Round 3: Train all proposed LoRAs in parallel on GPU
Round 4: Models evaluate each other's results, propose next iteration

Usage:
  modal run co_lora.py
  modal run co_lora.py --target "improve mathematical reasoning"
  modal run co_lora.py --models "qwen3:1.7b,llama3.2:3b,phi4-mini"
"""

import modal
import json, time, os, sys

app = modal.App("neural-mirror-co-lora")
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

HF_MAP = {
    "qwen3:1.7b": "Qwen/Qwen3-1.7B",
    "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4-mini": "microsoft/phi-4-mini-instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

DEFAULT_MODELS = ["qwen3:1.7b", "llama3.2:3b", "phi4-mini"]


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


def pull_and_warm(name):
    import subprocess, urllib.request
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    subprocess.run(["ollama", "pull", name], env=env,
                  capture_output=True, text=True, timeout=600)
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": name, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
    except:
        pass


def find_gguf(name):
    import subprocess
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    r = subprocess.run(["ollama", "show", name, "--modelfile"],
                      env=env, capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if line.startswith('FROM /'):
            p = line[5:].strip()
            if os.path.isfile(p): return p
    return None


def ask_model(model_name, system, prompt, host="http://localhost:11434"):
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
    return result.get('message', {}).get('content', '').replace('<think>','').replace('</think>','').strip()


def parse_json_from_text(text):
    """Extract JSON object from model response."""
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except:
            pass
    return None


def profile_gguf(path):
    sys.path.insert(0, "/app")
    from introspect import GGUFModel
    m = GGUFModel(path)
    meta = m.metadata
    arch = meta.get('general.architecture', '?')
    name = meta.get('general.name', '?')
    params = sum(t['n_elements'] for t in m.tensors.values())
    layers = set()
    for t in m.tensors:
        if 'blk.' in t:
            ps = t.split('.')
            for i, p in enumerate(ps):
                if p == 'blk' and i+1 < len(ps):
                    try: layers.add(int(ps[i+1]))
                    except: pass
    max_layer = max(layers) if layers else 0

    # Norm stats
    first_norms, last_norms = [], []
    for tname in m.tensors:
        if 'blk.0.' in tname and 'norm' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=128)
            if vals: first_norms.append(sum(vals)/len(vals))
        if f'blk.{max_layer}.' in tname and 'norm' in tname:
            vals = m.dequant_f32_sample(tname, max_elements=128)
            if vals: last_norms.append(sum(vals)/len(vals))

    fn = sum(first_norms)/len(first_norms) if first_norms else 0
    ln = sum(last_norms)/len(last_norms) if last_norms else 0

    # Sparsity of layer 0 attn
    sparsity = {}
    for tname in m.tensors:
        if 'blk.0.' in tname and ('attn_q' in tname or 'attn_v' in tname) and 'norm' not in tname:
            vals = m.dequant_f32_sample(tname, max_elements=256)
            if vals:
                nz = sum(1 for v in vals if abs(v) < 0.001) / len(vals) * 100
                key = tname.split('blk.0.')[-1]
                sparsity[key] = f"{nz:.0f}%"

    return {
        'model_name': name, 'arch': arch,
        'params': f"{params/1e9:.2f}B",
        'n_layers': len(layers), 'max_layer': max_layer,
        'embed_dim': meta.get(f'{arch}.embedding_length', '?'),
        'heads': meta.get(f'{arch}.attention.head_count', '?'),
        'kv_heads': meta.get(f'{arch}.attention.head_count_kv', '?'),
        'ff_dim': meta.get(f'{arch}.feed_forward_length', '?'),
        'norm_growth': f"{ln/fn:.2f}x" if fn else "?",
        'first_norm': round(fn, 4), 'last_norm': round(ln, 4),
        'sparsity': sparsity,
    }


# ── Round 1: Self-Propose ────────────────────────────────────

@app.function(image=image, volumes={"/cache": model_cache}, timeout=600, memory=16384, cpu=4)
def round1_self_propose(model_name: str, profile: dict, roster: str, target_task: str) -> dict:
    """Each model inspects itself and proposes a LoRA for itself."""
    proc = start_ollama()
    pull_and_warm(model_name)

    s = profile
    system = f"""You are {s['model_name']} ({s['params']}, {s['arch']}, {s['n_layers']} layers).
{roster}

Everyone is being fine-tuned to: {target_task}

Your weight profile:
- Norm growth: {s['norm_growth']} (first layer: {s['first_norm']}, last: {s['last_norm']})
- Embedding dim: {s['embed_dim']}, Attention heads: {s['heads']}, KV heads: {s['kv_heads']}
- Layer 0 sparsity: {json.dumps(s['sparsity'])}

Propose a LoRA config for YOURSELF. Consider your specific architecture.
Respond with ONLY a JSON object:
{{
  "target_modules": ["q_proj", "v_proj"],
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "reasoning": "why these choices for MY architecture"
}}
No <think> tags. JSON only."""

    response = ask_model(model_name, system,
        f"Propose your LoRA config for: {target_task}. Consider your specific weight structure.")

    proposal = parse_json_from_text(response) or {
        "target_modules": ["q_proj", "v_proj"], "r": 16,
        "lora_alpha": 32, "lora_dropout": 0.05,
        "reasoning": f"default (parse failed): {response[:200]}"
    }

    proc.terminate()
    return {"model": model_name, "self_proposal": proposal, "raw": response[:500]}


# ── Round 2: Cross-Propose ───────────────────────────────────

@app.function(image=image, volumes={"/cache": model_cache}, timeout=600, memory=16384, cpu=4)
def round2_cross_propose(
    proposer_name: str,
    proposer_profile: dict,
    target_name: str,
    target_profile: dict,
    target_self_proposal: dict,
    all_round1: dict,
    roster: str,
    target_task: str,
) -> dict:
    """One model proposes a LoRA for ANOTHER model, informed by Round 1."""
    proc = start_ollama()
    pull_and_warm(proposer_name)

    # Summarize round 1
    r1_summary = ""
    for name, r1 in all_round1.items():
        prop = r1.get('self_proposal', {})
        r1_summary += f"\n  {name} proposed for itself: r={prop.get('r')}, modules={prop.get('target_modules')}\n"
        r1_summary += f"    Reasoning: {prop.get('reasoning', '?')[:200]}\n"

    tp = target_profile
    system = f"""You are {proposer_profile['model_name']} ({proposer_profile['params']}).

Round 1 results — what everyone proposed for themselves:
{r1_summary}

Now you must propose a LoRA for {tp['model_name']} ({tp['params']}, {tp['arch']}, {tp['n_layers']} layers).

{tp['model_name']}'s weight profile:
- Norm growth: {tp['norm_growth']}
- Embedding dim: {tp['embed_dim']}, Heads: {tp['heads']}, KV heads: {tp['kv_heads']}
- FFN dim: {tp['ff_dim']}
- Layer 0 sparsity: {json.dumps(tp['sparsity'])}

{tp['model_name']} proposed for itself: {json.dumps(target_self_proposal, indent=2)[:500]}

Do you agree with their self-proposal? Or would you change something?
Consider: their architecture differs from yours. What would YOU recommend?

Respond with ONLY a JSON object:
{{
  "for_model": "{target_name}",
  "agree_with_self": true/false,
  "target_modules": [...],
  "r": ...,
  "lora_alpha": ...,
  "lora_dropout": ...,
  "reasoning": "why I'd change/keep this for their architecture"
}}
No <think> tags. JSON only."""

    response = ask_model(proposer_name, system,
        f"Propose a LoRA for {tp['model_name']} to improve at: {target_task}")

    proposal = parse_json_from_text(response) or {
        "for_model": target_name,
        "agree_with_self": True,
        **target_self_proposal,
        "reasoning": f"parse failed, deferring to self: {response[:200]}",
    }

    proc.terminate()
    return {
        "proposer": proposer_name,
        "for_model": target_name,
        "cross_proposal": proposal,
        "raw": response[:500],
    }


# ── Round 3: Train ───────────────────────────────────────────

@app.function(image=image, volumes={"/cache": model_cache}, gpu="A10G", timeout=900, memory=32768)
def round3_train(
    model_name: str,
    self_proposal: dict,
    cross_proposals: list,
    target_task: str,
) -> dict:
    """Train using the consensus LoRA config (merge self + cross proposals)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    hf_name = HF_MAP.get(model_name)
    if not hf_name:
        return {"model": model_name, "error": f"No HF mapping for {model_name}"}

    # Build consensus config: majority vote on modules, average rank
    all_proposals = [self_proposal] + [cp.get('cross_proposal', {}) for cp in cross_proposals]

    # Count module votes
    module_votes = {}
    ranks = []
    alphas = []
    for p in all_proposals:
        for m in p.get('target_modules', []):
            module_votes[m] = module_votes.get(m, 0) + 1
        if p.get('r'):
            ranks.append(p['r'])
        if p.get('lora_alpha'):
            alphas.append(p['lora_alpha'])

    # Take modules with > 1 vote, or top-voted
    threshold = len(all_proposals) / 2
    consensus_modules = [m for m, v in module_votes.items() if v >= threshold]
    if not consensus_modules:
        consensus_modules = sorted(module_votes, key=module_votes.get, reverse=True)[:3]

    consensus_r = int(sum(ranks) / len(ranks)) if ranks else 16
    consensus_alpha = int(sum(alphas) / len(alphas)) if alphas else 32

    consensus = {
        "target_modules": consensus_modules,
        "r": consensus_r,
        "lora_alpha": consensus_alpha,
        "lora_dropout": 0.05,
        "n_votes": len(all_proposals),
        "module_votes": module_votes,
    }

    print(f"🧬 Training {model_name} with consensus LoRA")
    print(f"   Consensus from {len(all_proposals)} proposals:")
    print(f"   Modules: {consensus_modules} (votes: {module_votes})")
    print(f"   Rank: {consensus_r}, Alpha: {consensus_alpha}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir="/cache/hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, cache_dir="/cache/hf",
        torch_dtype=torch.bfloat16, device_map="auto", load_in_4bit=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Loaded: {total_params/1e9:.2f}B params")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=consensus_r, lora_alpha=consensus_alpha,
        lora_dropout=0.05, target_modules=consensus_modules, bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   LoRA: {trainable/1e6:.1f}M trainable ({trainable/total_params*100:.2f}%)")

    # Training data
    if 'reason' in target_task.lower() or 'math' in target_task.lower():
        examples = [
            {"text": f"Question: What is {a} * {b}?\nLet me think step by step.\n{a} * {b} = {a*b}\nThe answer is {a*b}."}
            for a in range(2, 15) for b in range(2, 8)
        ][:60]
    else:
        examples = [
            {"text": f"Instruction: {target_task}\nI'll approach this carefully. "
                     f"First, I need to understand what's being asked. Then I'll reason through it step by step."}
        ] * 60

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")

    dataset = Dataset.from_list(examples).map(tokenize, remove_columns=["text"])

    # Train
    args = TrainingArguments(
        output_dir=f"/cache/co_lora/{model_name.replace(':','_')}",
        num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        warmup_steps=10, logging_steps=10, save_strategy="no",
        fp16=True, report_to="none", remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    # Collect LoRA stats
    lora_stats = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_stats[name] = {
                'mean': round(param.data.float().mean().item(), 6),
                'std': round(param.data.float().std().item(), 6),
                'norm': round(param.data.float().norm().item(), 4),
            }

    # Quick eval
    model.eval()
    eval_prompts = ["What is 7 * 13?", "What is 25 + 38?", "What is 100 / 4?"]
    outputs = []
    for p in eval_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True)[:200])

    # Save adapter
    adapter_path = f"/cache/co_lora/{model_name.replace(':','_')}/adapter"
    model.save_pretrained(adapter_path)

    print(f"   ✅ Done in {elapsed:.0f}s, loss: {result.training_loss:.4f}")

    return {
        "model": model_name,
        "consensus": consensus,
        "trainable_params": f"{trainable/1e6:.1f}M",
        "trainable_pct": f"{trainable/total_params*100:.2f}%",
        "training_loss": round(result.training_loss, 4),
        "training_time": f"{elapsed:.0f}s",
        "lora_stats_sample": dict(list(lora_stats.items())[:6]),
        "eval_outputs": outputs,
    }


# ── Round 4: Cross-Evaluate ──────────────────────────────────

@app.function(image=image, volumes={"/cache": model_cache}, timeout=600, memory=16384, cpu=4)
def round4_evaluate(
    evaluator_name: str,
    evaluator_profile: dict,
    all_results: dict,
    roster: str,
    target_task: str,
) -> dict:
    """Each model evaluates ALL training results and picks winners."""
    proc = start_ollama()
    pull_and_warm(evaluator_name)

    results_summary = ""
    for name, r in all_results.items():
        results_summary += f"\n--- {name} ---\n"
        results_summary += f"  Consensus config: modules={r.get('consensus',{}).get('target_modules')}, "
        results_summary += f"r={r.get('consensus',{}).get('r')}, alpha={r.get('consensus',{}).get('lora_alpha')}\n"
        results_summary += f"  Module votes: {r.get('consensus',{}).get('module_votes')}\n"
        results_summary += f"  Trainable: {r.get('trainable_params')} ({r.get('trainable_pct')})\n"
        results_summary += f"  Loss: {r.get('training_loss')}\n"
        results_summary += f"  Time: {r.get('training_time')}\n"
        results_summary += f"  Sample LoRA norms: "
        for sname, stats in list(r.get('lora_stats_sample', {}).items())[:3]:
            short = '.'.join(sname.split('.')[-3:])
            results_summary += f"{short}={stats.get('norm',0):.4f} "
        results_summary += f"\n  Eval outputs: {json.dumps(r.get('eval_outputs',[])[:2])[:300]}\n"

    system = f"""You are {evaluator_profile['model_name']} ({evaluator_profile['params']}).

All models were fine-tuned to: {target_task}
{roster}

Training results:
{results_summary}

Evaluate ALL results. For each model:
1. Was the consensus LoRA config good? Why/why not?
2. Did the loss converge well?
3. Are the LoRA weight norms healthy?
4. How do the eval outputs look?

Then: RANK all models from best to worst adaptation.
Finally: What would you change for the NEXT iteration?

Be specific. Reference numbers. No <think> tags."""

    evaluation = ask_model(evaluator_name, system,
        "Evaluate all training results. Rank them. What would you change?")

    proc.terminate()
    return {"evaluator": evaluator_name, "evaluation": evaluation}


# ── Main ─────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    models: str = None,
    target: str = "improve step-by-step mathematical reasoning",
):
    MODELS = models.split(',') if models else DEFAULT_MODELS

    print("━" * 65)
    print("  🪞🧬🤝 Neural Mirror — Co-LoRA")
    print("  Models collaboratively design adapters for each other")
    print("━" * 65)
    print(f"\n  Models: {', '.join(MODELS)}")
    print(f"  Target: {target}\n")

    t_start = time.time()

    # ── Profile all models ──
    print("📋 Profiling...")

    # Profile in parallel using round1 containers (they pull + profile)
    profile_results = {}
    profile_futs = {}
    for m in MODELS:
        # Use a lightweight function just to profile
        profile_futs[m] = round1_self_propose.spawn(m, {
            'model_name': m, 'params': '?', 'arch': '?', 'n_layers': 0,
            'max_layer': 0, 'embed_dim': '?', 'heads': '?', 'kv_heads': '?',
            'ff_dim': '?', 'norm_growth': '?', 'first_norm': 0, 'last_norm': 0,
            'sparsity': {},
        }, "Models: " + ", ".join(MODELS), target)

    # Actually, let's profile properly with a dedicated function
    # For now just proceed with round 1 which does profile internally

    # ── Round 1: Self-propose (parallel) ──
    print("\n" + "━" * 65)
    print("  Round 1: Each model proposes LoRA for ITSELF")
    print("━" * 65 + "\n")

    # We need profiles first — let's use a bootstrap approach
    # Each round1 container profiles + proposes
    @app.function(image=image, volumes={"/cache": model_cache}, timeout=600, memory=16384, cpu=4)
    def bootstrap_round1(model_name, all_models, target_task):
        proc = start_ollama()
        pull_and_warm(model_name)
        gguf_path = find_gguf(model_name)
        if not gguf_path:
            proc.terminate()
            return {"model": model_name, "error": "no GGUF"}

        profile = profile_gguf(gguf_path)
        roster = "Models: " + ", ".join(all_models)

        s = profile
        system = f"""You are {s['model_name']} ({s['params']}, {s['arch']}, {s['n_layers']} layers).
{roster}
Task: {target_task}

Your weights: norm growth {s['norm_growth']}, embed {s['embed_dim']}, heads {s['heads']}, KV {s['kv_heads']}, FFN {s['ff_dim']}
Sparsity: {json.dumps(s['sparsity'])}

Propose a LoRA config for YOURSELF as JSON:
{{"target_modules": [...], "r": N, "lora_alpha": N, "lora_dropout": 0.05, "reasoning": "..."}}
No <think> tags. JSON only."""

        response = ask_model(model_name, system,
            f"Propose your LoRA for: {target_task}")

        proposal = parse_json_from_text(response) or {
            "target_modules": ["q_proj", "v_proj"], "r": 16,
            "lora_alpha": 32, "lora_dropout": 0.05,
            "reasoning": f"default: {response[:200]}",
        }

        proc.terminate()
        return {"model": model_name, "profile": profile, "self_proposal": proposal, "raw": response[:500]}

    r1_futs = {m: bootstrap_round1.spawn(m, MODELS, target) for m in MODELS}
    r1_results = {}
    profiles = {}
    for m, fut in r1_futs.items():
        try:
            r = fut.get()
            r1_results[m] = r
            profiles[m] = r.get('profile', {})
            prop = r.get('self_proposal', {})
            p = profiles[m]
            print(f"  🪞 {p.get('model_name','?')} ({p.get('params','?')}, {p.get('arch','?')})")
            print(f"     Proposed: r={prop.get('r')}, modules={prop.get('target_modules')}")
            print(f"     Reasoning: {prop.get('reasoning','?')[:150]}")
            print()
        except Exception as e:
            print(f"  ❌ {m}: {e}\n")

    roster = "Models:\n" + "\n".join(
        f"  - {profiles.get(m,{}).get('model_name',m)} ({profiles.get(m,{}).get('params','?')}, {profiles.get(m,{}).get('arch','?')})"
        for m in MODELS if m in profiles
    )

    # ── Round 2: Cross-propose (each proposes for the others) ──
    print("━" * 65)
    print("  Round 2: Each model proposes LoRA for the OTHERS")
    print("━" * 65 + "\n")

    r2_futs = []
    for proposer in MODELS:
        if proposer not in profiles:
            continue
        for target_model in MODELS:
            if target_model == proposer or target_model not in profiles:
                continue
            r2_futs.append((proposer, target_model, round2_cross_propose.spawn(
                proposer, profiles[proposer],
                target_model, profiles[target_model],
                r1_results.get(target_model, {}).get('self_proposal', {}),
                {m: r1_results.get(m, {}) for m in MODELS},
                roster, target,
            )))

    cross_proposals = {m: [] for m in MODELS}
    for proposer, target_model, fut in r2_futs:
        try:
            r = fut.get()
            cp = r.get('cross_proposal', {})
            cross_proposals[target_model].append(r)
            p_name = profiles.get(proposer, {}).get('model_name', proposer)
            t_name = profiles.get(target_model, {}).get('model_name', target_model)
            agree = cp.get('agree_with_self', '?')
            print(f"  {p_name} → {t_name}: agree={agree}, r={cp.get('r')}, modules={cp.get('target_modules')}")
            if cp.get('reasoning'):
                print(f"    {cp['reasoning'][:150]}")
            print()
        except Exception as e:
            print(f"  ❌ {proposer}→{target_model}: {e}\n")

    # ── Round 3: Train all (parallel on GPUs) ──
    print("━" * 65)
    print("  Round 3: Training consensus LoRAs (parallel GPUs)")
    print("━" * 65 + "\n")

    train_futs = {}
    for m in MODELS:
        if m not in r1_results:
            continue
        train_futs[m] = round3_train.spawn(
            m,
            r1_results[m].get('self_proposal', {}),
            cross_proposals.get(m, []),
            target,
        )

    train_results = {}
    for m, fut in train_futs.items():
        try:
            r = fut.get()
            train_results[m] = r
            p = profiles.get(m, {})
            print(f"  ✅ {p.get('model_name','?')}")
            print(f"     Consensus: modules={r.get('consensus',{}).get('target_modules')}, r={r.get('consensus',{}).get('r')}")
            print(f"     Votes: {r.get('consensus',{}).get('module_votes')}")
            print(f"     Loss: {r.get('training_loss')}, Time: {r.get('training_time')}")
            print(f"     Trainable: {r.get('trainable_params')} ({r.get('trainable_pct')})")
            print()
        except Exception as e:
            print(f"  ❌ {m}: {e}\n")

    # ── Round 4: Cross-evaluate ──
    print("━" * 65)
    print("  Round 4: Models evaluate each other's results")
    print("━" * 65 + "\n")

    eval_futs = {}
    for m in MODELS:
        if m not in profiles:
            continue
        eval_futs[m] = round4_evaluate.spawn(m, profiles[m], train_results, roster, target)

    for m, fut in eval_futs.items():
        try:
            r = fut.get()
            p = profiles.get(m, {})
            print(f"  🪞 {p.get('model_name','?')}'s evaluation:")
            for line in r.get('evaluation', '').split('\n'):
                if line.strip():
                    print(f"     {line.strip()[:120]}")
            print()
        except Exception as e:
            print(f"  ❌ {m}: {e}\n")

    elapsed = time.time() - t_start

    # ── Summary ──
    print("━" * 65)
    print(f"  🪞🧬🤝 Co-LoRA Complete — {elapsed:.0f}s")
    print("━" * 65)
    print(f"\n  Models: {len(profiles)}")
    print(f"  Rounds: 4 (self-propose → cross-propose → train → evaluate)")
    print(f"  Target: {target}")
    print()

    if train_results:
        print(f"  {'Model':<25s} {'Consensus Modules':<30s} {'r':>3s} {'Loss':>8s}")
        print(f"  {'─'*25} {'─'*30} {'─'*3} {'─'*8}")
        for m, r in train_results.items():
            name = profiles.get(m, {}).get('model_name', m)
            mods = ','.join(r.get('consensus',{}).get('target_modules',[]))
            rank = r.get('consensus',{}).get('r','?')
            loss = r.get('training_loss','?')
            print(f"  {name:<25s} {mods:<30s} {str(rank):>3s} {str(loss):>8s}")

    print()

    # Save
    transcript = {
        'models': MODELS,
        'target': target,
        'profiles': profiles,
        'round1_self_proposals': {m: r1_results.get(m, {}).get('self_proposal') for m in MODELS},
        'round2_cross_proposals': {m: [cp.get('cross_proposal') for cp in cps] for m, cps in cross_proposals.items()},
        'round3_training': train_results,
        'elapsed': round(elapsed, 1),
    }
    out = os.path.expanduser('~/neural-mirror/co_lora_transcript.json')
    try:
        with open(out, 'w') as f:
            json.dump(transcript, f, indent=2, default=str)
        print(f"  Transcript: {out}")
    except:
        pass
