"""Extend admitted literacy adapters with strict 14-cell braid-frame competence."""

import json
import os
import random
import time
import traceback

import modal

from braille_literacy import SEEDS as LITERACY_SEEDS, _build_tasks, _literacy_prompt, _score
from braille_protocol import (
    BRAID_OPERATIONS, BRAID_RELATIONS, decode_braid_strict, encode_braid_strict,
)

app = modal.App("neural-mirror-braid-curriculum")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "huggingface_hub<1.0", "accelerate",
        "peft", "safetensors", "sentencepiece",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("braille_literacy.py", "/root/braille_literacy.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
)

MODELS = {
    "qwen": {
        "hf": "Qwen/Qwen3-4B",
        "parent": "qwen3-4b-braille-literacy-v3",
        "output": "qwen3-4b-braille-braid-v1",
    },
    "llama": {
        "hf": "unsloth/Llama-3.2-3B-Instruct",
        "parent": "llama3.2-3b-braille-literacy-v1",
        "output": "llama3.2-3b-braille-braid-v1",
    },
}
TRAIN_SEEDS = list(range(3000, 3256))
BRAID_EXAM_SEEDS = [37, 73]
FRAME_SPEC = """STRICT BRAID V1: each frame is exactly 14 Unicode braille cells; byte=codepoint-U+2800.
Bytes: [type=32,version=1,sender,round,operation,subject,evidence_lo,evidence_hi,relation,confidence,caveat,value_f16_lo,value_f16_hi,crc8].
Operations: 1 observe,2 support,3 challenge,4 propose_test,5 synthesize.
Relations: 1 increase,2 decrease,3 similar,4 different,5 supports,6 contradicts,7 needs_test.
CRC is CRC-8/ATM polynomial 0x07 over bytes 0..12. Output only exact JSON.
"""


def _prefix(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "Exact protocol operation. Output JSON only."},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _semantic(rng):
    return {
        "sender_idx": rng.randrange(0, 12),
        "round_idx": rng.randrange(1, 8),
        "operation": rng.choice(list(BRAID_OPERATIONS)),
        "subject_idx": rng.randrange(0, 12),
        "evidence_id": rng.randrange(1, 4096),
        "relation": rng.choice(list(BRAID_RELATIONS)),
        "confidence_pct": rng.randrange(0, 101),
        "caveat_flags": rng.randrange(0, 16),
        "value": rng.choice([-2.0, -0.5, -0.125, 0.0, 0.001, 0.25, 1.0, 3.5]),
    }


def _tasks(seed):
    rng = random.Random(seed)
    decodes, encodes = [], []
    for index in range(4):
        semantic = _semantic(rng)
        frame = encode_braid_strict(**semantic)
        expected = decode_braid_strict(frame)
        expected.pop("checksum")
        decodes.append({"id": f"D{index+1}", "frame": frame, "expected": expected})
    for index in range(4):
        semantic = _semantic(rng)
        frame = encode_braid_strict(**semantic)
        encodes.append({"id": f"E{index+1}", **semantic, "expected": frame})
    return {"decodes": decodes, "encodes": encodes}


def _prompt(tasks):
    public = {
        "decodes": [{"id": x["id"], "frame": x["frame"]} for x in tasks["decodes"]],
        "encodes": [{k: v for k, v in x.items() if k != "expected"} for x in tasks["encodes"]],
    }
    return FRAME_SPEC + "\nTasks:\n" + json.dumps(public, ensure_ascii=False, separators=(",", ":")) + (
        '\nReturn {"decodes":[{"id":"D1",version,sender_idx,round_idx,operation,subject_idx,evidence_id,relation,confidence_pct,caveat_flags,value},...],'
        '"encodes":[{"id":"E1","frame":"..."},...]}. Preserve order and use no extra keys.'
    )


def _answer(tasks):
    return {
        "decodes": [{"id": x["id"], **x["expected"]} for x in tasks["decodes"]],
        "encodes": [{"id": x["id"], "frame": x["expected"]} for x in tasks["encodes"]],
    }


def _score_braid(tasks, answer):
    expected = _answer(tasks)
    checks = []
    for group in ("decodes", "encodes"):
        actual = answer.get(group) if isinstance(answer, dict) else None
        if not isinstance(actual, list):
            actual = []
        by_id = {x.get("id"): x for x in actual if isinstance(x, dict)}
        for item in expected[group]:
            checks.append({"id": item["id"], "passed": by_id.get(item["id"]) == item})
    return {"passed": all(x["passed"] for x in checks), "correct": sum(x["passed"] for x in checks), "total": 8, "checks": checks}


def _generate(model, tokenizer, prompt, max_new_tokens=1600):
    import torch
    inputs = tokenizer(_prefix(tokenizer, prompt), return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
    return raw, json.loads(raw)


@app.function(image=image, volumes={"/cache": volume}, gpu="H100", memory=65536, timeout=5400, max_containers=2)
def train_braid(key: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[key]
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["hf"], cache_dir="/cache/hf")
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(spec["hf"], cache_dir="/cache/hf", dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
        base.config.use_cache = False
        base.gradient_checkpointing_enable(); base.enable_input_require_grads()
        model = PeftModel.from_pretrained(base, f"/cache/braille-adapters/{spec['parent']}", is_trainable=True)
        examples=[]
        for seed in TRAIN_SEEDS:
            tasks=_tasks(seed); prompt_ids=tokenizer(_prefix(tokenizer,_prompt(tasks)))["input_ids"]
            answer_ids=tokenizer(json.dumps(_answer(tasks),ensure_ascii=False,separators=(",",":"))+(tokenizer.eos_token or ""),add_special_tokens=False)["input_ids"]
            examples.append((prompt_ids+answer_ids,[-100]*len(prompt_ids)+answer_ids))
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=2.5e-5,weight_decay=.01); opt.zero_grad(set_to_none=True)
        rng=random.Random(8181); order=list(range(len(examples))); rng.shuffle(order); losses=[]
        for pos,idx in enumerate(order):
            ids,labels=examples[idx]; x=torch.tensor([ids],device="cuda")
            out=model(input_ids=x,attention_mask=torch.ones_like(x),labels=torch.tensor([labels],device="cuda"),use_cache=False)
            (out.loss/4).backward(); losses.append(float(out.loss.detach()))
            if (pos+1)%4==0 or pos==len(order)-1:
                torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True)
        path=f"/cache/braille-adapters/{spec['output']}"; model.save_pretrained(path); tokenizer.save_pretrained(path); volume.commit()
        model.config.use_cache=True; braid_exam=[]
        for seed in BRAID_EXAM_SEEDS:
            tasks=_tasks(seed); raw,answer=_generate(model,tokenizer,_prompt(tasks)); score=_score_braid(tasks,answer); score.update({"seed":seed,"raw":raw}); braid_exam.append(score)
        literacy=[]
        for seed in LITERACY_SEEDS:
            tasks=_build_tasks(seed); raw,answer=_generate(model,tokenizer,_literacy_prompt(tasks),2200); score=_score(tasks,answer); score.update({"seed":seed}); literacy.append(score)
        passed=all(x["passed"] for x in braid_exam+literacy)
        result={"model":key,"adapter":spec["output"],"parent":spec["parent"],"passed":passed,
                "braid_correct":sum(x["correct"] for x in braid_exam),"braid_total":16,
                "literacy_correct":sum(x["correct"] for x in literacy),"literacy_total":28,
                "first_loss":losses[0],"last_loss":losses[-1],"braid_exam":braid_exam,"literacy_exam":literacy}
    except Exception as e:
        result={"model":key,"passed":False,"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()[-6000:]}
    out=f"/cache/braid-results/{key}.json"; os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w") as f: json.dump(result,f,indent=2,ensure_ascii=False)
    volume.commit(); return result


@app.local_entrypoint()
def main(models: str="qwen,llama"):
    for key in [x.strip() for x in models.split(",") if x.strip()]:
        call=train_braid.spawn(key); print(key,call.object_id)
