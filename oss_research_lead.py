"""
Neural Mirror — GPT-OSS 120B Research Lead

GPT-OSS 120B joins the swarm as an advisor, not a LoRA-training target.
It audits evidence produced by the smaller models, separates measurements
from interpretations, and specifies the next controlled experiment.

Usage:
  modal run oss_research_lead.py
  modal run oss_research_lead.py --model gpt-oss:120b
"""

import json
import os
import subprocess
import time
import urllib.request

import modal

app = modal.App("neural-mirror-oss-research-lead")
model_cache = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "zstd", "procps")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
)

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "verdict": {"type": "string", "enum": ["promising", "inconclusive", "invalid"]},
        "verified_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string"},
                    "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    "evidence": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["finding", "strength", "evidence", "evidence_ids"],
            },
        },
        "unsupported_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "problem": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["claim", "problem", "replacement"],
            },
        },
        "critical_confounds": {"type": "array", "items": {"type": "string"}},
        "next_experiment": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "protocol": {"type": "array", "items": {"type": "string"}},
                "controls": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "success_criterion": {"type": "string"},
            },
            "required": ["question", "protocol", "controls", "metrics", "success_criterion"],
        },
        "swarm_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "model": {"type": "string"},
                    "job": {"type": "string"},
                },
                "required": ["role", "model", "job"],
            },
        },
    },
    "required": [
        "headline", "verdict", "verified_findings", "unsupported_claims",
        "critical_confounds", "next_experiment", "swarm_roles",
    ],
}


def _validate_review(review, evidence):
    """Reject evidence hallucinations before accepting the advisor's prose."""
    import re

    rows = evidence.get("ablation", {}).get("interventions", [])
    known_ids = {row["id"] for row in rows}
    known_tensors = {row["tensor"] for row in rows}
    errors = []

    for index, item in enumerate(review.get("verified_findings", []), 1):
        ids = set(item.get("evidence_ids", []))
        unknown_ids = sorted(ids - known_ids)
        if unknown_ids:
            errors.append(f"verified finding {index} cites unknown IDs: {unknown_ids}")
        if not ids and "Co-LoRA" not in item.get("finding", ""):
            errors.append(f"verified finding {index} has no ablation evidence IDs")

        text = f"{item.get('finding', '')} {item.get('evidence', '')}"
        mentioned = set(re.findall(r"model\.layers\.\d+\.[A-Za-z0-9_.]+?\.weight", text))
        unknown_tensors = sorted(mentioned - known_tensors)
        if unknown_tensors:
            errors.append(f"verified finding {index} names untested tensors: {unknown_tensors}")
        lowered = text.lower()
        if "partial zero" in lowered or "90% of elements" in lowered:
            errors.append(
                f"verified finding {index} misstates the intervention; every A-row is full_tensor_zero"
            )

    return errors


def _start_ollama():
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = "/cache/ollama"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
    env["OLLAMA_KEEP_ALIVE"] = "30m"
    proc = subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return proc, env
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Ollama failed to start")


@app.function(
    image=image,
    volumes={"/cache": model_cache},
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=3600,
)
def audit_swarm(model_name: str, evidence: dict) -> dict:
    """Have GPT-OSS audit the swarm's existing experimental evidence."""
    proc, env = _start_ollama()
    try:
        pull = subprocess.run(
            ["ollama", "pull", model_name], env=env,
            capture_output=True, text=True, timeout=3600,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"pull failed: {pull.stderr[-1000:]}")

        system = """You are the senior research lead for Neural Mirror, a swarm of
small language models studying their own weights. Your job is scientific audit,
not encouragement. Treat raw measurements as evidence and model explanations as
unverified hypotheses. Never infer tensor function merely from its name, size,
layer, norm, or a single intervention. Do not compare language-model training
losses across architectures/tokenizers as if they were a common benchmark.

Return valid JSON matching the requested schema. Put the most important finding
first. Every ablation finding must cite the exact supplied A-IDs in evidence_ids.
Do not invent, rename, or interpolate tensors. All A-row interventions are full-
tensor zeroing, regardless of changed_pct. Be concise but technically specific.
Design one controlled experiment that can falsify the leading hypothesis."""

        prompt = f"""Audit this Neural Mirror evidence package.

KNOWN METHODOLOGY LIMITATIONS (verify their consequences):
- Ablation v2 uses each instruct model's native chat template, greedy decoding, and answer-key correctness.
- It still has only 11 prompts; answer matching is rule-based rather than human or blinded-judge scoring.
- 'changed' means exact-string inequality; it is a sensitivity measure, not a quality metric.
- Each intervention zeroed one entire projection/norm tensor, an extreme out-of-distribution perturbation.
- There were no random-row, matched-magnitude noise, repeated-model, or bootstrap controls.
- Compare matched module types across layers, but do not generalize beyond Qwen3-1.7B from one run.
- Co-LoRA reported each model's own training loss; tokenizers and base models differ.
- Co-LoRA evaluation had only three training-like arithmetic prompts and no pre-training baseline.
- GGUF Q4_K/Q6_K dequantization in the prototype is simplified, so numerical weight claims need validation.

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}

Decide what, if anything, is established. The intervention rows are an evidence
ledger: cite their IDs exactly, and never describe changed_pct as the fraction of
a tensor zeroed. Correct the swarm's strongest unsupported claims, then specify
the highest-information next experiment."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        raw = {}
        review = {}
        validation_errors = []
        for attempt in range(2):
            payload = {
                "model": model_name,
                "messages": messages,
                "format": REVIEW_SCHEMA,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 32768},
            }
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1200) as resp:
                raw = json.loads(resp.read())
            content = raw.get("message", {}).get("content", "")
            try:
                review = json.loads(content)
            except json.JSONDecodeError:
                review = {"parse_error": True, "raw": content}

            validation_errors = _validate_review(review, evidence)
            if not validation_errors:
                break
            messages.extend([
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your review failed machine validation:\n- "
                        + "\n- ".join(validation_errors)
                        + "\nRegenerate the complete JSON. Cite only supplied A-IDs and tested tensors. "
                          "All recorded interventions were full-tensor zeroing."
                    ),
                },
            ])

        return {
            "advisor": model_name,
            "role": "research_lead",
            "review": review,
            "validation": {
                "passed": not validation_errors,
                "errors": validation_errors,
            },
            "timing": {
                "total_duration_ns": raw.get("total_duration"),
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "eval_count": raw.get("eval_count"),
            },
        }
    finally:
        proc.terminate()


def _compact_ablation(data):
    study = data.get("ablation_results", data)
    rows = []
    for index, item in enumerate(study.get("ranked_by_impact", []), 1):
        examples = []
        for category, prompts in item.get("details", {}).items():
            for prompt, detail in prompts.items():
                if detail.get("status") != "UNCHANGED" and len(examples) < 2:
                    examples.append({
                        "category": category,
                        "prompt": prompt,
                        "status": detail.get("status"),
                        "baseline": detail.get("baseline"),
                        "ablated": detail.get("ablated"),
                    })
        rows.append({
            "id": f"A{index:02d}",
            "intervention": "full_tensor_zero",
            "tensor": item.get("tensor"),
            "changed_pct": item.get("change_pct"),
            "baseline_accuracy": item.get("baseline_accuracy"),
            "ablated_accuracy": item.get("ablated_accuracy"),
            "accuracy_drop_pp": item.get("accuracy_drop_pp"),
            "mean_text_similarity": item.get("mean_text_similarity"),
            "params": item.get("params"),
            "pct_of_model": item.get("pct_of_model"),
            "examples": examples,
        })
    return {
        "model": study.get("model"),
        "total_params": study.get("total_params"),
        "methodology": study.get("methodology", {"version": 2, "quality_metric": "answer-key accuracy"}),
        "prompt_count": study.get("baseline_prompts"),
        "baseline_outputs": study.get("baseline"),
        "interventions": rows,
        "small_model_self_analysis": data.get("self_analysis", {}).get("analysis"),
    }


def _compact_colora(data):
    training = {}
    for model, result in data.get("round3_training", {}).items():
        training[model] = {
            "consensus": result.get("consensus"),
            "training_loss": result.get("training_loss"),
            "trainable_params": result.get("trainable_params"),
            "eval_outputs": result.get("eval_outputs"),
        }
    return {
        "task": data.get("target"),
        "self_proposals": data.get("round1_self_proposals"),
        "cross_proposals": data.get("round2_cross_proposals"),
        "training": training,
    }


def _compact_roundtable(data):
    # Keep profiles and only a short prefix of each response. This is context,
    # not primary quantitative evidence.
    rounds = {}
    for round_num, findings in data.get("rounds", {}).items():
        rounds[round_num] = {
            model: text[:1200] for model, text in findings.items()
        }
    return {"participants": data.get("participants"), "rounds": rounds}


@app.local_entrypoint()
def main(model: str = "gpt-oss:120b"):
    root = os.path.dirname(os.path.abspath(__file__))

    def load(name):
        path = os.path.join(root, name)
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    evidence = {
        "ablation": _compact_ablation(load("ablation_results.json")),
        "co_lora": _compact_colora(load("co_lora_transcript.json")),
        "roundtable": _compact_roundtable(load("roundtable_transcript.json")),
    }

    print("━" * 68)
    print("  🧠 GPT-OSS 120B — Neural Mirror Research Lead")
    print("  Auditing ablation, Co-LoRA, and roundtable evidence on H100")
    print("━" * 68)

    result = audit_swarm.remote(model, evidence)
    review = result.get("review", {})

    validation = result.get("validation", {})
    print(f"\n  Evidence validation: {'PASS' if validation.get('passed') else 'FAIL'}")
    for error in validation.get("errors", []):
        print(f"    - {error}")
    print(f"  Verdict: {review.get('verdict', '?').upper()}")
    print(f"  Headline: {review.get('headline', '?')}\n")

    print("  Verified findings:")
    for item in review.get("verified_findings", []):
        print(f"  - [{item.get('strength', '?')}] {item.get('finding', '?')}")
        print(f"    {item.get('evidence', '')}")

    print("\n  Claims corrected:")
    for item in review.get("unsupported_claims", []):
        print(f"  - {item.get('claim', '?')}")
        print(f"    Problem: {item.get('problem', '')}")
        print(f"    Replace with: {item.get('replacement', '')}")

    experiment = review.get("next_experiment", {})
    print(f"\n  Next experiment: {experiment.get('question', '?')}")
    for index, step in enumerate(experiment.get("protocol", []), 1):
        print(f"    {index}. {step}")
    print(f"  Success: {experiment.get('success_criterion', '?')}")

    out = os.path.join(root, "oss120b_review.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out}")
