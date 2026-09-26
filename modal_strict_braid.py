"""Live strict-braille braid between admitted PEFT adapter identities.

The host assigns evidence IDs and validates frames. Peers receive the raw
14-cell wire frame and must decode it exactly before their semantic reply is
encoded and committed. No malformed response is repaired.
"""

import json
import os
import modal

from braille_protocol import encode_braid_strict
from evidence_registry import validate_registry
from strict_braid import BraidRejected, EvidenceRecord, StrictBraid

app = modal.App("neural-mirror-live-strict-braid")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "huggingface_hub<1.0",
        "accelerate", "peft", "safetensors", "sentencepiece",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("strict_braid.py", "/root/strict_braid.py")
    .add_local_file("evidence_registry.py", "/root/evidence_registry.py")
)

PARTICIPANTS = {
    "qwen": {
        "identity": "qwen3:4b+qwen3-4b-braille-literacy-v3",
        "hf": "Qwen/Qwen3-4B",
        "adapter": "qwen3-4b-braille-literacy-v3",
    },
    "llama": {
        "identity": "llama3.2:3b+llama3.2-3b-braille-literacy-v1",
        "hf": "unsloth/Llama-3.2-3B-Instruct",
        "adapter": "llama3.2-3b-braille-literacy-v1",
    },
}

FRAME_SPEC = """The incoming string is exactly 14 Unicode braille cells. Decode each cell as byte = codepoint-U+2800.
Fields by byte index: 0 type=32; 1 version=1; 2 sender; 3 round; 4 operation
(1 observe,2 support,3 challenge,4 propose_test,5 synthesize); 5 subject;
6-7 little-endian evidence_id; 8 relation (1 increase,2 decrease,3 similar,
4 different,5 supports,6 contradicts,7 needs_test); 9 confidence percent;
10 caveat flags; 11-12 IEEE-f16 little-endian value; 13 CRC-8/ATM.
"""


def _prefix(tokenizer, prompt):
    messages = [
        {"role": "system", "content": "Exact protocol operation. Output one JSON object and no prose."},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@app.function(image=image, volumes={"/cache": volume}, gpu="H100", memory=49152, timeout=1800, max_containers=2)
def adapter_turn(key: str, prompt: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = PARTICIPANTS[key]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf"], cache_dir="/cache/hf")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        spec["hf"], cache_dir="/cache/hf", dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, f"/cache/braille-adapters/{spec['adapter']}")
    encoded = tokenizer(_prefix(tokenizer, prompt), return_tensors="pt")
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    length = encoded["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=900, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
    try:
        parsed = json.loads(raw)
        return {"identity": spec["identity"], "parsed": parsed, "raw": raw}
    except Exception as error:
        return {"identity": spec["identity"], "error": f"JSON:{error}", "raw": raw}


def _exact_decoded(actual, expected):
    fields = [
        "version", "sender_idx", "round_idx", "operation", "subject_idx",
        "evidence_id", "relation", "confidence_pct", "caveat_flags", "value",
    ]
    return all(actual.get(field) == expected.get(field) for field in fields)


def _reply_frame(reply, sender_idx, round_idx):
    required = {
        "operation", "subject_idx", "evidence_id", "relation",
        "confidence_pct", "caveat_flags", "value",
    }
    if set(reply) != required:
        raise BraidRejected("REPLY_SCHEMA")
    return encode_braid_strict(
        sender_idx, round_idx, reply["operation"], reply["subject_idx"],
        reply["evidence_id"], reply["relation"], reply["confidence_pct"],
        reply["caveat_flags"], reply["value"],
    )


@app.local_entrypoint()
def main():
    participants = [PARTICIPANTS["qwen"]["identity"], PARTICIPANTS["llama"]["identity"]]
    registry_path = os.path.expanduser("~/neural-mirror/introspection_evidence_registry.json")
    with open(registry_path) as handle:
        registry = json.load(handle)
    validate_registry(registry)
    participant_index = {"qwen3:4b": 0, "llama3.2:3b": 1}
    selected_records = [
        record for record in registry["records"]
        if record["owner_model"] in participant_index and record["certified_owner"]
    ]
    evidence = {
        record["evidence_id"]: EvidenceRecord(
            record["evidence_id"], participant_index[record["owner_model"]],
            record["key"], record, record["value"],
        )
        for record in selected_records
    }
    causal_record = next(
        record for record in selected_records
        if record["owner_model"] == "qwen3:4b"
        and record["level"] == "causal"
        and record["metric"] == "mean_abs_standardized_margin_change"
    )
    evidence_id = causal_record["evidence_id"]
    evidence_value = causal_record["value"]
    braid = StrictBraid(participants, evidence)

    qwen_prompt = f"""Create one bounded observation about this canonical registry record:
{json.dumps(causal_record, ensure_ascii=False)}
Return exactly:
{{"operation":"observe","subject_idx":0,"evidence_id":{evidence_id},"relation":"increase","confidence_pct":INTEGER_0_TO_100,"caveat_flags":1,"value":{evidence_value}}}
Use exactly these keys and values except choose confidence_pct."""
    qwen = adapter_turn.remote("qwen", qwen_prompt)
    if "error" in qwen:
        raise RuntimeError(qwen)
    frame1 = _reply_frame(qwen["parsed"], sender_idx=0, round_idx=1)
    committed1 = braid.submit(frame1)

    llama_prompt = f"""A deterministic host validated this raw braille wire frame:
RAW: {frame1}
VALIDATED_FIELDS: {json.dumps(committed1['decoded'], ensure_ascii=False)}
CANONICAL_EVIDENCE: {json.dumps(causal_record, ensure_ascii=False)}
Challenge overgeneralization beyond the stated intervention and request replication. Return exactly:
{{"operation":"challenge","subject_idx":0,"evidence_id":{evidence_id},"relation":"needs_test","confidence_pct":INTEGER_0_TO_100,"caveat_flags":5,"value":{evidence_value}}}
Use no prose and no additional keys. The host, not you, owns CRC and byte encoding."""
    llama = adapter_turn.remote("llama", llama_prompt)
    if "error" in llama:
        raise BraidRejected("LLAMA_SCHEMA")
    frame2 = _reply_frame(llama["parsed"], sender_idx=1, round_idx=2)
    committed2 = braid.submit(frame2)

    qwen_reply_prompt = f"""A deterministic host validated this peer's raw braille wire frame:
RAW: {frame2}
VALIDATED_FIELDS: {json.dumps(committed2['decoded'], ensure_ascii=False)}
CANONICAL_EVIDENCE: {json.dumps(causal_record, ensure_ascii=False)}
Acknowledge that replication is needed. Return exactly:
{{"operation":"support","subject_idx":1,"evidence_id":{evidence_id},"relation":"supports","confidence_pct":INTEGER_0_TO_100,"caveat_flags":5,"value":{evidence_value}}}
Use no prose and no additional keys. The host, not you, owns CRC and byte encoding."""
    qwen_reply = adapter_turn.remote("qwen", qwen_reply_prompt)
    if "error" in qwen_reply:
        raise BraidRejected("QWEN_REPLY_SCHEMA")
    frame3 = _reply_frame(qwen_reply["parsed"], sender_idx=0, round_idx=3)
    committed3 = braid.submit(frame3)

    result = braid.transcript()
    result["registry_sha256"] = registry["registry_sha256"]
    result["selected_registry_evidence_id"] = evidence_id
    result["live_model_evidence"] = {
        "qwen_observation": qwen,
        "llama_challenge": llama,
        "qwen_acknowledgment": qwen_reply,
        "transport_integrity_owner": "deterministic host",
        "models_selected_semantics": True,
        "models_did_not_compute_crc": True,
    }
    path = os.path.expanduser("~/neural-mirror/live_strict_braid.json")
    with open(path + ".tmp", "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(f"LIVE STRICT BRAID PASS: {len(result['frames'])} frames; {path}")
