"""Tokenizer-cost comparison for the strict braid transport encodings."""

import base64
import json
import os

import modal

from braille_protocol import encode_braid_strict, uncells

app = modal.App("neural-mirror-transport-benchmark")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==4.57.6", "huggingface_hub<1.0", "sentencepiece", "tiktoken")
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
)

TOKENIZERS = {
    "qwen3:4b": "Qwen/Qwen3-4B",
    "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct",
    "granite3.3:8b": "ibm-granite/granite-3.3-8b-instruct",
    "glm4:9b": "THUDM/glm-4-9b-chat-hf",
    "falcon3:7b": "tiiuae/Falcon3-7B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi4-mini": "microsoft/phi-4-mini-instruct",
    "solar:10.7b": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "gemma3:4b": "unsloth/gemma-3-4b-it",
    "olmo2:7b": "allenai/OLMo-2-1124-7B-Instruct",
    "yi:6b": "01-ai/Yi-6B-Chat",
}

SEMANTIC = {
    "sender_idx": 0, "round_idx": 2, "operation": "challenge", "subject_idx": 1,
    "evidence_id": 513, "relation": "needs_test", "confidence_pct": 87,
    "caveat_flags": 5, "value": -0.25,
}
FRAME = encode_braid_strict(**SEMANTIC)
RAW = uncells(FRAME)
REPRESENTATIONS = {
    "braille": FRAME,
    "compact_json": json.dumps(SEMANTIC, separators=(",", ":")),
    "readable_json": json.dumps(SEMANTIC, indent=2),
    "base64": base64.b64encode(RAW).decode(),
    "hex": RAW.hex(),
}


@app.function(image=image, volumes={"/cache": volume}, cpu=2, memory=4096, timeout=600, max_containers=12)
def measure(model_name):
    from transformers import AutoTokenizer
    hf=TOKENIZERS[model_name]
    tokenizer=AutoTokenizer.from_pretrained(hf,cache_dir="/cache/hf",trust_remote_code=model_name=="glm4:9b")
    return {"model":model_name,"hf":hf,"counts":{name:len(tokenizer(text,add_special_tokens=False)["input_ids"]) for name,text in REPRESENTATIONS.items()}}


@app.local_entrypoint()
def main():
    results=list(measure.map(list(TOKENIZERS),order_outputs=False))
    summary={"semantic":SEMANTIC,"representations":REPRESENTATIONS,"results":sorted(results,key=lambda x:x['model'])}
    for encoding in REPRESENTATIONS:
        values=[x['counts'][encoding] for x in results]
        summary.setdefault('aggregate',{})[encoding]={"mean":sum(values)/len(values),"min":min(values),"max":max(values)}
    path=os.path.expanduser('~/neural-mirror/transport_benchmark.json')
    with open(path,'w') as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print(json.dumps(summary['aggregate'],indent=2)); print(path)
