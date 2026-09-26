"""Paired base-vs-braille-adapter evaluation on identical HF checkpoints."""

import json
import math
import os
import random
import statistics
import time
import traceback
from contextlib import nullcontext

import modal

from braille_literacy import TASKS_PER_SEED, _build_tasks, _literacy_prompt, _score

app = modal.App("neural-mirror-paired-adapter-eval")
volume = modal.Volume.from_name("neural-mirror-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers==4.57.6", "huggingface_hub<1.0", "accelerate",
        "peft", "safetensors", "sentencepiece", "datasets", "protobuf", "tiktoken",
    )
    .add_local_file("braille_protocol.py", "/root/braille_protocol.py")
    .add_local_file("braille_literacy.py", "/root/braille_literacy.py")
    .add_local_file("swarm_registry.py", "/root/swarm_registry.py")
)

MODELS = {
    "qwen3:4b": {"hf": "Qwen/Qwen3-4B", "adapter": "qwen3-4b-braille-literacy-v3"},
    "llama3.2:3b": {"hf": "unsloth/Llama-3.2-3B-Instruct", "adapter": "llama3.2-3b-braille-literacy-v1"},
    "granite3.3:8b": {"hf": "ibm-granite/granite-3.3-8b-instruct", "adapter": "granite3.3-8b-braille-literacy-v1"},
    "glm4:9b": {"hf": "THUDM/glm-4-9b-chat-hf", "adapter": "glm4-9b-braille-literacy-v1", "trust_remote_code": True},
    "falcon3:7b": {"hf": "tiiuae/Falcon3-7B-Instruct", "adapter": "falcon3-7b-braille-literacy-v2"},
    "mistral:7b": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "adapter": "mistral-7b-braille-literacy-v2"},
    "solar:10.7b": {"hf": "upstage/SOLAR-10.7B-Instruct-v1.0", "adapter": "solar-10.7b-braille-literacy-v2"},
    "gemma3:4b": {"hf": "unsloth/gemma-3-4b-it", "adapter": "gemma3-4b-braille-literacy-v2", "model_class": "image_text"},
    "olmo2:7b": {"hf": "allenai/OLMo-2-1124-7B-Instruct", "adapter": "olmo2-7b-braille-literacy-v2"},
    "yi:6b": {"hf": "01-ai/Yi-6B-Chat", "adapter": "yi-6b-braille-literacy-v2"},
}
PROTOCOL_DIAGNOSTIC_SEEDS = [41041, 53053]
MMLU_SUBJECTS = ["abstract_algebra", "anatomy", "computer_security", "high_school_psychology"]


def _prefix(tokenizer, user, system="Answer accurately."):
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    try:
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except (TypeError,ValueError,AttributeError):
        try: return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        except Exception: return f"System: {system}\nUser: {user}\nAssistant:"


def _load_benchmarks(items=64):
    from datasets import load_dataset
    labels=["A","B","C","D"]
    arc=[]
    ds=load_dataset("allenai/ai2_arc","ARC-Challenge",split="validation",cache_dir="/cache/datasets")
    for row in ds:
        choices=row["choices"]
        if len(choices["text"])==4 and row["answerKey"] in choices["label"]:
            arc.append({"id":"arc:"+row["id"],"question":row["question"],"choices":choices["text"],
                        "labels":choices["label"],"answer_index":choices["label"].index(row["answerKey"])})
    random.Random(1101).shuffle(arc); arc=arc[:items]

    hs=[]
    ds=load_dataset("Rowan/hellaswag",split="validation",cache_dir="/cache/datasets")
    for index,row in enumerate(ds):
        if len(row["endings"])==4:
            hs.append({"id":f"hellaswag:{index}","question":row["ctx"],"choices":row["endings"],
                       "labels":labels,"answer_index":int(row["label"])})
    random.Random(2202).shuffle(hs); hs=hs[:items]

    mmlu=[]
    ds=load_dataset("cais/mmlu","all",split="test",cache_dir="/cache/datasets")
    per_subject=max(1,items//len(MMLU_SUBJECTS))
    for subject in MMLU_SUBJECTS:
        rows=[]
        for index,row in enumerate(ds):
            if row.get("subject")==subject:
                rows.append({"id":f"mmlu:{subject}:{index}","question":row["question"],"choices":row["choices"],
                             "labels":labels,"answer_index":int(row["answer"])})
        random.Random(3303+len(subject)).shuffle(rows); mmlu.extend(rows[:per_subject])
    return {"arc_challenge":arc,"hellaswag":hs,"mmlu":mmlu[:items]}


def _choice_prompt(tokenizer,item):
    options="\n".join(f"{label}. {text}" for label,text in zip(item["labels"],item["choices"]))
    return _prefix(tokenizer,f"{item['question']}\n\n{options}\n\nAnswer with only the correct option letter.",
                   "Answer multiple-choice questions accurately with one option letter.")


def _score_item(model,tokenizer,item):
    import torch
    prefix_ids=tokenizer(_choice_prompt(tokenizer,item),add_special_tokens=True,truncation=True,max_length=768)["input_ids"]
    sequences=[]; lengths=[]
    for label in item["labels"]:
        answer=tokenizer(" "+label,add_special_tokens=False)["input_ids"]
        sequences.append(prefix_ids+answer); lengths.append(len(answer))
    maximum=max(map(len,sequences)); pad=tokenizer.pad_token_id
    ids=torch.tensor([seq+[pad]*(maximum-len(seq)) for seq in sequences],device="cuda")
    mask=torch.tensor([[1]*len(seq)+[0]*(maximum-len(seq)) for seq in sequences],device="cuda")
    with torch.inference_mode(): logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    lp=torch.log_softmax(logits.float(),dim=-1); scores=[]; start=len(prefix_ids)
    for row,length in enumerate(lengths):
        value=0.0
        for offset in range(length): value+=float(lp[row,start+offset-1,ids[row,start+offset]])
        scores.append(value/length)
    predicted=max(range(4),key=scores.__getitem__); correct=item["answer_index"]
    best_wrong=max(score for i,score in enumerate(scores) if i!=correct)
    return {"id":item["id"],"predicted":predicted,"correct":correct,"is_correct":predicted==correct,
            "scores":scores,"correct_margin":scores[correct]-best_wrong}


def _softmax(values):
    maximum=max(values); exps=[math.exp(x-maximum) for x in values]; total=sum(exps)
    return [x/total for x in exps]


def _percentile(values,p):
    ordered=sorted(values); pos=(len(ordered)-1)*p; low=math.floor(pos); high=math.ceil(pos)
    return ordered[low] if low==high else ordered[low]*(high-pos)+ordered[high]*(pos-low)


def _bootstrap_delta(base,adapted,seed=7171,samples=2000):
    deltas=[float(a["is_correct"])-float(b["is_correct"]) for b,a in zip(base,adapted)]
    rng=random.Random(seed); means=[]
    for _ in range(samples): means.append(statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas))
    return [round(_percentile(means,.025)*100,3),round(_percentile(means,.975)*100,3)]


def _pair_summary(base,adapted):
    flips=[]; kls=[]; margin_changes=[]
    for b,a in zip(base,adapted):
        flips.append(b["predicted"]!=a["predicted"]); pb=_softmax(b["scores"]); pa=_softmax(a["scores"])
        kls.append(sum(x*math.log(max(x,1e-12)/max(y,1e-12)) for x,y in zip(pb,pa)))
        margin_changes.append(a["correct_margin"]-b["correct_margin"])
    base_acc=statistics.fmean(float(x["is_correct"]) for x in base)
    adapter_acc=statistics.fmean(float(x["is_correct"]) for x in adapted)
    return {"base_accuracy":round(base_acc,6),"adapter_accuracy":round(adapter_acc,6),
            "accuracy_change_pp":round((adapter_acc-base_acc)*100,3),
            "accuracy_change_ci95_pp":_bootstrap_delta(base,adapted),
            "prediction_flip_pct":round(statistics.fmean(flips)*100,3),
            "mean_choice_kl_base_to_adapter":round(statistics.fmean(kls),8),
            "mean_correct_margin_change":round(statistics.fmean(margin_changes),6)}


def _generate(model,tokenizer,prompt,max_new=1800):
    import torch
    inputs=tokenizer(prompt,return_tensors="pt"); inputs={k:v.to("cuda") for k,v in inputs.items()}; length=inputs["input_ids"].shape[-1]
    with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=max_new,do_sample=False,
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0,length:],skip_special_tokens=True).strip()


def _structured_tasks():
    tasks=[]
    for i in range(16):
        expected={"id":f"S{i+1}","label":["alpha","beta","gamma","delta"][i%4],"value":17+i*3,"valid":i%2==0}
        tasks.append((expected, "Return exactly this JSON object with no prose or markdown: "+json.dumps(expected,separators=(",",":"))))
    return tasks


def _eval_generations(model,tokenizer,adapter_enabled):
    context=nullcontext() if adapter_enabled else model.disable_adapter()
    structured=[]; protocol=[]
    with context:
        for expected,user in _structured_tasks():
            raw=_generate(model,tokenizer,_prefix(tokenizer,user,"Follow the exact output schema."),300)
            try: actual=json.loads(raw)
            except Exception: actual=None
            structured.append({"expected":expected,"actual":actual,"passed":actual==expected,"raw":raw})
        for seed in PROTOCOL_DIAGNOSTIC_SEEDS:
            tasks=_build_tasks(seed)
            raw=_generate(model,tokenizer,_prefix(tokenizer,"/no_think\n"+_literacy_prompt(tasks),
                          "Exact protocol conformance test. Output only schema-valid JSON."),2200)
            try: score=_score(tasks,json.loads(raw))
            except Exception as error: score={"passed":False,"correct":0,"total":TASKS_PER_SEED,"error":str(error)}
            score.update({"seed":seed,"raw_response":raw}); protocol.append(score)
    return {"structured_correct":sum(x["passed"] for x in structured),"structured_total":len(structured),
            "protocol_correct":sum(x["correct"] for x in protocol),"protocol_total":28,
            "structured":structured,"protocol":protocol}


@app.function(image=image,volumes={"/cache":volume},gpu="H100",cpu=8,memory=65536,timeout=7200,max_containers=10)
def evaluate_model(model_name:str,items:int=64):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer
    spec=MODELS[model_name]; started=time.time()
    try:
        benchmarks=_load_benchmarks(items)
        tokenizer=AutoTokenizer.from_pretrained(spec["hf"],cache_dir="/cache/hf",trust_remote_code=spec.get("trust_remote_code",False))
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        loader=AutoModelForImageTextToText if spec.get("model_class")=="image_text" else AutoModelForCausalLM
        base=loader.from_pretrained(spec["hf"],cache_dir="/cache/hf",dtype=torch.bfloat16,
            trust_remote_code=spec.get("trust_remote_code",False),low_cpu_mem_usage=True).to("cuda")
        model=PeftModel.from_pretrained(base,f"/cache/braille-adapters/{spec['adapter']}"); model.eval(); model.config.use_cache=False
        suite_results={}; torch.cuda.reset_peak_memory_stats(); base_started=time.time()
        base_rows={}
        with model.disable_adapter():
            for suite,rows in benchmarks.items(): base_rows[suite]=[_score_item(model,tokenizer,item) for item in rows]
        base_seconds=time.time()-base_started; base_peak=torch.cuda.max_memory_allocated()
        # Exact deterministic sham on a fixed prefix.
        with model.disable_adapter(): sham=[_score_item(model,tokenizer,item) for item in benchmarks["arc_challenge"][:8]]
        sham_passed=all(a["scores"]==b["scores"] and a["predicted"]==b["predicted"] for a,b in zip(base_rows["arc_challenge"][:8],sham))
        torch.cuda.reset_peak_memory_stats(); adapter_started=time.time(); adapter_rows={}
        for suite,rows in benchmarks.items(): adapter_rows[suite]=[_score_item(model,tokenizer,item) for item in rows]
        adapter_seconds=time.time()-adapter_started; adapter_peak=torch.cuda.max_memory_allocated()
        for suite in benchmarks:
            suite_results[suite]={"summary":_pair_summary(base_rows[suite],adapter_rows[suite]),
                                  "base_rows":base_rows[suite],"adapter_rows":adapter_rows[suite]}
        generations_base=_eval_generations(model,tokenizer,False); generations_adapter=_eval_generations(model,tokenizer,True)
        result={"model":model_name,"hf_model":spec["hf"],"adapter":spec["adapter"],"status":"complete",
                "items_per_suite":items,"sham_passed":sham_passed,"suites":suite_results,
                "generations":{"base":generations_base,"adapter":generations_adapter},
                "performance":{"base_seconds":round(base_seconds,2),"adapter_seconds":round(adapter_seconds,2),
                    "latency_change_pct":round((adapter_seconds/base_seconds-1)*100,3),
                    "base_peak_memory_bytes":base_peak,"adapter_peak_memory_bytes":adapter_peak,
                    "peak_memory_change_bytes":adapter_peak-base_peak},"elapsed_seconds":round(time.time()-started,1)}
    except Exception as error:
        result={"model":model_name,"adapter":spec["adapter"],"status":"failed","error":f"{type(error).__name__}: {error}",
                "traceback":traceback.format_exc()[-8000:],"elapsed_seconds":round(time.time()-started,1)}
    path=f"/cache/paired-adapter-eval/{model_name.replace(':','-')}.json"; os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    volume.commit(); return result


@app.function(image=image,volumes={"/cache":volume},timeout=300)
def collect_results():
    directory="/cache/paired-adapter-eval"; out=[]
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.endswith('.json'):
                with open(os.path.join(directory,name)) as f: out.append(json.load(f))
    return out


@app.local_entrypoint()
def main(collect:bool=False,items:int=64):
    if collect:
        results=collect_results.remote(); path=os.path.expanduser('~/neural-mirror/paired_adapter_eval_results.json')
        with open(path,'w') as f: json.dump(results,f,indent=2,ensure_ascii=False)
        for r in results:
            if r['status']=='complete':
                deltas={k:v['summary']['accuracy_change_pp'] for k,v in r['suites'].items()}
                print(r['model'],deltas,'protocol',r['generations']['base']['protocol_correct'],'->',r['generations']['adapter']['protocol_correct'])
            else: print(r['model'],'FAILED',r.get('error'))
        return
    for name in MODELS:
        call=evaluate_model.spawn(name,items); print(name,call.object_id)
