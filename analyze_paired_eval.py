"""Summarize paired base/adapter evaluations with fixed-family and family-bootstrap CIs."""
import hashlib, json, math, random, statistics
from pathlib import Path

SOURCE=Path(__file__).with_name("paired_adapter_eval_results.json")
OUTPUT=Path(__file__).with_name("paired_adapter_eval_summary.json")
SAMPLES=20000


def percentile(values,p):
    x=sorted(values); pos=(len(x)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
    return x[lo] if lo==hi else x[lo]*(hi-pos)+x[hi]*(pos-lo)


def ci(values): return [round(percentile(values,.025),3),round(percentile(values,.975),3)]


def item_deltas(result,suite):
    s=result["suites"][suite]
    return [100*(float(a["is_correct"])-float(b["is_correct"])) for b,a in zip(s["base_rows"],s["adapter_rows"])]


def paired_item_ci(result,suites,seed=8821):
    rng=random.Random(seed); values=sum((item_deltas(result,s) for s in suites),[]); estimates=[]
    for _ in range(SAMPLES): estimates.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    return ci(estimates)


def exact_mcnemar(base_rows,adapter_rows):
    wins=sum((not b["is_correct"]) and a["is_correct"] for b,a in zip(base_rows,adapter_rows))
    losses=sum(b["is_correct"] and (not a["is_correct"]) for b,a in zip(base_rows,adapter_rows))
    n=wins+losses
    if n==0: return {"adapter_wins":wins,"adapter_losses":losses,"exact_two_sided_p":1.0}
    k=min(wins,losses)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return {"adapter_wins":wins,"adapter_losses":losses,"exact_two_sided_p":round(min(1.0,2*tail),8)}


def bootstrap(results,suites,family_resampling=False,seed=92821):
    rng=random.Random(seed); estimates=[]
    groups=[sum((item_deltas(r,s) for s in suites),[]) for r in results]
    for _ in range(SAMPLES):
        selected=[groups[rng.randrange(len(groups))] for _ in groups] if family_resampling else groups
        estimates.append(statistics.fmean(statistics.fmean(g[rng.randrange(len(g))] for _ in g) for g in selected))
    return ci(estimates)


def main():
    all_results=json.loads(SOURCE.read_text()); results=[r for r in all_results if r["status"]=="complete"]
    suites=list(results[0]["suites"]); suite_summary={}
    for suite in suites:
        deltas=[r["suites"][suite]["summary"]["accuracy_change_pp"] for r in results]
        suite_summary[suite]={
            "base_macro_accuracy":round(statistics.fmean(r["suites"][suite]["summary"]["base_accuracy"] for r in results),6),
            "adapter_macro_accuracy":round(statistics.fmean(r["suites"][suite]["summary"]["adapter_accuracy"] for r in results),6),
            "macro_change_pp":round(statistics.fmean(deltas),3),
            "fixed_family_item_bootstrap_ci95_pp":bootstrap(results,[suite]),
            "family_and_item_bootstrap_ci95_pp":bootstrap(results,[suite],True),
            "models_improved":sum(x>0 for x in deltas),"models_unchanged":sum(x==0 for x in deltas),"models_declined":sum(x<0 for x in deltas),
            "macro_prediction_flip_pct":round(statistics.fmean(r["suites"][suite]["summary"]["prediction_flip_pct"] for r in results),3),
            "macro_choice_kl":round(statistics.fmean(r["suites"][suite]["summary"]["mean_choice_kl_base_to_adapter"] for r in results),8),
        }
    all_deltas=[statistics.fmean(r["suites"][s]["summary"]["accuracy_change_pp"] for s in suites) for r in results]
    base_protocol=sum(r["generations"]["base"]["protocol_correct"] for r in results)
    adapter_protocol=sum(r["generations"]["adapter"]["protocol_correct"] for r in results)
    protocol_total=sum(r["generations"]["adapter"]["protocol_total"] for r in results)
    base_struct=sum(r["generations"]["base"]["structured_correct"] for r in results)
    adapter_struct=sum(r["generations"]["adapter"]["structured_correct"] for r in results)
    struct_total=sum(r["generations"]["adapter"]["structured_total"] for r in results)
    summary={
        "provenance":{"source":"paired_adapter_eval_results.json","source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      "protocol_diagnostic_seeds":[41041,53053]},
        "design":{"models":len(results),"suites":suites,"items_per_model_per_suite":results[0]["items_per_suite"],
                  "general_items_total":len(results)*len(suites)*results[0]["items_per_suite"],"bootstrap_samples":SAMPLES,
                  "paired_same_checkpoint":True,"adapter_disabled_base_control":True,"all_shams_passed":all(r["sham_passed"] for r in results)},
        "general_capability":{"suites":suite_summary,"overall_macro_change_pp":round(statistics.fmean(all_deltas),3),
            "fixed_family_item_bootstrap_ci95_pp":bootstrap(results,suites),
            "family_and_item_bootstrap_ci95_pp":bootstrap(results,suites,True),
            "models_net_improved":sum(x>0 for x in all_deltas),"models_net_unchanged":sum(x==0 for x in all_deltas),"models_net_declined":sum(x<0 for x in all_deltas)},
        "protocol":{"base_correct":base_protocol,"adapter_correct":adapter_protocol,"total":protocol_total,
                    "base_accuracy":round(base_protocol/protocol_total,6),"adapter_accuracy":round(adapter_protocol/protocol_total,6)},
        "structured_json":{"base_correct":base_struct,"adapter_correct":adapter_struct,"total":struct_total,
                           "base_accuracy":round(base_struct/struct_total,6),"adapter_accuracy":round(adapter_struct/struct_total,6)},
        "performance":{"median_reported_latency_change_pct":round(statistics.median(r["performance"]["latency_change_pct"] for r in results),3),
            "warning":"Base always ran before adapter, so cache warmup/order confounds latency. Do not interpret as adapter speedup."},
        "per_model":[],
        "limitations":["64 examples per benchmark and ten model families; individual-family estimates are noisy.",
            "The fixed-family CI conditions on these ten checkpoints; the family bootstrap also represents checkpoint sampling uncertainty.",
            "Multiple-choice label conditional likelihood is not a full generative benchmark implementation.",
            "Protocol seeds are fresh diagnostic seeds, not a new admission certification and must not be used for training.",
            "No random-weight or matched-norm LoRA control was included in this first paired evaluation."]}
    for r,delta in zip(results,all_deltas):
        base_rows=sum((r["suites"][s]["base_rows"] for s in suites),[])
        adapter_rows=sum((r["suites"][s]["adapter_rows"] for s in suites),[])
        summary["per_model"].append({"model":r["model"],"mean_general_change_pp":round(delta,3),
            "general_change_ci95_pp":paired_item_ci(r,suites),"discordant_accuracy_pairs":exact_mcnemar(base_rows,adapter_rows),
            "suite_change_pp":{s:r["suites"][s]["summary"]["accuracy_change_pp"] for s in suites},
            "protocol":f"{r['generations']['base']['protocol_correct']}/{r['generations']['base']['protocol_total']} -> {r['generations']['adapter']['protocol_correct']}/{r['generations']['adapter']['protocol_total']}",
            "structured_json":f"{r['generations']['base']['structured_correct']}/{r['generations']['base']['structured_total']} -> {r['generations']['adapter']['structured_correct']}/{r['generations']['adapter']['structured_total']}"})
    OUTPUT.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
