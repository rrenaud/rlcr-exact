"""One line per finished hotpot-rlvr run: config, final dev/test greedy metrics, conditional means, answer KL,
and the generation check (gen_eval/gen_eval.json) when present. Sorted by dev greedy Brier.
usage: hp_table.py [EXPERIMENT_DIR]"""
import glob, json, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen"
out = []
for path in glob.glob(f"{root}/run-*/train_*.jsonl"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    a = next((r["args"] for r in rows if r["event"] == "data"), {})
    if a.get("steps", 0) < 100 or a.get("train_problems"):
        continue
    dev = [e for e in rows if e["event"] == "eval" and e["split"] == "dev"]
    test = [e for e in rows if e["event"] == "eval" and e["split"] == "test"]
    if not dev or not test:
        continue
    d, t = dev[-1], test[-1]
    run = path.split("/")[-2]
    g = os.path.join(root, run, "gen_eval", "gen_eval.json")
    gen = json.load(open(g)) if os.path.exists(g) else None
    cfg = (f"{a['objective']:7s} lr={a['lr']:<7g} W={a.get('answer_kl', 0):<6g} F={a.get('support_floor', 0):<6g}"
           + (f" wsd={a.get('warmup_steps', 0)}/{a.get('decay_frac', 0):g}" if a.get("warmup_steps") or a.get("decay_frac") else ""))
    line = (f"{cfg:52s} dev {d['greedy_brier']:.3f}/{d['greedy_auroc']:.3f} c|y {d.get('greedy_mean_correct', float('nan')):.2f}/"
            f"{d.get('greedy_mean_wrong', float('nan')):.2f} kl {d.get('answer_kl', float('nan')):.2f}  test {t['greedy_brier']:.3f}/{t['greedy_auroc']:.3f}")
    if gen:
        line += "  gen " + " ".join(f"{s}:acc{r['accuracy_delta']:+.3f} chg{r['answer_changed']:.2f} B{r['lora']['brier']:.3f} A{r['lora']['auroc']:.3f}"
                                    for s, r in gen.items())
    out.append((d["greedy_brier"], line + f"  [{run[4:19]}]"))
for _, line in sorted(out):
    print(line)
