"""Pre-registered analysis (docs/prereg_fair_sweep.md). Two phases, so test cannot steer any choice:

  prereg_analysis.py select   dev only: eligibility, frontiers, paired steps; writes docs/prereg/selections.json, the
                              (run, step) pairs whose test split the rule says to score, grouped per run for prereg_eval.py
  prereg_analysis.py report   adds the test scores of exactly those checkpoints: frontier tables, H1-H3, references

Inputs: the runs of gpuq experiments rlcr-prereg-A / -B (description "prereg@REV CID ARM"): their checkpoint events
(compute_s per registered step) and prereg_eval/{dev,test}_stepN.json; the reference panel (rlcr-prereg-ref).
--root and --runs let the dry run point it at synthetic data.
"""
import argparse, collections, glob, gzip, json, math, os, re

import numpy as np

ROOT = "/data/gpuq-runs/rrenaud"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "prereg")
A_CACHE_GPU_H = (33 + 121) / 60  # gpuq runs 04bf759137 + 4511bf5189: the 4k-question rollout cache + KL reference
BUDGETS = {"A": [0.05, 0.1, 0.25, 0.5, 1, 2], "B": [0.25, 0.5, 1, 2, 4, 7]}  # training GPU-hours
H1_STEP = {"A": 300, "B": 200}
NOISE = 0.02  # eligibility: answer EM and format may not fall more than this below the RLVR start
RLCR = "hotpot-v2-brier-7b-no-split"
RLVR = "hotpot-v2-correctness-7b"


def load_runs(root):
    runs = []
    for stage in "AB":
        for d in sorted(glob.glob(os.path.join(root, f"rlcr-prereg-{stage}", "run-*"))):
            m = json.load(open(os.path.join(d, "manifest.json")))
            hit = re.match(r"prereg@(\S+) (\S+) (exact|sampled)$", m["description"])
            if not hit:
                continue
            ck = {}
            for f in glob.glob(os.path.join(d, "train_*.jsonl")):
                for line in open(f):
                    e = json.loads(line)
                    if e["event"] == "checkpoint":
                        ck[e["step"]] = e["compute_s"] / 3600  # the last event wins (a step redone after a resume)
            ev = {}
            for f in glob.glob(os.path.join(d, "prereg_eval", "*_step*.json")):
                split, step = re.match(r"(dev|test)_step(\d+)\.json$", os.path.basename(f)).groups()
                ev.setdefault(int(step), {})[split] = json.load(open(f))
            runs.append(dict(dir=d, run=os.path.basename(d), rev=hit[1], cid=hit[2], arm=hit[3], stage=stage,
                             replicate=hit[2].count("r") > 0, gpu_h=ck, evals=ev, done=m.get("state") == "completed"))
    return runs


def load_refs(root):
    refs = collections.defaultdict(lambda: collections.defaultdict(list))  # model -> split -> [metrics per regen]
    for f in sorted(glob.glob(os.path.join(root, "rlcr-prereg-ref", "run-*", "prereg_eval", "*_regen*.json"))):
        split, model = re.match(r"(dev|test)_(.+)_regen\d+\.json$", os.path.basename(f)).groups()
        refs[model][split].append(json.load(open(f)))
    return refs


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def points(runs, base):
    """Every eligible (run, step) with a dev score: the candidates of every selection."""
    out = []
    for r in runs:
        for step, e in sorted(r["evals"].items()):
            if "dev" not in e or step not in r["gpu_h"]:
                continue
            d = e["dev"]
            ok = d["em"] >= base["em"] - NOISE and d["format_ok"] >= base["format_ok"] - NOISE
            out.append(dict(r=r, step=step, gpu_h=r["gpu_h"][step], dev=d, eligible=ok))
    return out


def select(runs, refs):
    base = {k: mean([m[k] for m in refs[RLVR]["dev"]]) for k in ("em", "format_ok")}
    pts = [p for p in points(runs, base) if not p["r"]["replicate"]]
    picks = collections.defaultdict(set)  # run dir -> steps to score on test
    frontier = []
    for stage in "AB":
        for arm in ("exact", "sampled", "any"):
            for b in BUDGETS[stage]:
                cand = [p for p in pts if p["r"]["stage"] == stage and (arm == "any" or p["r"]["arm"] == arm)
                        and p["eligible"] and p["gpu_h"] <= b]
                if cand:
                    p = min(cand, key=lambda p: p["dev"]["brier"])
                    frontier.append(dict(stage=stage, arm=arm, budget_h=b, run=p["r"]["run"], cid=p["r"]["cid"],
                                         step=p["step"], gpu_h=round(p["gpu_h"], 3), dev_brier=p["dev"]["brier"]))
                    picks[p["r"]["dir"]].add(p["step"])
    best = {}
    for r in runs:  # every run: its last checkpoint (H2) and its best eligible dev checkpoint
        if not r["evals"]:
            continue
        last = max(r["gpu_h"]) if r["gpu_h"] else None
        if last is not None and "dev" in r["evals"].get(last, {}):
            picks[r["dir"]].add(last)
        mine = [p for p in points([r], base) if p["eligible"]]
        if mine:
            p = min(mine, key=lambda p: p["dev"]["brier"])
            picks[r["dir"]].add(p["step"])
            best[r["run"]] = p["step"]
    # replication (registered): per stage and arm, the config of the full-budget frontier point, 2 new seeds
    replicate = sorted({(f["cid"], f["arm"]) for f in frontier if f["arm"] != "any" and f["budget_h"] == BUDGETS[f["stage"]][-1]})
    sel = dict(base_dev=base, frontier=frontier, best_dev_step=best, replicate=replicate,
               test=[dict(run_dir=d, steps=sorted(s)) for d, s in sorted(picks.items())])
    return sel


def paired(runs, stage, step, split):
    by = collections.defaultdict(dict)
    for r in runs:
        if r["stage"] == stage and not r["replicate"] and split in r["evals"].get(step, {}):
            by[r["cid"]][r["arm"]] = r["evals"][step][split]["brier"]
    rows = [(c, v["exact"], v["sampled"]) for c, v in sorted(by.items()) if len(v) == 2]
    d = [s - e for _, e, s in rows]  # > 0: exact better
    wins = sum(x > 0 for x in d)
    p = sum(math.comb(len(d), k) for k in range(wins, len(d) + 1)) / 2 ** len(d) if d else float("nan")
    return rows, d, wins, p


def confidence(text):
    """The last <confidence> in [0, 1], as rlcr_exact.online._confidence (that module needs vLLM)."""
    found = re.findall(r"<confidence>(.*?)</confidence>", text, re.S)
    try:
        c = float(found[-1].strip())
    except (IndexError, ValueError):
        return None
    return c if 0 <= c <= 1 else None


def per_question(path):
    """Per-question (correct, confidence with missing = 0) from a saved completions file."""
    out = {}
    for q in map(json.loads, gzip.open(path, "rt")):
        c = confidence(q["outputs"][0])
        out[q["id"]] = (float(q["correct"][0]), 0.0 if c is None else c)
    return out


CACHE = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"


def zero_training_baselines():
    """Post-hoc recalibration of the RLVR start, fit on the 500 dev questions and applied to the 500 test questions
    (the realistic protocol, not fit on test): the dev base rate; Platt scaling of RLVR's stated confidence; Platt
    scaling of its answer-token probability (geometric mean over the <answer> tokens, closing-tag tokens excluded).
    All from the cache's greedy RLVR completions; their correctness labels are the cache's (RLCR's reward)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    feats = {}
    for split, n in (("dev", 500), ("test", 500)):
        f = os.path.join(CACHE, f"rollouts_hotpot_fresh_{split}.jsonl")
        rows = [json.loads(l) for l in open(f)][:n]
        top = {}
        for t in map(json.loads, gzip.open(f + ".topk.jsonl.gz", "rt")):
            if t["j"] == 0 and t.get("answer"):
                a = t["answer"]
                lp = [x for x, e in zip(a["sampled_logprob"], a["edge"]) if not e]
                top[t["id"]] = math.exp(sum(lp) / len(lp)) if lp else None
        y = np.array([float(r["correct"][0]) for r in rows])
        conf = np.array([confidence(r["outputs"][0]) or 0.0 for r in rows])
        ap = np.array([top.get(r["id"]) if top.get(r["id"]) is not None else 0.5 for r in rows])
        feats[split] = (y, conf, ap, sum(top.get(r["id"]) is None for r in rows))
    logit = lambda x: np.log(np.clip(x, 1e-3, 1 - 1e-3) / (1 - np.clip(x, 1e-3, 1 - 1e-3)))[:, None]
    yd, cd, ad, _ = feats["dev"]; yt, ct, at, miss = feats["test"]
    out = {"dev base rate (constant)": np.full(len(yt), yd.mean()), "RLVR stated confidence, as is": ct}
    for name, xd, xt in (("Platt(RLVR stated confidence)", cd, ct), ("Platt(answer-token probability)", ad, at)):
        out[name] = LogisticRegression(C=1e6).fit(logit(xd), yd).predict_proba(logit(xt))[:, 1]
    res = {k: (float(np.mean((v - yt) ** 2)), float(roc_auc_score(yt, v)) if np.ptp(v) > 0 else 0.5) for k, v in out.items()}
    return res, miss


def boot_diff(a, b, n=10000, seed=0):
    """Paired bootstrap over questions of Brier(a) - Brier(b): mean and 95% interval."""
    ids = sorted(set(a) & set(b))
    da = np.array([(a[i][1] - a[i][0]) ** 2 for i in ids]); db = np.array([(b[i][1] - b[i][0]) ** 2 for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (n, len(ids)))
    diffs = (da[idx] - db[idx]).mean(1)
    return float((da - db).mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def report(runs, refs, sel):
    lines = ["# Pre-registered results", ""]
    ref = lambda m, s, k: mean([x[k] for x in refs[m][s] if x.get(k) is not None])
    lines += ["## Reference panel (our pipeline; mean over regenerations, spread = max - min)", "",
              "| model | split | n | accuracy | em | Brier | AUROC | ECE | Brier spread |", "|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(refs):
        for s in ("dev", "test"):
            xs = refs[m][s]
            if xs:
                br = [x["brier"] for x in xs]
                lines.append(f"| {m} | {s} | {len(xs)} | {ref(m, s, 'accuracy'):.3f} | {ref(m, s, 'em'):.3f} | {mean(br):.3f} | "
                             f"{ref(m, s, 'auroc'):.3f} | {ref(m, s, 'ece'):.3f} | {max(br) - min(br):.3f} |")
    base, miss = zero_training_baselines()
    lines += ["", f"## Zero-training baselines (fit on dev, applied to test; {miss} test answers without token probabilities scored 0.5)", "",
              "| baseline | test Brier | test AUROC |", "|---|---|---|"]
    lines += [f"| {k} | {b:.3f} | {a:.3f} |" for k, (b, a) in base.items()]
    rlcr_q = None
    f = sorted(glob.glob(os.path.join(ROOT, "rlcr-prereg-ref", "run-*", "prereg_eval", f"test_{RLCR}_regen0.jsonl.gz")))
    if f:
        rlcr_q = per_question(f[0])
    rd = {r["dir"]: r for r in runs}
    lines += ["", "## Frontier (selected on dev; test of the selected checkpoint)", "",
              "| stage | arm | budget h | config | step | train h | +cache h | dev Brier | test Brier | test AUROC | test em | vs RLCR test Brier [95%] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fr in sel["frontier"]:
        r = next(r for r in runs if r["run"] == fr["run"])
        t = r["evals"].get(fr["step"], {}).get("test")
        if t is None:
            lines.append(f"| {fr['stage']} | {fr['arm']} | {fr['budget_h']} | {fr['cid']} | {fr['step']} | {fr['gpu_h']:.2f} | | "
                         f"{fr['dev_brier']:.3f} | (not scored) | | | |")
            continue
        vs = ""
        q = os.path.join(r["dir"], "prereg_eval", f"test_step{fr['step']}.jsonl.gz")
        if rlcr_q and os.path.exists(q):
            m, lo, hi = boot_diff(per_question(q), rlcr_q)
            vs = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        cache = f"{fr['gpu_h'] + A_CACHE_GPU_H:.2f}" if fr["stage"] == "A" else ""
        lines.append(f"| {fr['stage']} | {fr['arm']} | {fr['budget_h']} | {fr['cid']} | {fr['step']} | {fr['gpu_h']:.2f} | {cache} | "
                     f"{fr['dev_brier']:.3f} | {t['brier']:.3f} | {t['auroc'] or float('nan'):.3f} | {t['em']:.3f} | {vs} |")
    h3 = [fr for fr in sel["frontier"] if fr["stage"] == "A" and fr["arm"] == "any" and fr["budget_h"] <= 1]
    if h3 and rlcr_q:
        fr = max(h3, key=lambda f: f["budget_h"])
        r = next(r for r in runs if r["run"] == fr["run"])
        q = os.path.join(r["dir"], "prereg_eval", f"test_step{fr['step']}.jsonl.gz")
        if os.path.exists(q):
            m, lo, hi = boot_diff(per_question(q), rlcr_q)
            em = r["evals"][fr["step"]]["test"]["em"]
            lines += ["", f"## H3: Stage A, at most 1 GPU-h of training ({fr['gpu_h']:.2f} h, {fr['cid']} {r['arm']} step {fr['step']})", "",
                      f"- test Brier minus RLCR's: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]; within 0.02: {'yes' if hi <= 0.02 else 'no'} "
                      f"(upper bound {hi:+.3f}); test em {em:.3f}"]
    lines += ["", "## H1 (speed): paired dev Brier at the registered step, sampled minus exact (> 0: exact better)", ""]
    for stage in "AB":
        rows, d, wins, p = paired(runs, stage, H1_STEP[stage], "dev")
        lines.append(f"- Stage {stage}, step {H1_STEP[stage]}: " + ", ".join(f"{c} {s - e:+.3f}" for c, e, s in rows) +
                     f"; exact better in {wins}/{len(d)}, one-sided sign test p = {p:.3f}, median {np.median(d) if d else float('nan'):+.3f}")
    lines += ["", "## H2 (endpoint): paired test Brier at the last checkpoint, sampled minus exact (equivalence margin 0.01)", ""]
    for stage in "AB":
        last = 1000 if stage == "A" else 600
        rows, d, wins, p = paired(runs, stage, last, "test")
        lines.append(f"- Stage {stage}: " + ", ".join(f"{c} {s - e:+.3f}" for c, e, s in rows) +
                     f"; mean {mean(d):+.3f}, exact better in {wins}/{len(d)} (sign test p = {p:.3f})")
    lines += ["", "## Instability: test Brier at the best-dev checkpoint vs the last one, per run", "",
              "| run | config | arm | best-dev step | test there | last-step test |", "|---|---|---|---|---|---|"]
    for r in runs:
        b = sel["best_dev_step"].get(r["run"])
        last = max(r["gpu_h"]) if r["gpu_h"] else None
        tb = r["evals"].get(b, {}).get("test", {}).get("brier") if b else None
        tl = r["evals"].get(last, {}).get("test", {}).get("brier") if last else None
        lines.append(f"| {r['run'][-10:]} | {r['cid']} | {r['arm']} | {b} | {tb if tb is None else round(tb, 3)} | {tl if tl is None else round(tl, 3)} |")
    return "\n".join(lines) + "\n"


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "report"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    ROOT = args.root
    runs, refs = load_runs(args.root), load_refs(args.root)
    print(f"{len(runs)} runs, reference models: {sorted(refs)}")
    path = os.path.join(args.out, "selections.json")
    if args.phase == "select":
        sel = select(runs, refs)
        json.dump(sel, open(path, "w"), indent=1)
        print(f"wrote {path}: {len(sel['frontier'])} frontier points, {sum(len(t['steps']) for t in sel['test'])} test checkpoints")
        for t in sel["test"]:
            print(t["run_dir"], ",".join(map(str, t["steps"])))
    else:
        sel = json.load(open(path))
        text = report(runs, refs, sel)
        open(os.path.join(args.out, "results.md"), "w").write(text)
        print(text)


if __name__ == "__main__":
    main()
