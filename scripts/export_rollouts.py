"""Export per-rollout answers of every rlcr11 pre-registered run (Stages A, B, C and the reference panel) for analysis.

Reads OUT/prereg_eval/{dev,test}_<tag>.jsonl.gz written by prereg_eval.py under
ROOT/rlcr-prereg-{A,B,C,ref}/run-*/ and writes, into results/rollouts/ (or --out):

  rollouts_{dev,test}.csv.gz    one row per (run, checkpoint, question): the last <answer>, the parsed <confidence>,
                                RLCR's correctness label, format_ok, truncated, completion length
  runs.csv                      one row per run: stage, cid, arm, replicate, seed, lr, W, F, prereg rev, gpuq state
  questions_{dev,test}.jsonl.gz question id, gold answer(s), source (HotpotQA, from the rollout cache)
  completions_{dev,test}.jsonl.gz (--with-text only) the full completion text; large, see README

Confidence and answer parsing match rlcr_exact.online._confidence / _last (copied here, since online.py imports
vLLM): the last <confidence> parsed into [0, 1], else null. prereg_eval.py scores a null confidence as 0.

usage (any env with pandas):
  python3 scripts/export_rollouts.py [--root /data/gpuq-runs/rrenaud] [--out results/rollouts] [--with-text]
"""
import argparse, glob, gzip, json, os, re

import pandas as pd

ROOT = "/data/gpuq-runs/rrenaud"
CACHE = "rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"  # prompts and gold answers (prereg_eval.py's CACHE)
SPLIT_N = {"dev": 500, "test": 500}
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _last(tag, text):
    found = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return found[-1].strip() if found else None


def _confidence(text):
    try:
        c = float(_last("confidence", text))
    except (TypeError, ValueError):
        return None
    return c if 0 <= c <= 1 else None


def configs():
    out = {}
    for d in ("prereg", "prereg_rlcr_init"):
        for c in json.load(open(os.path.join(REPO, "docs", d, "configs.json")))["configs"]:
            out[c["cid"]] = c
    return out


def run_meta(d, cfg):
    m = json.load(open(os.path.join(d, "manifest.json")))
    meta = dict(run=os.path.basename(d), experiment=os.path.basename(os.path.dirname(d)), state=m.get("state"),
                description=m.get("description"), parent_run=m.get("parent_run"))
    hit = re.match(r"prereg@(\S+) (\S+) (exact|sampled)$", m.get("description") or "")
    if hit:
        rev, cid, arm = hit.groups()
        base, rep = re.match(r"([A-C]\d+)(?:r(\d+))?$", cid).groups()
        c = cfg.get(base, {})
        meta |= dict(rev=rev, stage=base[0], cid=cid, base_cid=base, arm=arm, replicate=int(rep or 0),
                     seed=c.get("seed", 0) + 1000 * int(rep or 0) if c else None,
                     lr=c.get("lr"), W=c.get("W"), F=c.get("F"))
    else:
        meta |= dict(stage="ref" if meta["experiment"] == "rlcr-prereg-ref" else None)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "rollouts"))
    ap.add_argument("--stages", default="A,B,C,ref")
    ap.add_argument("--with-text", action="store_true", help="also write the full completions (large)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = configs()

    runs, rows, texts = [], {s: [] for s in SPLIT_N}, {s: [] for s in SPLIT_N}
    for stage in args.stages.split(","):
        for d in sorted(glob.glob(os.path.join(args.root, f"rlcr-prereg-{stage}", "run-*"))):
            if not os.path.exists(os.path.join(d, "manifest.json")):
                continue
            meta = run_meta(d, cfg)
            if meta["stage"] is None:  # not a prereg run (e.g. a smoke test in the same experiment)
                continue
            files = sorted(glob.glob(os.path.join(d, "prereg_eval", "*.jsonl.gz")))
            meta["n_eval_files"] = len(files)
            cs = [json.loads(l).get("compute_s", 0) for f in glob.glob(os.path.join(d, "train_*.jsonl")) for l in open(f)
                  if '"checkpoint"' in l]
            meta["gpu_h"] = round(max(cs, default=0) / 3600, 3)  # training compute only: a lower bound on GPU time
            runs.append(meta)
            for f in files:
                split, tag = re.match(r"(dev|test)_(.+)\.jsonl\.gz$", os.path.basename(f)).groups()
                hit_step, hit_regen = re.match(r"step(\d+)$", tag), re.match(r"(.+)_regen(\d+)$", tag)
                key = dict(run=meta["run"], stage=meta["stage"], cid=meta.get("cid") or (hit_regen and hit_regen[1]),
                           arm=meta.get("arm"), step=int(hit_step[1]) if hit_step else None,
                           regen=int(hit_regen[2]) if hit_regen else None)
                for line in gzip.open(f, "rt"):
                    q = json.loads(line)
                    t = q["outputs"][0]
                    rows[split].append(key | dict(
                        qid=q["id"], answer=_last("answer", t), confidence=_confidence(t),
                        confidence_text=_last("confidence", t), correct=float(q["correct"][0]),
                        format_ok=float(q["format_ok"][0]), truncated=bool(q["truncated"][0]), n_chars=len(t)))
                    if args.with_text:
                        texts[split].append(dict(run=meta["run"], step=key["step"], regen=key["regen"], qid=q["id"],
                                                 completion=t))

    rs = pd.DataFrame(runs)
    for k in ("replicate", "seed"):
        rs[k] = rs[k].astype("Int64")
    rs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    for split in SPLIT_N:
        df = pd.DataFrame(rows[split])
        if not len(df):
            continue
        df["step"], df["regen"] = df["step"].astype("Int64"), df["regen"].astype("Int64")
        df = df.sort_values(["stage", "cid", "arm", "run", "step", "regen", "qid"], na_position="first")
        df.to_csv(os.path.join(args.out, f"rollouts_{split}.csv.gz"), index=False,
                  compression=dict(method="gzip", mtime=0))  # mtime 0: re-exports of the same data are byte-identical
        print(f"{split}: {len(df):,} rows, {df['run'].nunique()} runs, "
              f"{df.groupby(['run', 'step', 'regen'], dropna=False).ngroups} checkpoints")
        if args.with_text:
            with gzip.GzipFile(os.path.join(args.out, f"completions_{split}.jsonl.gz"), "wb", mtime=0) as f:
                f.write("".join(json.dumps(t) + "\n" for t in texts[split]).encode())
        cache = os.path.join(args.root, CACHE, f"rollouts_hotpot_fresh_{split}.jsonl")
        qs = [json.loads(line) for line in list(open(cache))[: SPLIT_N[split]]]
        with gzip.GzipFile(os.path.join(args.out, f"questions_{split}.jsonl.gz"), "wb", mtime=0) as f:
            f.write("".join(json.dumps(dict(id=q["id"], answer=q["answer"], source=q.get("source"))) + "\n"
                            for q in qs).encode())
    gpu_h = float(rs["gpu_h"].sum())
    json.dump(dict(gpu_h=round(gpu_h, 2), budget_mb=round(gpu_h, 2)), open(os.path.join(args.out, "budget.json"), "w"))
    print(f"GPU-hours (training compute): {gpu_h:.1f} -> publish budget {gpu_h:.1f} MB (CLAUDE.md: 1 MB per GPU-hour)")
    print("runs by stage/arm:\n" + rs.groupby(["stage", "arm"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
