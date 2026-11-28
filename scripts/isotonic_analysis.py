"""How much does post-hoc isotonic recalibration help each arm (exact vs sampled)?

For every checkpoint that has greedy completions on both splits (prereg_eval/{dev,test}_stepN.jsonl.gz), fit an
isotonic map stated confidence -> P(correct) on its 500 dev questions (PAV over the 11 levels) and apply it to its 500
test questions. Confidence conventions are prereg_eval's: last <confidence> in [0, 1], missing = 0.

Brier = REL - RES + UNC (Murphy, grouping by stated level). Isotonic recalibration attacks REL only: it cannot reorder
questions or split a level, so the Brier it leaves is (up to fit noise and monotonicity) the refinement UNC - RES.
Hence, per checkpoint on test:
  gain     = Brier(raw) - Brier(iso, fit on dev)          what a labelled dev set buys after training
  REL, RES Murphy terms of the raw stated levels           gain ~ REL (in-sample ceiling), RES ~ discrimination

Paired comparison per (stage, config, seed) at a common step (the last checkpoint) and at each arm's own best-dev step:
  d_gain = gain(sampled) - gain(exact)       > 0: isotonic helps sampled more
  d_iso  = Brier_iso(sampled) - Brier_iso(exact)   > 0: exact still better after both are recalibrated
Uncertainty: bootstrap over questions, dev and test resampled jointly across arms (same question ids) with the isotonic
map refit every replicate, so fit noise is included. That interval is conditional on the trained runs; the across-pair
sign count and t interval treat runs as the unit.

usage: isotonic_analysis.py [--root /data/gpuq-runs/rrenaud] [--stages ABC] [--boot 2000] [--out FILE.md]
       isotonic_analysis.py --selftest     (synthetic root, checks the pipeline end to end)
"""
import argparse, collections, glob, gzip, json, math, os, re, sys, tempfile

import numpy as np

ROOT = "/data/gpuq-runs/rrenaud"
REFS = ("hotpot-v2-brier-7b-no-split", "hotpot-v2-correctness-7b")


def confidence(text):
    found = re.findall(r"<confidence>(.*?)</confidence>", text, re.S)
    try:
        c = float(found[-1].strip())
    except (IndexError, ValueError):
        return None
    return c if 0 <= c <= 1 else None


def load_completions(path):
    """id -> (correct, confidence with missing = 0)."""
    out = {}
    for q in map(json.loads, gzip.open(path, "rt")):
        c = confidence(q["outputs"][0])
        out[q["id"]] = (float(q["correct"][0]), 0.0 if c is None else c)
    return out


# ---------- isotonic (PAV) on discrete levels ----------

def iso_fit(c, y):
    """Weighted PAV over the distinct levels of c. Returns (knots, values) for np.interp (clipped at the ends)."""
    lv, inv = np.unique(c, return_inverse=True)
    w = np.bincount(inv, minlength=len(lv)).astype(float)
    s = np.bincount(inv, weights=y, minlength=len(lv))
    vals, wts, cnt = [], [], []  # blocks
    for k in range(len(lv)):
        vals.append(s[k] / w[k]); wts.append(w[k]); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, n2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, n1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(n1 + n2)
    fitted = np.repeat(vals, cnt)
    return lv, fitted


def iso_apply(fit, c):
    lv, fv = fit
    return np.interp(c, lv, fv) if len(lv) > 1 else np.full(len(c), fv[0])


def murphy(c, y):
    """REL, RES, UNC grouping by the stated level (exact identity: Brier = REL - RES + UNC)."""
    lv, inv = np.unique(c, return_inverse=True)
    n = len(y); w = np.bincount(inv) / n
    ybar = np.bincount(inv, weights=y) / np.bincount(inv)
    base = y.mean()
    return float(np.sum(w * (lv - ybar) ** 2)), float(np.sum(w * (ybar - base) ** 2)), float(base * (1 - base))


def auroc(c, y):
    pos, neg = c[y == 1], c[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney with average ranks for ties
    allc = np.concatenate([pos, neg]); order = np.argsort(allc, kind="mergesort"); ranks = np.empty(len(allc))
    sc = allc[order]; i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc[j + 1] == sc[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def arrays(d, ids):
    a = np.array([d[i] for i in ids])
    return a[:, 1], a[:, 0]  # c, y


def checkpoint_stats(dev, test, dev_ids, test_ids):
    cd, yd = arrays(dev, dev_ids); ct, yt = arrays(test, test_ids)
    p = iso_apply(iso_fit(cd, yd), ct)
    raw, iso = float(np.mean((ct - yt) ** 2)), float(np.mean((p - yt) ** 2))
    rel, res, unc = murphy(ct, yt)
    lv, cnt = np.unique(ct, return_counts=True)
    pr = cnt / cnt.sum()
    return dict(raw=raw, iso=iso, gain=raw - iso, rel=rel, res=res, unc=unc, auroc=auroc(ct, yt), auroc_iso=auroc(p, yt),
                acc=float(yt.mean()), mean_conf=float(ct.mean()), levels=len(lv), level_entropy=float(-(pr * np.log(pr)).sum()))


def gain_iso(dev, test, dev_ids, test_ids, di, ti):
    """Brier raw and iso on a bootstrap resample (index arrays into the id lists)."""
    cd, yd = arrays(dev, dev_ids); ct, yt = arrays(test, test_ids)
    cd, yd, ct, yt = cd[di], yd[di], ct[ti], yt[ti]
    p = iso_apply(iso_fit(cd, yd), ct)
    raw = np.mean((ct - yt) ** 2)
    return raw, np.mean((p - yt) ** 2)


# ---------- data ----------

def load_runs(root, stages):
    runs = []
    for stage in stages:
        for d in sorted(glob.glob(os.path.join(root, f"rlcr-prereg-{stage}", "run-*"))):
            try:
                m = json.load(open(os.path.join(d, "manifest.json")))
            except OSError:
                continue
            hit = re.match(r"prereg@(\S+) (\S+) (exact|sampled)$", m.get("description", ""))
            if not hit:
                continue
            ev = collections.defaultdict(dict)
            for f in glob.glob(os.path.join(d, "prereg_eval", "*_step*.json")):
                mt = re.match(r"(dev|test)_step(\d+)\.json$", os.path.basename(f))
                if not mt:  # e.g. dev2_stepN.json (Stage B amendment split)
                    continue
                split, step = mt.groups()
                ev[int(step)][split] = json.load(open(f))
            both = sorted(s for s in ev if all(os.path.exists(os.path.join(d, "prereg_eval", f"{sp}_step{s}.jsonl.gz"))
                                               for sp in ("dev", "test")))
            runs.append(dict(dir=d, run=os.path.basename(d), stage=stage, cid=hit[2], arm=hit[3], evals=ev, both=both))
    return runs


def best_dev_step(r):
    """Lowest dev Brier among steps with both splits dumped (the prereg selection also applies eligibility; the steps
    with test dumps are already the ones it picked)."""
    cand = [s for s in r["both"] if "brier" in r["evals"][s].get("dev", {})]
    return min(cand, key=lambda s: r["evals"][s]["dev"]["brier"]) if cand else None


def comp(r, step):
    p = lambda sp: os.path.join(r["dir"], "prereg_eval", f"{sp}_step{step}.jsonl.gz")
    return load_completions(p("dev")), load_completions(p("test"))


def boot_pairs(pairs, nboot, seed=0):
    """pairs: list of ((dev_e, test_e), (dev_s, test_s)). Joint question bootstrap across every pair (shared ids).
    Returns per-pair arrays of d_gain, d_iso, d_raw over replicates, and the pooled mean."""
    dev_ids = sorted(set.intersection(*[set(x[0]) for p in pairs for x in p]))
    test_ids = sorted(set.intersection(*[set(x[1]) for p in pairs for x in p]))
    rng = np.random.default_rng(seed)
    out = np.zeros((nboot, len(pairs), 3))
    for b in range(nboot):
        di = rng.integers(0, len(dev_ids), len(dev_ids)); ti = rng.integers(0, len(test_ids), len(test_ids))
        for k, (e, s) in enumerate(pairs):
            re_, ie = gain_iso(*e, dev_ids, test_ids, di, ti)
            rs, is_ = gain_iso(*s, dev_ids, test_ids, di, ti)
            out[b, k] = ((rs - is_) - (re_ - ie), is_ - ie, rs - re_)
    return out, dev_ids, test_ids


def ci(x):
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def across_runs(d):
    d = np.asarray(d); n = len(d)
    wins = int((d > 0).sum())
    p = sum(math.comb(n, k) for k in range(max(wins, n - wins), n + 1)) / 2 ** n * 2 if n else float("nan")
    if n > 1:
        from statistics import stdev
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
             12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 21: 2.086, 31: 2.042}
        tq = t.get(n, t[max(k for k in t if k <= n)])
        h = tq * stdev(d) / math.sqrt(n)
        return wins, min(p, 1.0), (float(d.mean() - h), float(d.mean() + h))
    return wins, float("nan"), (float("nan"), float("nan"))


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}" if x < 0 or nd < 0 else f"{x:.{nd}f}"


def analyze(root, stages, nboot):
    runs = load_runs(root, stages)
    L = ["# Isotonic recalibration: exact vs sampled", "",
         f"root `{root}`, stages {stages}; {len(runs)} runs, {sum(bool(r['both']) for r in runs)} with dev+test completions. "
         f"Isotonic fit on each checkpoint's dev completions, scored on test; bootstrap {nboot} joint question resamples.", ""]
    # per checkpoint table
    L += ["## Per checkpoint (test)", "",
          "| stage | config | arm | step | acc | Brier raw | Brier iso | gain | REL | RES | AUROC | levels | level H |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cache = {}
    for r in sorted(runs, key=lambda r: (r["stage"], r["cid"], r["arm"])):
        for s in r["both"]:
            dv, ts = cache[(r["dir"], s)] = comp(r, s)
            ids_d, ids_t = sorted(dv), sorted(ts)
            st = checkpoint_stats(dv, ts, ids_d, ids_t)
            L.append(f"| {r['stage']} | {r['cid']} | {r['arm']} | {s} | {st['acc']:.3f} | {st['raw']:.4f} | {st['iso']:.4f} | "
                     f"{st['gain']:+.4f} | {st['rel']:.4f} | {st['res']:.4f} | {st['auroc']:.3f} | {st['levels']} | {st['level_entropy']:.2f} |")
    # reference models
    refrows = []
    for m in REFS:
        fd = sorted(glob.glob(os.path.join(root, "rlcr-prereg-ref", "run-*", "prereg_eval", f"dev_{m}_regen0.jsonl.gz")))
        ft = sorted(glob.glob(os.path.join(root, "rlcr-prereg-ref", "run-*", "prereg_eval", f"test_{m}_regen0.jsonl.gz")))
        if fd and ft:
            dv, ts = load_completions(fd[0]), load_completions(ft[0])
            st = checkpoint_stats(dv, ts, sorted(dv), sorted(ts))
            refrows.append(f"| {m} | {st['raw']:.4f} | {st['iso']:.4f} | {st['gain']:+.4f} | {st['rel']:.4f} | {st['res']:.4f} | {st['auroc']:.3f} | {st['levels']} |")
    if refrows:
        L += ["", "## Reference models (regen 0)", "", "| model | Brier raw | Brier iso | gain | REL | RES | AUROC | levels |",
              "|---|---|---|---|---|---|---|---|"] + refrows

    # pairs
    by = collections.defaultdict(dict)
    for r in runs:
        by[(r["stage"], r["cid"])][r["arm"]] = r
    for mode in ("last", "best-dev"):
        pairs, labels = [], []
        for (stage, cid), arms in sorted(by.items()):
            if len(arms) < 2:
                continue
            e, s = arms["exact"], arms["sampled"]
            if mode == "last":
                common = sorted(set(e["both"]) & set(s["both"]))
                if not common:
                    continue
                se = ss = common[-1]
            else:
                se, ss = best_dev_step(e), best_dev_step(s)
                if se is None or ss is None:
                    continue
            pairs.append((cache.get((e["dir"], se)) or comp(e, se), cache.get((s["dir"], ss)) or comp(s, ss)))
            labels.append((stage, cid, se, ss))
        if not pairs:
            continue
        bs, dev_ids, test_ids = boot_pairs(pairs, nboot)
        L += ["", f"## Paired, {mode} checkpoint (sampled minus exact; d_gain > 0: isotonic helps sampled more; "
                  f"d_iso > 0: exact better after recalibration)", "",
              f"{len(dev_ids)} dev / {len(test_ids)} test questions common to all.", "",
              "| stage | config | steps e/s | d_raw [95%] | d_iso [95%] | d_gain [95%] | share of raw gap closed |",
              "|---|---|---|---|---|---|---|"]
        point = []
        for k, (stage, cid, se, ss) in enumerate(labels):
            e, s = pairs[k]
            de = checkpoint_stats(*e, dev_ids, test_ids); ds = checkpoint_stats(*s, dev_ids, test_ids)
            dg, di, dr = ds["gain"] - de["gain"], ds["iso"] - de["iso"], ds["raw"] - de["raw"]
            point.append((dg, di, dr))
            share = f"{dg / dr:.0%}" if abs(dr) > 1e-9 else "—"
            L.append(f"| {stage} | {cid} | {se}/{ss} | {dr:+.4f} [{ci(bs[:, k, 2])[0]:+.4f}, {ci(bs[:, k, 2])[1]:+.4f}] | "
                     f"{di:+.4f} [{ci(bs[:, k, 1])[0]:+.4f}, {ci(bs[:, k, 1])[1]:+.4f}] | "
                     f"{dg:+.4f} [{ci(bs[:, k, 0])[0]:+.4f}, {ci(bs[:, k, 0])[1]:+.4f}] | {share} |")
        P = np.array(point)
        for stage in sorted({l[0] for l in labels}) + ["all"]:
            ix = [i for i, l in enumerate(labels) if stage == "all" or l[0] == stage]
            if not ix:
                continue
            pooled = bs[:, ix, :].mean(1)
            L.append("")
            L.append(f"- **{('stage ' + stage) if stage != 'all' else 'all stages'}**, {len(ix)} pairs, mean over pairs:")
            for j, name in ((2, "d_raw "), (1, "d_iso "), (0, "d_gain")):
                w, p, (lo, hi) = across_runs(P[ix, j])
                lo_b, hi_b = ci(pooled[:, j])
                L.append(f"  - {name} {P[ix, j].mean():+.4f}; question bootstrap [{lo_b:+.4f}, {hi_b:+.4f}]; "
                         f"across pairs: {w}/{len(ix)} positive, sign test p = {fmt(p)}, t interval [{fmt(lo, 4)}, {fmt(hi, 4)}]")
    return "\n".join(L) + "\n"


# ---------- self test ----------

def selftest():
    """Synthetic root: both arms share the same latent difficulty ranking; sampled's stated level is a monotone
    distortion of exact's (overconfident) plus coarser levels. Expect d_gain > 0 and d_iso ~ small."""
    rng = np.random.default_rng(1)
    root = tempfile.mkdtemp(prefix="iso_selftest_")
    n = 500
    lat = {sp: rng.normal(size=n) for sp in ("dev", "test")}
    y = {sp: (rng.random(n) < 1 / (1 + np.exp(-1.5 * lat[sp]))).astype(float) for sp in lat}
    levels = np.round(np.arange(11) / 10, 1)

    def stated(sp, arm, noise):
        z = 1.5 * lat[sp] + rng.normal(scale=noise, size=n)
        p = 1 / (1 + np.exp(-z))
        if arm == "sampled":
            p = np.clip(p ** 0.4, 0, 1)  # overconfident, monotone
            return np.round(np.round(p * 5) / 5, 1)  # coarser: 6 levels
        return levels[np.round(p * 10).astype(int)]

    for stage, cids in (("B", ["B0", "B1", "B2"]), ("C", ["C0", "C0r1", "C0r2"])):
        for cid in cids:
            for arm in ("exact", "sampled"):
                d = os.path.join(root, f"rlcr-prereg-{stage}", f"run-{cid}-{arm}")
                os.makedirs(os.path.join(d, "prereg_eval"))
                json.dump(dict(description=f"prereg@abc {cid} {arm}"), open(os.path.join(d, "manifest.json"), "w"))
                for step in (300, 600):
                    for sp in ("dev", "test"):
                        c = stated(sp, arm, 0.8)
                        with gzip.open(os.path.join(d, "prereg_eval", f"{sp}_step{step}.jsonl.gz"), "wt") as f:
                            for i in range(n):
                                txt = "<answer>x</answer>" + ("" if (arm == "sampled" and i % 97 == 0) else f"<confidence>{c[i]}</confidence>")
                                f.write(json.dumps(dict(id=f"{sp}{i}", outputs=[txt], correct=[y[sp][i]])) + "\n")
                        json.dump(dict(brier=float(np.mean((c - y[sp]) ** 2))), open(os.path.join(d, "prereg_eval", f"{sp}_step{step}.json"), "w"))
    # PAV unit checks
    lv, fv = iso_fit(np.array([0., 0., .5, .5, 1., 1.]), np.array([1., 0., 0., 0., 1., 1.]))
    assert np.allclose(fv, [1 / 4, 1 / 4, 1]), fv
    c = rng.random(300).round(1); yy = (rng.random(300) < c).astype(float)
    rel, res, unc = murphy(c, yy)
    assert abs(np.mean((c - yy) ** 2) - (rel - res + unc)) < 1e-12
    text = analyze(root, "BC", 200)
    print(text)
    assert "d_gain" in text
    print("selftest ok:", root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--stages", default="ABC")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    text = analyze(a.root, a.stages, a.boot)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
