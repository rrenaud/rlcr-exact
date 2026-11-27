"""CPU toy: how RLCR's estimator choices change what a verbalized-confidence head learns.

Answers are fixed; only an 11-level confidence head trains, by RL on RLCR's reward
R(z, c) = 0.5 format (always 1 here) + 0.5 z + 0.5 (1 - (c - z)^2).

World. A question has a true solve rate p (Beta, fitted so groups of 8 are all-wrong / all-right at the rates our
HotpotQA training logs show: 33% / 46%). Each rollout is right with probability p. The head sees
  x = logit(p) + N(0, sigma_x)                  (a noisy read of the question's difficulty)
  s = mu (2z - 1) + N(0, 1)                      (a weak read of its own answer's correctness)
through fixed RBF features, with linear logits over the levels (shared parameters across questions). The best
possible confidence is the Bayes posterior P(z = 1 | x, s), whose Brier is the floor.

Knobs (the Pitfalls paper's RLCR, arXiv 2609.32470, uses sampled-shared + std + filter):
  estimator  exact     sum_l q_l R_l, analytic gradient over every level
             shared    k = 1: draw c ~ q, advantage R(z, c) - group mean (RLCR; correctness noise included)
             segmented k = 1: advantage on the Brier part only (CoCA-style separate confidence advantage)
  std        divide advantages by the group's reward std (GRPO / DAPO default)
  filter     none | refill: keep drawing questions until the step has 16 groups that are neither all right nor
             all wrong (DAPO dynamic sampling)
  init       rlvr: the RLVR checkpoint's prior (95% of mass on 0.9 / 1.0); calibsft: a short supervised warm
             start on the paper's target 0.5 * group solve rate + 0.5 * z (group of 50), before RL
  opt        adam | sgd

Prediction under the filter if the head knew only p: the Brier optimum moves to (p - p^G) / (1 - p^G - (1-p)^G),
0.87 at p = 0.99 and 0.15 at p = 0.05.

  python scripts/toy_filter.py --out /path/results.jsonl         # the grid (~minutes on CPU)
  python scripts/toy_filter.py --report /path/results.jsonl      # tables
"""
import argparse, itertools, json, math, os, sys

import numpy as np
import torch

LEVELS = torch.linspace(0, 1, 11)
G, QPS = 8, 16  # rollouts per question, questions per step


def fit_beta(all_wrong=0.33, all_right=0.46, g=G):
    """Beta(a, b) whose groups of g are all wrong / all right at the given rates (grid search on E[(1-p)^g], E[p^g])."""
    from scipy.special import beta as B
    best = None
    for a in np.exp(np.linspace(np.log(0.05), np.log(3), 120)):
        for b in np.exp(np.linspace(np.log(0.05), np.log(3), 120)):
            aw, ar = B(a, b + g) / B(a, b), B(a + g, b) / B(a, b)
            err = (aw - all_wrong) ** 2 + (ar - all_right) ** 2
            if best is None or err < best[0]:
                best = (err, a, b)
    return best[1], best[2]


class World:
    def __init__(self, a, b, sigma_x, mu, rng):
        self.a, self.b, self.sigma_x, self.mu, self.rng = a, b, sigma_x, mu, rng

    def questions(self, n):
        p = self.rng.beta(self.a, self.b, n).clip(1e-4, 1 - 1e-4)
        x = np.log(p / (1 - p)) + self.rng.normal(0, self.sigma_x, n)
        return p, x

    def rollouts(self, p, x, g):
        z = (self.rng.random((len(p), g)) < p[:, None]).astype(np.float32)
        s = self.mu * (2 * z - 1) + self.rng.normal(0, 1, z.shape)
        return z, np.repeat(x[:, None], g, 1), s


def features(x, s):
    """[..., F]: RBF bumps over x (difficulty read) and over s (own-answer read), plus a bias."""
    cx = torch.linspace(-12, 12, 13); cs = torch.linspace(-3, 3, 7)
    fx = torch.exp(-0.5 * ((x[..., None] - cx) / 2.0) ** 2)
    fs = torch.exp(-0.5 * ((s[..., None] - cs) / 0.8) ** 2)
    return torch.cat([fx, fs, torch.ones_like(x)[..., None]], -1)


F = 13 + 7 + 1
RLVR_PRIOR = torch.tensor([.001, .001, .001, .001, .002, .003, .005, .02, .05, .24, .676])


def reward(z, c):  # RLCR, format always ok
    return 0.5 + 0.5 * z + 0.5 * (1 - (c - z) ** 2)


def brier_part(z, c):
    return 0.5 * (1 - (c - z) ** 2)


def bayes_posterior(world, x, s):
    """P(z = 1 | x, s) by numerical integration over p (the floor any confidence head can reach)."""
    grid = torch.linspace(1e-4, 1 - 1e-4, 2001, dtype=torch.float64)
    lp = torch.log(grid / (1 - grid))
    prior = (world.a - 1) * torch.log(grid) + (world.b - 1) * torch.log(1 - grid)
    lik_x = -0.5 * ((x.double()[..., None] - lp) / world.sigma_x) ** 2
    w = torch.softmax(prior + lik_x, -1)
    ep = (w * grid).sum(-1)  # E[p | x]
    l1 = -0.5 * (s.double() - world.mu) ** 2; l0 = -0.5 * (s.double() + world.mu) ** 2
    return (ep * l1.exp() / (ep * l1.exp() + (1 - ep) * l0.exp())).float()


def evaluate(W, world, rng_seed=12345, n=20000):
    rng = np.random.default_rng(rng_seed)
    ev = World(world.a, world.b, world.sigma_x, world.mu, rng)
    p, x = ev.questions(n)
    z, xx, s = ev.rollouts(p, x, 1)
    z, xx, s = map(lambda a: torch.tensor(a[:, 0], dtype=torch.float32), (z, xx, s))
    with torch.no_grad():
        q = torch.softmax(features(xx, s) @ W, -1)
        c = q @ LEVELS
        bay = bayes_posterior(world, xx, s)
    from sklearn.metrics import roc_auc_score
    edges = torch.linspace(0, 1, 11)
    idx = torch.bucketize(c, edges[1:-1])
    ece = sum((idx == k).float().mean() * abs(c[idx == k].mean() - z[idx == k].mean())
              for k in range(10) if (idx == k).any()).item()
    marg = q.mean(0)
    pt = torch.tensor(p)
    bins = {"p<0.1": pt < 0.1, "p>0.9": pt > 0.9, "p>0.98": pt > 0.98, "p<0.02": pt < 0.02}
    return dict(brier=((c - z) ** 2).mean().item(), bayes_brier=((bay - z) ** 2).mean().item(), ece=ece,
                auroc=roc_auc_score(z.numpy(), c.numpy()),
                marg_support=math.exp(-(marg * marg.clamp_min(1e-12).log()).sum().item()),
                row_entropy=-(q * q.clamp_min(1e-12).log()).sum(-1).mean().item(),
                **{f"conf_{k}": c[m].mean().item() for k, m in bins.items() if m.any()},
                **{f"bayes_{k}": bay[m].mean().item() for k, m in bins.items() if m.any()},
                **{f"acc_{k}": z[m].mean().item() for k, m in bins.items() if m.any()})


def calibsft_init(W, world, rng, steps=150, lr=0.03, n_group=50, lam=0.5):
    """The paper's CalibSFT on the confidence head: cross-entropy to the target lam * (solve rate of 50) + (1 - lam) z,
    rounded to the nearest level, target-balanced by level (their 'balanced to the smallest group', here by weights)."""
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(steps):
        p, x = world.questions(64)
        z, xx, s = world.rollouts(p, x, n_group)
        rate = z.mean(1, keepdims=True)
        tgt = lam * rate + (1 - lam) * z
        lev = torch.tensor(np.rint(tgt * 10).astype(np.int64)).flatten()
        counts = torch.bincount(lev, minlength=11).float().clamp_min(1)
        wts = (1 / counts)[lev]; wts = wts / wts.sum()
        logq = torch.log_softmax(features(torch.tensor(xx).flatten().float(), torch.tensor(s).flatten().float()) @ W, -1)
        loss = -(wts * logq[torch.arange(len(lev)), lev]).sum()
        opt.zero_grad(); loss.backward(); opt.step()


def run(cfg, world_seed=0):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    world = World(cfg["a"], cfg["b"], cfg["sigma_x"], cfg["mu"], rng)
    W = torch.zeros(F, 11)
    W.data[-1] = RLVR_PRIOR.log()  # the bias feature carries the prior
    W.data[:-1] = 0.01 * torch.randn(F - 1, 11)
    W.requires_grad_(True)
    if cfg["init"] == "calibsft":
        calibsft_init(W, world, rng)
    opt = (torch.optim.Adam if cfg["opt"] == "adam" else torch.optim.SGD)([W], lr=cfg["lr"])
    drawn = 0
    curve = []
    for step in range(1, cfg["steps"] + 1):
        zs, xs, ss = [], [], []
        while len(zs) < QPS:
            p, x = world.questions(QPS)
            z, xx, s = world.rollouts(p, x, G)
            drawn += QPS
            for i in range(QPS):
                if cfg["filter"] == "refill" and z[i].min() == z[i].max():
                    continue
                if len(zs) < QPS:
                    zs.append(z[i]); xs.append(xx[i]); ss.append(s[i])
            if cfg["filter"] == "refill" and drawn > 400 * step:
                break  # safety: never seen at these rates
        z = torch.tensor(np.stack(zs)); xx = torch.tensor(np.stack(xs)).float(); s = torch.tensor(np.stack(ss)).float()
        logq = torch.log_softmax(features(xx, s) @ W, -1)  # [Q, G, L]
        q = logq.exp()
        if cfg["estimator"] == "exact":
            R = reward(z[..., None], LEVELS)  # [Q, G, L]
            loss = -(q * R).sum(-1)
            if cfg["std"]:  # the analogue: scale each group by its exact-reward std
                r = (q * R).sum(-1).detach()
                loss = loss / (r.std(1, keepdim=True) + 1e-4)
            loss = loss.mean()
        else:
            j = torch.multinomial(q.detach().reshape(-1, 11), 1).reshape(z.shape)
            c = LEVELS[j]
            r = reward(z, c) if cfg["estimator"] == "shared" else brier_part(z, c)
            adv = r - r.mean(1, keepdim=True)
            if cfg["std"]:
                adv = adv / (r.std(1, keepdim=True) + 1e-4)
            loss = -(adv * logq.gather(-1, j[..., None])[..., 0]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in cfg.get("curve_steps", ()):
            curve.append(dict(step=step, **{k: v for k, v in evaluate(W.detach(), world, n=5000).items()
                                            if k in ("brier", "auroc", "marg_support")}))
    out = dict(cfg, rollout_questions=drawn, **evaluate(W.detach(), world))
    if curve:
        out["curve"] = curve
    return out


def grid(args):
    a, b = fit_beta()
    print(f"Beta({a:.3f}, {b:.3f})", file=sys.stderr)
    base = dict(a=a, b=b, sigma_x=args.sigma_x, mu=args.mu, steps=args.steps)
    cells = []
    # main grid: estimator x std x filter x init, Adam, lr swept
    for est, std, filt, init in itertools.product(["exact", "shared", "segmented"], [False, True],
                                                  ["none", "refill"], ["rlvr", "calibsft"]):
        cells.append(dict(estimator=est, std=std, filter=filt, init=init, opt="adam"))
    # optimizer check (variance vs mean gradient): plain setup under SGD
    for est in ["exact", "shared", "segmented"]:
        cells.append(dict(estimator=est, std=False, filter="none", init="rlvr", opt="sgd"))
    lrs = {"adam": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "sgd": [1.0, 3.0, 10.0, 30.0, 100.0]}
    jobs = [dict(base, **cell, lr=lr, seed=seed) for cell in cells for lr in lrs[cell["opt"]] for seed in range(args.seeds)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers, initializer=torch.set_num_threads, initargs=(1,)) as pool, \
            open(args.out, "w") as f:
        for i, res in enumerate(pool.imap_unordered(run, jobs)):
            f.write(json.dumps(res) + "\n"); f.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)}", file=sys.stderr)


def report(path):
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    keys = ["estimator", "std", "filter", "init", "opt"]
    # per cell: the lr with the best mean Brier over seeds (tuning on the same eval: optimistic, equally for all)
    agg = df.groupby(keys + ["lr"]).mean(numeric_only=True).reset_index()
    best = agg.loc[agg.groupby(keys)["brier"].idxmin()]
    cols = keys + ["lr", "brier", "bayes_brier", "ece", "auroc", "marg_support", "row_entropy",
                   "conf_p<0.02", "conf_p<0.1", "conf_p>0.9", "conf_p>0.98", "acc_p>0.98", "rollout_questions"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(best[[c for c in cols if c in best]].round(3).sort_values(["opt", "init", "filter", "std", "brier"]).to_string(index=False))
    sd = df.merge(best[keys + ["lr"]], on=keys + ["lr"]).groupby(keys)["brier"].std()
    print("\nseed sd of Brier at the chosen lr: median %.4f, max %.4f" % (sd.median(), sd.max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--report")
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigma-x", type=float, default=4.0); ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="one run per estimator, 30 steps")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    elif args.smoke:
        a, b = fit_beta()
        for est in ["exact", "shared", "segmented"]:
            for filt in ["none", "refill"]:
                r = run(dict(a=a, b=b, sigma_x=args.sigma_x, mu=args.mu, steps=30, estimator=est, std=True, filter=filt,
                             init="calibsft", opt="adam", lr=1e-2, seed=0))
                print(est, filt, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    else:
        grid(args)
