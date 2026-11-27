"""Fit the toy's question-difficulty distribution to the real "k of 8 right" histogram, several ways, and dump
everything the explainer page draws (loss surfaces, fits, implied quantities) to JSON.

Data: B-stage training logs (RLVR start, 10 runs x 600 steps x 16 questions), groups with all 8 rollouts logged.
  python scripts/toy_world_fit.py results/toy_filter/world_fit.json
"""
import json, sys

import numpy as np
from scipy import optimize, stats
from scipy.special import betaln, gammaln, expit, logit

G = 8
COUNTS = np.array([28794, 3244, 2089, 1752, 1854, 2086, 2793, 4706, 38754], float)  # prereg-B, all steps
EMP = COUNTS / COUNTS.sum()
K = np.arange(G + 1)
LOGC = gammaln(G + 1) - gammaln(K + 1) - gammaln(G - K + 1)
PGRID = np.linspace(1e-6, 1 - 1e-6, 4001)


def bb_pmf(a, b):  # beta-binomial: P(k of G right) when p ~ Beta(a, b)
    return np.exp(LOGC + betaln(K + a, G - K + b) - betaln(a, b))


def mix_pmf(w, ps):  # a distribution of p given as weights on points
    return (w[:, None] * stats.binom.pmf(K[None], G, ps[:, None])).sum(0)


def nll(pmf):
    return -(COUNTS * np.log(np.clip(pmf, 1e-300, None))).sum()


SAT = nll(EMP)  # saturated model: the best any distribution of p could possibly do


def endpoint_loss(a, b):
    pm = bb_pmf(a, b)
    return (pm[0] - EMP[0]) ** 2 + (pm[-1] - EMP[-1]) ** 2


def summarize(name, pmf, density=None, extra=None):
    # implied quantities the toy cares about; moments from pmf (identified) where possible
    k = K / G
    out = dict(name=name, pmf=pmf.tolist(), excess_nll=nll(pmf) - SAT, chi2=float(((COUNTS - COUNTS.sum() * pmf) ** 2 /
                                                                                     (COUNTS.sum() * pmf)).sum()),
               keep=float(1 - pmf[0] - pmf[-1]))
    if density is not None:
        out["density"] = density
    return out | (extra or {})


def main(path):
    res = {"counts": COUNTS.tolist(), "emp": EMP.tolist(), "G": G}
    la = np.linspace(np.log(0.02), np.log(3), 121); lb = la.copy()
    res["grid_log_a"] = la.tolist(); res["grid_log_b"] = lb.tolist()
    E = np.zeros((len(la), len(lb))); N = np.zeros_like(E); P0 = np.zeros_like(E); P8 = np.zeros_like(E)
    for i, x in enumerate(la):
        for j, y in enumerate(lb):
            pm = bb_pmf(np.exp(x), np.exp(y))
            E[i, j] = endpoint_loss(np.exp(x), np.exp(y)); N[i, j] = (nll(pm) - SAT) / COUNTS.sum()
            P0[i, j], P8[i, j] = pm[0], pm[-1]
    res["endpoint_surface"] = E.tolist(); res["nll_surface"] = N.tolist()
    res["P0_surface"] = P0.tolist(); res["P8_surface"] = P8.tolist()

    fits = []
    # 1. endpoint match (what the toy used: fit_beta's grid search, refined)
    r = optimize.minimize(lambda t: endpoint_loss(*np.exp(t)), np.log([0.11, 0.09]), method="Nelder-Mead",
                          options=dict(xatol=1e-10, fatol=1e-14))
    a1, b1 = np.exp(r.x)
    fits.append(summarize("beta_endpoints", bb_pmf(a1, b1), extra=dict(a=a1, b=b1)))
    # 2. beta-binomial maximum likelihood on all 9 bins
    r = optimize.minimize(lambda t: nll(bb_pmf(*np.exp(t))), np.log([0.1, 0.1]), method="Nelder-Mead",
                          options=dict(xatol=1e-10, fatol=1e-8))
    a2, b2 = np.exp(r.x)
    fits.append(summarize("beta_mle", bb_pmf(a2, b2), extra=dict(a=a2, b=b2)))
    # 3. logit-normal MLE (p = sigmoid(N(m, s)))
    zq, wq = np.polynomial.hermite_e.hermegauss(200); wq = wq / wq.sum()
    ln_pmf = lambda m, s: mix_pmf(wq, expit(m + s * zq))
    r = optimize.minimize(lambda t: nll(ln_pmf(t[0], np.exp(t[1]))), [0.3, np.log(4)], method="Nelder-Mead")
    m3, s3 = r.x[0], np.exp(r.x[1])
    fits.append(summarize("logit_normal", ln_pmf(m3, s3), extra=dict(m=m3, s=s3)))
    # 4. spike-and-slab: point masses at 0 and 1 plus a Beta for the rest (4 parameters)
    def ss_pmf(t):
        w = np.exp(t[:3]) / np.exp(t[:3]).sum(); a, b = np.exp(t[3:])
        return w[0] * (K == 0) + w[1] * (K == G) + w[2] * bb_pmf(a, b)
    r = optimize.minimize(lambda t: nll(ss_pmf(t)), [0, 0, -1, 0, 0], method="Nelder-Mead",
                          options=dict(maxiter=20000, xatol=1e-9, fatol=1e-9))
    w4 = np.exp(r.x[:3]) / np.exp(r.x[:3]).sum(); a4, b4 = np.exp(r.x[3:])
    fits.append(summarize("spike_slab", ss_pmf(r.x), extra=dict(w0=w4[0], w1=w4[1], w_slab=w4[2], a=a4, b=b4)))
    # 5. NPMLE: weights on a grid of p by EM (the most flexible distribution; matches the histogram if it can)
    ps = np.linspace(0, 1, 201); w = np.full(len(ps), 1 / len(ps))
    L = stats.binom.pmf(K[None], G, ps[:, None])  # [P, K]
    for _ in range(20000):
        post = w[:, None] * L; post /= post.sum(0, keepdims=True)
        w = (post * COUNTS[None]).sum(1) / COUNTS.sum()
    fits.append(summarize("npmle", mix_pmf(w, ps), extra=dict(atoms=[[float(p), float(x)] for p, x in zip(ps, w) if x > 1e-3])))

    # densities for plotting (on a grid avoiding the endpoints), and implied facts each world asserts
    pg = np.linspace(0.0025, 0.9975, 400)
    for f in fits:
        if f["name"] in ("beta_endpoints", "beta_mle"):
            f["density"] = stats.beta.pdf(pg, f["a"], f["b"]).tolist()
            draws = stats.beta.rvs(f["a"], f["b"], size=400000, random_state=0)
        elif f["name"] == "logit_normal":
            f["density"] = (stats.norm.pdf(logit(pg), f["m"], f["s"]) / (pg * (1 - pg))).tolist()
            draws = expit(stats.norm.rvs(f["m"], f["s"], size=400000, random_state=0))
        elif f["name"] == "spike_slab":
            f["density"] = (f["w_slab"] * stats.beta.pdf(pg, f["a"], f["b"])).tolist()
            rng = np.random.default_rng(0); u = rng.random(400000)
            draws = np.where(u < f["w0"], 0.0, np.where(u < f["w0"] + f["w1"], 1.0, stats.beta.rvs(f["a"], f["b"], size=400000, random_state=1)))
        else:
            draws = np.random.default_rng(0).choice(ps, 400000, p=w / w.sum())
        f["share_p_lt_0.02"] = float((draws < 0.02).mean()); f["share_p_gt_0.98"] = float((draws > 0.98).mean())
        f["difficulty_brier_floor"] = float((draws * (1 - draws)).mean())  # best Brier knowing p exactly
        f["mean_p"] = float(draws.mean())
    res["p_grid"] = pg.tolist(); res["fits"] = fits
    res["note"] = ("E[p(1-p)] = P(two rollouts disagree)/2 is identified by the histogram (needs moments 1-2 only); "
                   "every quantity of the form E[f(p)] with f a polynomial of degree <= 8 is identified")
    # identified directly from the histogram: E[p(1-p)] = (1/2) P(two random rollouts of a question disagree)
    disagree = sum(EMP[k] * 2 * k * (G - k) / (G * (G - 1)) for k in K)
    res["identified_brier_floor"] = disagree / 2
    json.dump(res, open(path, "w"))
    for f in fits:
        print(f"{f['name']:15s} excessNLL {f['excess_nll']:9.1f}  chi2 {f['chi2']:9.1f}  keep {f['keep']:.3f}  "
              f"floor {f['difficulty_brier_floor']:.4f}  p<.02 {f['share_p_lt_0.02']:.3f}  p>.98 {f['share_p_gt_0.98']:.3f}  "
              + " ".join(f"{k}={v:.3f}" for k, v in f.items() if k in ("a", "b", "m", "s", "w0", "w1", "w_slab")))
        print("   pmf", np.round(f["pmf"], 4).tolist())
    print("emp     ", EMP.round(4).tolist())
    print("identified E[p(1-p)] =", round(res["identified_brier_floor"], 4))


if __name__ == "__main__":
    main(sys.argv[1])
