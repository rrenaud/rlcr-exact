"""Toy subcase of the poker equity task: one fixed heads-up matchup (default A♠K♦ vs 8♥8♣), flop only.

Every flop is enumerable: 48C3 = 17,296 flops x 990 runouts = 1.7M boards, evaluated once (~12 s with
rlcr_exact.poker.eval7), so the exact equity of every state is known. Used by notebooks/poker_akv88.py.

  enumerate_flops(hero, villain) -> dict(flops (N,3), wins (N,) runouts hero wins outright of 990, ties, eq)
  features(flops, kind)          -> (N, d) float32: "onehot" (52 cards) or "onehot+counts" (+ rank / suit counts)
  categories(flops, hero, vil)   -> per-flop label a poker player would name (set for villain, hero pairs, miss...)
  sampled_targets(wins, k, rng)  -> mean of k runouts drawn WITHOUT replacement (hypergeometric): k = 1 is one
                                    sampled showdown (RLCR's label), k = 990 is the exact equity
  fit(X, y, ...)                 -> small torch model trained full-batch on Brier (= squared error, a proper score)
"""
import itertools, os, time
from math import comb

import numpy as np

from . import poker

N_RUNOUTS = 990  # C(45, 2): turn + river after a flop with four hole cards dead


def _colex_index(pos):
    """Index of each sorted k-combination (rows of pos, ascending positions) in colex order: sum_i C(pos_i, i+1)."""
    k = pos.shape[1]
    table = np.array([[comb(n, i + 1) for n in range(64)] for i in range(k)], dtype=np.int64)
    return sum(table[i][pos[:, i]] for i in range(k))


def enumerate_flops(hero, villain, cache_dir=os.path.expanduser("~/.cache/poker_toy")):
    hero, villain = list(hero), list(villain)
    key = "_".join(poker.names(hero + villain).split())
    path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return dict(flops=z["flops"], wins=z["wins"], ties=z["ties"], eq=z["wins"] / N_RUNOUTS, seconds=0.0, cached=True)
    t0 = time.time()
    rest = np.array([c for c in range(52) if c not in hero + villain])
    flops = np.array(list(itertools.combinations(range(len(rest)), 3)), dtype=np.int64)  # positions in rest
    fidx = _colex_index(flops)
    order = np.argsort(fidx); flops = flops[order]  # row i <-> colex index i
    wins = np.zeros(len(flops), np.int64); ties = np.zeros(len(flops), np.int64)
    boards = itertools.combinations(range(len(rest)), 5)
    H, V = np.array(hero), np.array(villain)
    while True:
        chunk = np.array(list(itertools.islice(boards, 300_000)), dtype=np.int64)
        if not len(chunk):
            break
        b = rest[chunk]; n = len(b)
        hs = poker.eval7(np.concatenate([np.tile(H, (n, 1)), b], 1))
        vs = poker.eval7(np.concatenate([np.tile(V, (n, 1)), b], 1))
        w, t = (hs > vs).astype(np.int64), (hs == vs).astype(np.int64)
        for tri in itertools.combinations(range(5), 3):  # each 5-card board completes 10 flops
            ids = _colex_index(chunk[:, list(tri)])
            np.add.at(wins, ids, w); np.add.at(ties, ids, t)
    flops = rest[flops]
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, flops=flops, wins=wins, ties=ties)
    return dict(flops=flops, wins=wins, ties=ties, eq=wins / N_RUNOUTS, seconds=time.time() - t0, cached=False)


def features(flops, kind="onehot"):
    n = len(flops)
    X = np.zeros((n, 52), np.float32); X[np.arange(n)[:, None], flops] = 1
    if kind == "onehot":
        return X
    R, S = flops // 4, flops % 4
    rc = np.stack([(R == r).sum(1) for r in range(13)], 1)
    sc = np.stack([(S == s).sum(1) for s in range(4)], 1)
    return np.concatenate([X, rc, sc], 1).astype(np.float32)


def categories(flops, hero, villain):
    """Coarse labels by the hero's and villain's ranks and suits (works for any pair-vs-two-cards matchup)."""
    hr = {c // 4 for c in hero}; vr = {c // 4 for c in villain}; hs = {c % 4 for c in hero}
    R, S = flops // 4, flops % 4
    v_hit = np.isin(R, list(vr)).any(1)
    h_hit = np.isin(R, list(hr)).any(1)
    fd = np.stack([(S == s).sum(1) for s in hs], 1).max(1) >= 2  # two to the hero's suit on board
    out = np.where(v_hit, "villain sets", np.where(h_hit, "hero pairs", np.where(fd, "miss + flush draw", "miss")))
    return out


CATEGORY_ORDER = ["villain sets", "miss", "miss + flush draw", "hero pairs"]
_RANK_WORD = dict(zip("23456789TJQKA", ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                         "jack", "queen", "king", "ace"]))
_PLURAL = {"six": "sixes"}


def category_labels(hero, villain):
    """Plain-language label per category key, for a reader who knows the rules but not forum jargon
    ("hero" = you, "villain" = opponent, "set" = three of a kind made with a pocket pair)."""
    w = lambda c: _RANK_WORD[poker.RANKS[c // 4]]
    pl = lambda word: _PLURAL.get(word, word + "s")
    vr = sorted({w(c) for c in villain}); hr = sorted({w(c) for c in hero}, key=lambda x: -list(_RANK_WORD.values()).index(x))
    opp = (f"opponent makes three {pl(vr[0])}" if len(vr) == 1 else f"a {' or '.join(vr)} flops for your opponent")
    you = f"you pair your {' or '.join(hr)}" if len(hr) > 1 else f"you make three {pl(hr[0])}"
    ranks = hr + vr
    none = ranks[0] if len(ranks) == 1 else ", ".join(ranks[:-1]) + " or " + ranks[-1]
    suited = len({c % 4 for c in hero}) == 1
    fd = "two cards of your suit" if suited else "two cards of one of your suits"
    return {"villain sets": opp, "hero pairs": you, "miss": f"no {none} on the flop",
            "miss + flush draw": f"no {none}, but {fd} (flush draw)"}


def sampled_targets(wins, k, rng):
    """Mean outcome of k distinct runouts per flop (hypergeometric): k = 1 one showdown, k = 990 exact."""
    return rng.hypergeometric(wins, N_RUNOUTS - wins, k) / k


def fit(X_tr, y_tr, hidden=32, epochs=300, lr=3e-3, wd=1e-4, seed=0, threads=None):
    """Full-batch Adam on squared error through a sigmoid (Brier on soft or hard targets). Returns predict(X)."""
    import torch
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    d = X_tr.shape[1]
    m = (torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
         if hidden else torch.nn.Linear(d, 1))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((torch.sigmoid(m(Xt).squeeze(1)) - yt) ** 2).mean()
        loss.backward(); opt.step()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(X)).squeeze(1)).numpy()
    return predict


def scores(q, p):
    """Prediction q against exact equity p: MSE to truth, Brier vs oracle (= MSE + irreducible p(1-p)), correlation,
    and the Murphy split against the truth with bins = 20 equal-width prediction bins."""
    q, p = np.asarray(q, float), np.asarray(p, float)
    out = dict(mse=float(((q - p) ** 2).mean()), brier_oracle=float((p * (1 - q) ** 2 + (1 - p) * q ** 2).mean()),
               floor=float((p * (1 - p)).mean()), corr=float(np.corrcoef(q, p)[0, 1]) if q.std() > 0 else 0.0)
    b = np.minimum((q * 20).astype(int), 19); rel = res = 0.0; pb = p.mean()
    for i in np.unique(b):
        m = b == i; w = m.mean(); pi = p[m].mean()
        rel += w * (q[m].mean() - pi) ** 2; res += w * (pi - pb) ** 2
    out.update(REL=float(rel), RES=float(res), UNC=float(pb * (1 - pb)))
    return out
