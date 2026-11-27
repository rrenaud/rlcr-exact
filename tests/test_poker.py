"""Poker evaluator vs a slow, independent reference (best 5 of 7 by explicit category rules), and known equities."""
import itertools, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlcr_exact.poker import eval7, equity, equity_mc, cards, category


def ref5(cs):
    rs = sorted((c // 4 for c in cs), reverse=True)
    flush = len({c % 4 for c in cs}) == 1
    u = sorted(set(rs), reverse=True)
    straight = None
    if len(u) == 5 and u[0] - u[4] == 4: straight = u[0]
    if u == [12, 3, 2, 1, 0]: straight = 3
    cnt = Counter(rs)
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    shape = [g[1] for g in groups]; ordered = [g[0] for g in groups]
    if straight is not None and flush: return (8, straight)
    if shape == [4, 1]: return (7, *ordered)
    if shape == [3, 2]: return (6, *ordered)
    if flush: return (5, *rs)
    if straight is not None: return (4, straight)
    if shape == [3, 1, 1]: return (3, *ordered)
    if shape == [2, 2, 1]: return (2, *ordered)
    if shape == [2, 1, 1, 1]: return (1, *ordered)
    return (0, *rs)


def ref7(cs):
    return max(ref5(x) for x in itertools.combinations(cs, 5))


def test_random_pairs_agree():
    rng = np.random.default_rng(0)
    hands = np.array([rng.choice(52, 7, replace=False) for _ in range(6000)])
    v = eval7(hands)
    r = [ref7(list(h)) for h in hands]
    assert all(category(v[i]) == r[i][0] for i in range(len(r)))
    for i in range(0, len(r) - 1):
        a, b = r[i], r[i + 1]
        assert np.sign(v[i] - v[i + 1]) == int(a > b) - int(a < b), (hands[i], hands[i + 1], a, b)


def test_targeted_categories():
    rng = np.random.default_rng(1)
    # biased deals: few ranks/suits so quads, full houses, flushes, straights, wheels and 3-pair hands are common
    hs = []
    for _ in range(4000):
        ranks = rng.choice(13, rng.integers(2, 6), replace=False)
        suits = rng.choice(4, rng.integers(1, 5), replace=False)
        deck = [r * 4 + s for r in ranks for s in suits]
        if len(deck) >= 7: hs.append(rng.choice(deck, 7, replace=False))
    for h in [cards("Ah 2d 3c 4s 5h Kd Kc"), cards("Ah Kh Qh Jh Th 9h 2c"), cards("Ac Ad Ah As Kc Kd Kh"),
              cards("2c 2d 3c 3d 4c 4d Ah"), cards("9s 9h 9d 8s 8h 8d 2c"), cards("As 2s 3s 4s 5s 6d 7d")]:
        hs.append(np.array(h))
    hs = np.array(hs); v = eval7(hs); r = [ref7(list(h)) for h in hs]
    order_v = np.argsort(v, kind="stable"); vals = [r[i] for i in order_v]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    assert len({tuple(x) for x in r}) == len(set(v.tolist()))


def test_river_and_flop_vs_reference():
    rng = np.random.default_rng(2)
    for _ in range(5):
        d = rng.choice(52, 7, replace=False).tolist()
        hero, board, vil = d[:2], d[2:5], d[5:7]
        pw, pt, n = equity(hero, board, [tuple(vil)])
        rest = [c for c in range(52) if c not in d]
        w = t = 0
        for ro in itertools.combinations(rest, 2):
            a, b = ref7(hero + board + list(ro)), ref7(vil + board + list(ro))
            w += a > b; t += a == b
        assert n == 990 and abs(pw - w / 990) < 1e-12 and abs(pt - t / 990) < 1e-12


def test_known_preflop():
    pw, pt, _ = equity_mc(cards("As Ah"), [], [tuple(cards("Ks Kh"))], n=100_000)
    assert 0.80 < pw < 0.84  # AA vs KK is about 82%
    pw, pt, _ = equity_mc(cards("As Ah"), [], None, n=100_000)
    assert 0.84 < pw < 0.86  # AA vs a random hand is about 85%


def test_river_vs_random_counts():
    pw, pt, n = equity(cards("As Kd"), cards("Qs Jd 4c 9h 2s"))
    assert n == 990
    hero = cards("As Kd") + cards("Qs Jd 4c 9h 2s")
    rest = [c for c in range(52) if c not in hero]
    w = sum(ref7(hero) > ref7(list(v) + hero[2:]) for v in itertools.combinations(rest, 2))
    assert abs(pw - w / 990) < 1e-12
