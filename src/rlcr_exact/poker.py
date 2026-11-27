"""Texas Hold'em hand evaluation and exact showdown equity, vectorized in numpy (no poker library needed).

Cards are ints 0..51: rank = c // 4 (0 = deuce .. 12 = ace), suit = c % 4 (s, h, d, c). Text form "As", "Td", "7c".

eval7(cards) takes an (N, 7) int array and returns an (N,) int64 strength; higher wins, equal = split. The value is
category * 13**5 + five kicker ranks in base 13, with categories 0 high card .. 8 straight flush (a royal flush is
the top straight flush).

equity(hero, board, villain=None) enumerates every completion exactly. villain=None means a hand drawn uniformly from
the unseen cards (the "random hand"); a list of hands means uniform over those combos (a range), skipping combos
that clash with hero or board. Returns (p_win, p_tie, n_cases). Preflop vs a random hand is 1.7M boards x 1225
villain hands; equity() refuses that (use equity_mc).
"""
import itertools
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
SUIT_SYM = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
CAT_NAMES = ["high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads", "straight flush"]
P13 = 13 ** np.arange(4, -1, -1, dtype=np.int64)
AR13 = np.arange(13)


def card(s):
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def cards(s):
    s = s.replace(",", " ").split()
    if len(s) == 1 and len(s[0]) > 2:  # "AsKd"
        s = [s[0][i:i + 2] for i in range(0, len(s[0]), 2)]
    return [card(x) for x in s]


def name(c, sym=False):
    return RANKS[c // 4] + (SUIT_SYM[SUITS[c % 4]] if sym else SUITS[c % 4])


def names(cs, sym=False):
    return " ".join(name(c, sym) for c in cs)


def _topk(mask, k):
    """Ranks (descending) of the k highest True positions per row of an (N, 13) mask; -1 where missing."""
    idx = np.where(mask, AR13, -1)
    return -np.sort(-idx, axis=1)[:, :k]


def _straight_top(present):
    """(N, 13) bool rank-present -> top rank of the best straight, -1 if none (A-2-3-4-5 tops at the five)."""
    ext = np.concatenate([present[:, 12:13], present], axis=1)  # ext[:, i] = rank i - 1, ext[:, 0] = ace low
    top = np.full(len(present), -1)
    for j in range(10):  # window ext[j : j + 5] -> ranks j-1 .. j+3, top rank j + 3
        ok = ext[:, j:j + 5].all(axis=1)
        top = np.where(ok, j + 3, top)
    return top


def _pack(cat, ks):
    ks = np.where(ks < 0, 0, ks)
    if ks.shape[1] < 5:
        ks = np.concatenate([ks, np.zeros((len(ks), 5 - ks.shape[1]), ks.dtype)], axis=1)
    return cat.astype(np.int64) * 13 ** 5 + (ks.astype(np.int64) * P13).sum(axis=1)


def eval7(c):
    c = np.asarray(c)
    n = len(c)
    r, s = c // 4, c % 4
    cnt = np.zeros((n, 13), np.int8)
    for j in range(c.shape[1]):
        cnt[np.arange(n), r[:, j]] += 1
    present = cnt > 0
    out = np.zeros(n, np.int64)
    done = np.zeros(n, bool)

    # flush / straight flush
    scnt = np.stack([(s == k).sum(axis=1) for k in range(4)], axis=1)
    fs = scnt.argmax(axis=1)
    has_flush = scnt.max(axis=1) >= 5
    fmask = np.zeros((n, 13), bool)
    for j in range(c.shape[1]):
        m = s[:, j] == fs
        fmask[np.arange(n)[m], r[m, j]] = True
    fmask &= has_flush[:, None]
    sf = _straight_top(fmask)
    m = sf >= 0
    out[m] = _pack(np.full(m.sum(), 8), sf[m, None]); done |= m

    # quads
    q = _topk(cnt == 4, 1)[:, 0]
    m = (q >= 0) & ~done
    if m.any():
        kick = _topk(present & (AR13 != q[:, None]), 1)
        out[m] = _pack(np.full(m.sum(), 7), np.stack([q, kick[:, 0]], 1)[m]); done |= m

    # full house (trips + another trips or pair)
    t = _topk(cnt == 3, 2)
    p_fh = _topk((cnt >= 2) & (AR13 != t[:, 0:1]), 1)[:, 0]
    m = (t[:, 0] >= 0) & (p_fh >= 0) & ~done
    out[m] = _pack(np.full(m.sum(), 6), np.stack([t[:, 0], p_fh], 1)[m]); done |= m

    # flush
    m = has_flush & ~done
    out[m] = _pack(np.full(m.sum(), 5), _topk(fmask, 5)[m]); done |= m

    # straight
    st = _straight_top(present)
    m = (st >= 0) & ~done
    out[m] = _pack(np.full(m.sum(), 4), st[m, None]); done |= m

    # trips
    m = (t[:, 0] >= 0) & ~done
    if m.any():
        kick = _topk(present & (AR13 != t[:, 0:1]), 2)
        out[m] = _pack(np.full(m.sum(), 3), np.concatenate([t[:, 0:1], kick], 1)[m]); done |= m

    # two pair / pair
    pr = _topk(cnt == 2, 3)
    m = (pr[:, 1] >= 0) & ~done
    if m.any():
        kick = _topk(present & (AR13 != pr[:, 0:1]) & (AR13 != pr[:, 1:2]), 1)
        out[m] = _pack(np.full(m.sum(), 2), np.concatenate([pr[:, :2], kick], 1)[m]); done |= m
    m = (pr[:, 0] >= 0) & ~done
    if m.any():
        kick = _topk(present & (AR13 != pr[:, 0:1]), 3)
        out[m] = _pack(np.full(m.sum(), 1), np.concatenate([pr[:, :1], kick], 1)[m]); done |= m

    m = ~done
    out[m] = _pack(np.zeros(m.sum()), _topk(present, 5)[m])
    return out


def category(v):
    return np.asarray(v) // 13 ** 5


def equity(hero, board, villain=None, max_cases=5_000_000):
    """Exact P(hero wins), P(tie) at showdown, heads-up, all cards dealt out."""
    hero, board = list(hero), list(board)
    dead = set(hero) | set(board)
    assert len(dead) == len(hero) + len(board) == 2 + len(board), "duplicate cards"
    rest = [c for c in range(52) if c not in dead]
    if villain is None:
        villain = list(itertools.combinations(rest, 2))
    villain = [tuple(v) for v in villain if not (set(v) & dead)]
    need = 5 - len(board)
    n_cases = sum(_ncomb(len(rest) - 2, need) for _ in villain)
    if n_cases > max_cases:
        raise ValueError(f"{n_cases} cases; use equity_mc")
    win = tie = tot = 0
    # group by runout: evaluate hero once per runout, all compatible villain hands at once
    runouts = (np.array(list(itertools.combinations(rest, need)), dtype=np.int64) if need else np.zeros((1, 0), np.int64))
    vil = np.array(villain, dtype=np.int64)
    base = np.array(board, dtype=np.int64)
    hb = np.concatenate([np.tile(np.array(hero + board), (len(runouts), 1)), runouts], 1)
    hv = eval7(hb)
    # villain hands clash with a runout if they share a card
    for i in range(0, len(runouts), max(1, 200_000 // max(1, len(vil)))):
        ro = runouts[i:i + max(1, 200_000 // max(1, len(vil)))]
        R, V = len(ro), len(vil)
        clash = np.zeros((R, V), bool)
        for a in range(need):
            for b in range(2):
                clash |= ro[:, a:a + 1] == vil[None, :, b]
        rr, vv = np.nonzero(~clash)
        seven = np.concatenate([vil[vv], np.tile(base, (len(rr), 1)), ro[rr]], 1)
        vs = eval7(seven)
        hs = hv[i + rr]
        win += int((hs > vs).sum()); tie += int((hs == vs).sum()); tot += len(rr)
    return win / tot, tie / tot, tot


def equity_mc(hero, board, villain=None, n=200_000, seed=0):
    """Monte Carlo equity (for preflop vs a random hand); returns (p_win, p_tie, n). SE <= 0.5 / sqrt(n)."""
    rng = np.random.default_rng(seed)
    hero, board = list(hero), list(board)
    dead = set(hero) | set(board)
    rest = np.array([c for c in range(52) if c not in dead])
    need = 5 - len(board)
    if villain is None:
        draw = np.argsort(rng.random((n, len(rest))), axis=1)[:, :2 + need]
        d = rest[draw]
        vil, ro = d[:, :2], d[:, 2:]
    else:
        vlist = [v for v in villain if not (set(v) & dead)]
        vil = np.array(vlist)[rng.integers(len(vlist), size=n)]
        ro = np.zeros((n, need), np.int64)
        for i in range(n):  # rejection-free: shuffle remaining deck per sample
            left = np.setdiff1d(rest, vil[i], assume_unique=True)
            ro[i] = rng.choice(left, need, replace=False)
    b = np.tile(np.array(board, np.int64), (n, 1)) if board else np.zeros((n, 0), np.int64)
    hs = eval7(np.concatenate([np.tile(np.array(hero), (n, 1)), b, ro], 1))
    vs = eval7(np.concatenate([vil, b, ro], 1))
    return float((hs > vs).mean()), float((hs == vs).mean()), n


def deal_outcome(hero, board, villain, rng):
    """One sampled showdown: villain hand (from the spec) and runout. Returns (villain_hand, runout, +1/0/-1)."""
    dead = set(hero) | set(board)
    rest = [c for c in range(52) if c not in dead]
    if villain is None:
        v = tuple(rng.choice(rest, 2, replace=False).tolist())
    else:
        vl = [x for x in villain if not (set(x) & dead)]
        v = tuple(vl[rng.integers(len(vl))])
    left = [c for c in rest if c not in v]
    ro = rng.choice(left, 5 - len(board), replace=False).tolist()
    h = eval7(np.array([list(hero) + list(board) + ro]))[0]
    w = eval7(np.array([list(v) + list(board) + ro]))[0]
    return v, ro, int(np.sign(h - w))


def _ncomb(n, k):
    from math import comb
    return comb(n, k)
