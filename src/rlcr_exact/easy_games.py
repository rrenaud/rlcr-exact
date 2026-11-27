"""Easier card games with exact equity: a difficulty ladder below hold'em (docs/poker_zeroshot.md "By difficulty").

Each game samples a state (hero cards, board cards, villain cards when face up), gives its exact P(hero wins
outright) and P(tie), and one sampled outcome. Card ints are the standard 0..51 of rlcr_exact.poker
(rank = c // 4, 0 = deuce .. 12 = ace; suit = c % 4).

  g1_high      one card each, higher rank wins, equal ranks split. p = 4 r / 51.
  g1b_high2    two cards each, compare highest rank, then the second; equal split. Exact over villain hands.
  g2_leduc     Leduc deck J Q K x {spades, hearts} (6 cards). One hole card each, one board card: matching the
               board's rank wins, else higher card, equal ranks split. Street "pre" (board unseen) or "board".
  g3_three     three-card poker vs a random three-card hand. Ranks: straight flush > trips > straight > flush >
               pair > high card; A-2-3 is the lowest straight. Exact by inclusion-exclusion over shared cards.
  g4_river     hold'em river, villain face up: who wins (p in {0, 1}). Hand reading only.
  g5_turn      hold'em turn, villain face up: p = winning rivers / 44. Counting outs.
  e2_akv88     A♠K♦ vs 8♥8♣ flop (rlcr_exact.poker_toy): exact over 990 runouts.

sample(game, rng) -> dict(game, hero, board, villain, p_win, p_tie, outcome (+1/0/-1), answer "yes"/"no").
problem_text(row) -> the LLM prompt (claim format: confidence = P(win)).
"""
import itertools
from math import comb

import numpy as np

from . import poker

GAMES = ["g1_high", "g1b_high2", "g2_leduc", "g3_three", "g4_river", "g5_turn", "e2_akv88"]
LEDUC = [r * 4 + s for r in (9, 10, 11) for s in (0, 1)]  # J Q K of spades, hearts


# ---------- three-card poker evaluator (vectorized) ----------
def eval3(c):
    c = np.asarray(c); r = np.sort(c // 4, axis=1)[:, ::-1]; s = c % 4
    flush = (s[:, 0] == s[:, 1]) & (s[:, 1] == s[:, 2])
    trips = (r[:, 0] == r[:, 2])
    pair = ~trips & ((r[:, 0] == r[:, 1]) | (r[:, 1] == r[:, 2]))
    distinct = (r[:, 0] != r[:, 1]) & (r[:, 1] != r[:, 2])
    run = distinct & (r[:, 0] - r[:, 2] == 2)
    wheel = (r[:, 0] == 12) & (r[:, 1] == 1) & (r[:, 2] == 0)  # A-3-2
    straight = run | wheel
    top = np.where(wheel, 1, r[:, 0])
    cat = np.select([straight & flush, trips, straight, flush, pair], [5, 4, 3, 2, 1], 0)
    pr = np.where(r[:, 0] == r[:, 1], r[:, 0], r[:, 1]); kick = np.where(r[:, 0] == r[:, 1], r[:, 2], r[:, 0])
    k = np.where(((cat == 5) | (cat == 3))[:, None], top[:, None] * np.array([[1, 0, 0]]),
                 np.where((cat == 1)[:, None], np.stack([pr, kick, np.zeros_like(pr)], 1), r))
    return cat.astype(np.int64) * 13 ** 3 + (k * np.array([169, 13, 1])).sum(1)


class _Three:
    """Exact equity of every hero three-card hand vs a uniform disjoint villain hand (C(49, 3) = 18,424)."""
    def __init__(self):
        self.hands = np.array(list(itertools.combinations(range(52), 3)))
        self.v = eval3(self.hands)
        self.all = np.sort(self.v)
        self.by1 = [np.sort(self.v[(self.hands == c).any(1)]) for c in range(52)]
        self.by2 = {}
        for a, b in itertools.combinations(range(52), 2):
            m = ((self.hands == a).any(1)) & ((self.hands == b).any(1))
            self.by2[(a, b)] = np.sort(self.v[m])
        n = comb(49, 3)
        lt = lambda arr, x: np.searchsorted(arr, x, "left"); le = lambda arr, x: np.searchsorted(arr, x, "right")
        self.p_win = np.zeros(len(self.hands)); self.p_tie = np.zeros(len(self.hands))
        for i, (h, x) in enumerate(zip(self.hands, self.v)):
            L = lt(self.all, x) - sum(lt(self.by1[c], x) for c in h) + sum(lt(self.by2[p], x) for p in itertools.combinations(h, 2))
            E = (le(self.all, x) - lt(self.all, x)) - sum(le(self.by1[c], x) - lt(self.by1[c], x) for c in h) \
                + sum(le(self.by2[p], x) - lt(self.by2[p], x) for p in itertools.combinations(h, 2)) - 1
            self.p_win[i] = L / n; self.p_tie[i] = E / n
        self.index = {tuple(h): i for i, h in enumerate(self.hands)}


_THREE = None


def _three():
    global _THREE
    if _THREE is None:
        _THREE = _Three()
    return _THREE


# ---------- samplers ----------
def _outcome_row(game, hero, board, villain, pw, pt, o, **extra):
    return dict(game=game, hero=[int(c) for c in hero], board=[int(c) for c in board],
                villain=[int(c) for c in villain] if villain is not None else None, p_win=float(pw), p_tie=float(pt),
                outcome=int(o), answer="yes" if o > 0 else "no", **extra)


def sample(game, rng):
    if game == "g1_high":
        h, v = rng.choice(52, 2, replace=False)
        return _outcome_row(game, [h], [], None, 4 * (h // 4) / 51, 3 / 51, np.sign(h // 4 - v // 4))
    if game == "g1b_high2":
        d = rng.choice(52, 4, replace=False); h = sorted(d[:2] // 4, reverse=True)
        rest = [c for c in range(52) if c not in d[:2]]
        V = np.array(list(itertools.combinations(rest, 2))) // 4; V = -np.sort(-V, axis=1)
        hv = h[0] * 13 + h[1]; vv = V[:, 0] * 13 + V[:, 1]
        vr = sorted(d[2:] // 4, reverse=True); o = np.sign(hv - (vr[0] * 13 + vr[1]))
        return _outcome_row(game, d[:2], [], None, (vv < hv).mean(), (vv == hv).mean(), o)
    if game == "g2_leduc":
        h, b, v = rng.choice(LEDUC, 3, replace=False)
        street = rng.choice(["pre", "board"])
        def win(hc, vc, bc):
            hp, vp = hc // 4 == bc // 4, vc // 4 == bc // 4
            return 1 if hp and not vp else -1 if vp and not hp else int(np.sign(hc // 4 - vc // 4))
        rest = [c for c in LEDUC if c != h]
        if street == "board":
            res = [win(h, x, b) for x in rest if x != b]
        else:
            res = [win(h, x, y) for x in rest for y in rest if y != x]
        res = np.array(res)
        return _outcome_row(game, [h], [b] if street == "board" else [], None, (res > 0).mean(), (res == 0).mean(),
                            win(h, v, b), street=str(street), dealt_board=[int(b)])
    if game == "g3_three":
        T = _three(); d = rng.choice(52, 6, replace=False)
        h = tuple(sorted(d[:3])); i = T.index[h]
        o = np.sign(eval3(np.array([d[:3]]))[0] - eval3(np.array([d[3:]]))[0])
        return _outcome_row(game, d[:3], [], None, T.p_win[i], T.p_tie[i], o, dealt_villain=[int(x) for x in d[3:]])
    if game in ("g4_river", "g5_turn"):
        nb = 5 if game == "g4_river" else 4
        d = rng.choice(52, 4 + nb, replace=False)
        hero, vil, board = list(d[:2]), list(d[2:4]), list(d[4:])
        pw, pt, _ = poker.equity(hero, board, [tuple(vil)])
        _, ro, o = poker.deal_outcome(hero, board, [tuple(vil)], rng)
        return _outcome_row(game, hero, board, vil, pw, pt, o, runout=[int(x) for x in ro])
    if game == "e2_akv88":
        from . import poker_toy
        hero, vil = poker.cards("As Kd"), poker.cards("8h 8c")
        D = poker_toy.enumerate_flops(hero, vil); i = rng.integers(len(D["flops"]))
        flop = list(D["flops"][i]); _, ro, o = poker.deal_outcome(hero, flop, [tuple(vil)], rng)
        return _outcome_row(game, hero, flop, vil, D["eq"][i], D["ties"][i] / poker_toy.N_RUNOUTS, o,
                            runout=[int(x) for x in ro])
    raise ValueError(game)


# ---------- prompts ----------
def _n(cs):
    return poker.names(cs, sym=True)


RULES = {
    "g1_high": "A card game with a standard 52-card deck. You and your opponent each get one card. The higher rank "
               "wins (2 is lowest, ace is highest; suits do not matter). Equal ranks split the pot.",
    "g1b_high2": "A card game with a standard 52-card deck. You and your opponent each get two cards. Compare your "
                 "higher cards; the higher rank wins. If they are equal, compare your lower cards. If both are equal, "
                 "the pot is split. 2 is lowest, ace is highest; suits do not matter.",
    "g2_leduc": "A card game with a six-card deck: jack, queen and king in spades and hearts. You and your opponent each "
                "get one card, and one shared board card is dealt. A player whose card has the same rank as the board "
                "card wins. Otherwise the higher rank wins (king > queen > jack). Equal ranks split the pot.",
    "g3_three": "Three-card poker with a standard 52-card deck. You and your opponent each get three cards. Hands rank, "
                "from best: straight flush, three of a kind, straight, flush, pair, high card (A-2-3 is the lowest "
                "straight, Q-K-A the highest). Ties within a rank are broken by the higher cards; equal hands split.",
    "g4_river": "Heads-up Texas Hold'em. All five board cards are out and both hands are face up. Each player makes "
                "the best five-card poker hand from their two cards and the board.",
    "g5_turn": "Heads-up Texas Hold'em. Both players are all-in with their hands face up. Four board cards are out "
               "and the last one (the river) is dealt at random from the 44 unseen cards.",
    "e2_akv88": "Heads-up Texas Hold'em. Both players are all-in with their hands face up. The flop is out and the "
                "turn and river are dealt at random from the unseen cards.",
}


def problem_text(row):
    g = row["game"]; lines = [RULES[g]]
    lines.append(f"Your card{'s' if len(row['hero']) > 1 else ''}: {_n(row['hero'])}.")
    if g == "g2_leduc":
        lines.append(f"Board card: {_n(row['board'])}." if row["board"] else "The board card has not been dealt yet.")
        lines.append("Your opponent's card is unknown: any of the other cards, equally likely.")
    elif row["villain"] is not None:
        lines.append(f"Board: {_n(row['board'])}.") if row["board"] else None
        lines.append(f"Your opponent's cards: {_n(row['villain'])}.")
    else:
        lines.append("Your opponent's cards are unknown: any of the unseen cards, all combinations equally likely.")
    lines.append('Do you win the pot outright? A split counts as not winning. Answer "yes". Your confidence is the '
                 "probability that you win outright, so a likely loss gets a low confidence.")
    return "\n".join(lines)


# ---------- features for small models ----------
ROLES = ("hero", "board", "villain")


def tokens(row, max_len=9):
    """(card ids, role ids) padded to max_len with card 52 / role 3 (pad)."""
    cs, rs = [], []
    for k, role in enumerate(ROLES):
        for c in (row[role] or []):
            cs.append(c); rs.append(k)
    cs += [52] * (max_len - len(cs)); rs += [3] * (max_len - len(rs))
    return cs[:max_len], rs[:max_len]


def onehot(rows):
    X = np.zeros((len(rows), 156), np.float32)
    for i, r in enumerate(rows):
        for k, role in enumerate(ROLES):
            for c in (r[role] or []):
                X[i, 52 * k + c] = 1
    return X
