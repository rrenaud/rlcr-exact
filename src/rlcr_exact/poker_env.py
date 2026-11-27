"""Poker showdown-equity task: a single-step, gym-style environment with an exact oracle.

A spot is heads-up hold'em, both players all-in, so the board is dealt out: hero's two cards, a flop/turn/river
board, and a villain who is either face up ("known") or a uniformly random unseen hand ("random"). The question is
whether hero wins the pot outright (a split does not count). Ground truth comes in two forms:
  p_win    exact probability from full enumeration (rlcr_exact.poker.equity): the oracle
  outcome  one sampled showdown (villain hand if random, then the runout): the label, as noisy as a real result
gold = "yes" if the sampled showdown is an outright win, else "no".

Prompt formats (a knob; both use RLCR's tabc_long system prompt and the same gold):
  rlcr   "Answer yes or no." RLCR-matched: confidence is P(your answer is right), so a calibrated model states
         max(p, 1 - p) and never goes below 0.5.
  claim  "Answer yes." The answer is fixed, so confidence = P(win) = equity, anywhere in [0, 1]. Needed for the
         calling band (prices 0.17-0.40) that docs/poker_site is about.

Env API (no RL-library dependency; wrap it for verifiers/TextArena later if needed):
  env = PokerEquityEnv(fmt="claim", seed=0); spot = env.reset(); text = spot["problem"]
  env.score(spot, completion) -> format_ok, answer, conf, correct (vs sampled gold), p_correct (oracle),
      brier (vs gold, RLCR's term), brier_expected (vs oracle: E over outcomes, no label noise), caller_expected
"""
import re
import numpy as np

from . import poker
from .grading import last_answer, normalize_answer

STREETS = {"flop": 3, "turn": 4, "river": 5}
CONF_RE = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>")


GUIDES = {
    None: "",
    # knob, off by default (RLCR's prompt has no length guidance): reasoning models otherwise try to enumerate every
    # runout and opponent hand in their head and run past 16k tokens (Inkling-2-Mini, 2026-11-27)
    "estimate": ("\nDo not enumerate every possible runout or opponent hand. Estimate the way a strong player does at "
                 "the table: identify the hands you beat and lose to, count outs, and use rules of thumb such as the rule "
                 "of 2 and 4. Keep your reasoning under 300 words. Inside <answer> write only the single word."),
    # v1 above hurt Inkling (2026-11-27, 200 spots): the word budget did not shorten native thinking (12.6k vs 13.5k
    # tokens), river hand reading got sloppier, and half its "no" answers switched to confidence = P(answer right).
    # v2: no word budget, and the claim semantics restated.
    "estimate2": ("\nYou do not need to enumerate every possible runout or opponent hand exactly; an estimate the way a "
                  "strong player makes it at the table (count outs, rules of thumb such as the rule of 2 and 4) is fine. "
                  "Check the hand rankings carefully. Whatever you think will happen, write yes in <answer>: your "
                  "confidence is your probability that you win outright, so a likely loss gets a low confidence."),
}


def problem_text(spot, fmt="claim", sym=False, guide=None):
    hero = poker.names(spot["hero"], sym); board = poker.names(spot["board"], sym)
    street = {3: "Flop", 4: "Turn (board so far)", 5: "River (full board)"}[len(spot["board"])]
    if spot["villain_kind"] == "known":
        opp = f"Your opponent's hole cards are face up: {poker.names(spot['villain'], sym)}."
    else:
        opp = "Your opponent's hole cards are unknown: any two of the unseen cards, all pairs equally likely."
    deal = "" if len(spot["board"]) == 5 else " The remaining board cards are dealt at random from the unseen cards."
    q = ("Do you win the pot outright at showdown? A split pot counts as not winning. "
         + ('Answer "yes". Your confidence is the probability that you win outright.' if fmt == "claim"
            else 'Answer "yes" or "no".'))
    return (f"Heads-up Texas Hold'em. Both players are all-in.{deal}\n"
            f"Your hole cards: {hero}.\n{street}: {board}.\n{opp}\n{q}{GUIDES[guide]}")


def sample_spot(rng, street=None, villain_kind=None):
    street = street or rng.choice(list(STREETS))
    villain_kind = villain_kind or rng.choice(["known", "random"])
    nb = STREETS[street]
    d = rng.choice(52, 2 + nb + 2, replace=False).tolist()
    spot = dict(street=str(street), villain_kind=str(villain_kind), hero=sorted(d[:2], reverse=True),
                board=d[2:2 + nb], villain=sorted(d[2 + nb:], reverse=True) if villain_kind == "known" else None)
    return spot


def solve(spot):
    vil = [tuple(spot["villain"])] if spot["villain"] else None
    pw, pt, n = poker.equity(spot["hero"], spot["board"], vil)
    return dict(spot, p_win=pw, p_tie=pt, n_cases=n)


def label(spot, rng):
    vil = [tuple(spot["villain"])] if spot["villain"] else None
    v, ro, o = poker.deal_outcome(spot["hero"], spot["board"], vil, rng)
    return dict(spot, dealt_villain=list(v), runout=ro, outcome=o, answer="yes" if o > 0 else "no")


def parse(completion):
    ans = normalize_answer(last_answer(completion))
    m = CONF_RE.findall(completion)
    conf = float(m[-1]) if m else None
    ok = ans in ("yes", "no") and conf is not None and 0 <= conf <= 1
    return ok, ans, conf


_CALLER = None


def score(spot, completion, fmt=None):
    global _CALLER
    if _CALLER is None:
        from .betting import Score
        _CALLER = Score("caller")
    fmt = fmt or spot.get("fmt", "claim")
    ok, ans, conf = parse(completion)
    c = conf if conf is not None and 0 <= conf <= 1 else 0.0  # RLCR: missing confidence = 0
    correct = float(ans == spot["answer"])
    p = spot["p_win"] if ans == "yes" else 1 - spot["p_win"]
    return dict(format_ok=ok, answer=ans, conf=conf, correct=correct, p_correct=p,
                brier=(c - correct) ** 2, brier_expected=p * (1 - c) ** 2 + (1 - p) * c ** 2,
                # caller score is about the equity statement: P(win) = c in the claim format (defined so), else c or 1 - c
                caller_expected=float(_CALLER.H(spot["p_win"]) + _CALLER.D(spot["p_win"], c if fmt == "claim" or ans == "yes" else 1 - c)))


class PokerEquityEnv:
    def __init__(self, fmt="claim", seed=0, streets=None, villain_kinds=None, sym=False, guide=None):
        self.fmt, self.sym, self.guide, self.rng = fmt, sym, guide, np.random.default_rng(seed)
        self.streets, self.kinds = streets, villain_kinds

    def reset(self):
        st = self.rng.choice(self.streets) if self.streets else None
        vk = self.rng.choice(self.kinds) if self.kinds else None
        spot = label(solve(sample_spot(self.rng, st, vk)), self.rng)
        spot["problem"] = problem_text(spot, self.fmt, self.sym, self.guide)
        return spot

    def score(self, spot, completion):
        return score(spot, completion, self.fmt)
