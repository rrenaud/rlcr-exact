import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlcr_exact import poker, poker_env


def test_sampled_labels_match_oracle():
    rng = np.random.default_rng(0)
    for vk, st in [("random", "turn"), ("known", "flop"), ("random", "river")]:
        s = poker_env.solve(poker_env.sample_spot(rng, st, vk))
        o = np.array([poker_env.label(s, rng)["outcome"] for _ in range(4000)])
        se = np.sqrt(s["p_win"] * (1 - s["p_win"]) / 4000) + 1e-3
        assert abs((o > 0).mean() - s["p_win"]) < 4 * se, (vk, st, (o > 0).mean(), s["p_win"])
        assert abs((o == 0).mean() - s["p_tie"]) < 4 * np.sqrt(s["p_tie"] / 4000) + 2e-3


def test_score_and_parse():
    env = poker_env.PokerEquityEnv(fmt="claim", seed=1, streets=["river"])
    spot = env.reset()
    assert "Do you win the pot outright" in spot["problem"] and 'Answer "yes".' in spot["problem"]
    good = f"<think>x</think> <answer>yes</answer> <analysis>y</analysis> <confidence>{spot['p_win']:.3f}</confidence>"
    r = env.score(spot, good)
    assert r["format_ok"] and r["answer"] == "yes" and abs(r["p_correct"] - spot["p_win"]) < 1e-12
    assert r["correct"] == float(spot["outcome"] > 0)
    # the oracle confidence minimizes expected Brier and expected caller loss
    for c in [0.0, 0.2, 0.5, 0.9]:
        bad = good.replace(f"{spot['p_win']:.3f}", f"{c}")
        assert env.score(spot, bad)["brier_expected"] >= r["brier_expected"] - 1e-6
        assert env.score(spot, bad)["caller_expected"] >= r["caller_expected"] - 1e-3
    assert not env.score(spot, "<answer>maybe</answer>")["format_ok"]


def test_problem_text_known():
    s = dict(hero=poker.cards("As Kd"), board=poker.cards("Qs Jd 4c"), villain=poker.cards("9h 9c"), villain_kind="known")
    t = poker_env.problem_text(s, "rlcr")
    assert "As Kd" in t and "Qs Jd 4c" in t and "9h 9c" in t and 'Answer "yes" or "no".' in t
