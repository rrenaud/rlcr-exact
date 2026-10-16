"""grading.normalize_answer is RLCR's, and F1 behaves as the audits describe."""
import sys

from rlcr_exact.grading import f1_correct, normalize_answer, token_f1

sys.path.insert(0, "/home/rrenaud/rlcr_platt/RLCR")


def test_normalize_matches_rlcr():
    from reward_fns import normalize_answer as rlcr_normalize
    for s in ["The Beatles!", "  An  apple, a day ", "14 August 1947", "U.S.A.", "Leicester City F.C.", "6.8%", "THE the a"]:
        assert normalize_answer(s) == rlcr_normalize(s)


def test_f1():
    assert token_f1("14 August", "14 August 1947") == 0.8
    assert f1_correct("x <answer>Herbert Ross</answer>", "Herbert David Ross") == 1.0
    assert f1_correct("<answer>Lyon</answer>", "Paris") == 0.0
    assert f1_correct("no tags", "Paris") == 0.0
