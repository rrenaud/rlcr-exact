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


def test_tail_only_failure_and_em():
    from rlcr_exact.grading import em_correct, tail_only_failure
    good = "<think> x </think> <answer> The Beatles </answer> <analysis> y </analysis> <confidence> 0.7 </confidence>"
    assert tail_only_failure(good + "assistant") and tail_only_failure(good)
    assert not tail_only_failure(good.replace("<analysis>", ""))  # a missing tag is not a tail fault
    assert not tail_only_failure(good.replace("0.7", "1.3") + "junk")  # out of range
    assert not tail_only_failure("no tags at all")
    assert em_correct(good + "assistant", "beatles") == 1.0 and em_correct(good, "Rolling Stones") == 0.0
