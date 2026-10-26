"""Answer grading next to RLCR's exact match: SQuAD/HotpotQA token F1 over RLCR's normalization.

RLCR trains and evaluates HotpotQA with exact match after normalize_answer (reward_fns.py). Stage A trains on
those labels; evaluations also report metrics against F1 >= 0.5 (the conf-weighted-vote grading audit's
lenient tier), because exact match rejects ~10% of acceptable answers ("14 August" for "14 August 1947").
"""
import collections
import re
import string


def normalize_answer(s):
    """RLCR's normalize_answer, verbatim: lowercase, drop punctuation and articles, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1(prediction, gold):
    a, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = sum((collections.Counter(a) & collections.Counter(g)).values())
    return 0.0 if not common else 2 * common / (len(a) + len(g))


def last_answer(text):
    found = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.MULTILINE)
    return found[-1] if found else ""


def f1_correct(text, gold, threshold=0.5):
    """1.0 if the last <answer> has token F1 >= threshold against the gold answer."""
    return float(token_f1(last_answer(text), gold) >= threshold)


def agreement(texts):
    """[(k, m)] per rollout: k of the m = len(texts) - 1 other rollouts whose last <answer> equals its own after RLCR's
    normalize_answer (an empty answer agrees with nothing). The self-consistency count of each answer, no gold label."""
    answers = [normalize_answer(last_answer(t)) for t in texts]
    counts = collections.Counter(a for a in answers if a)
    return [((counts[a] - 1) if a else 0, len(texts) - 1) for a in answers]


def em_correct(text, gold):
    """1.0 if the last <answer> matches the gold answer exactly after RLCR's normalize_answer (format not checked)."""
    return float(normalize_answer(last_answer(text)) == normalize_answer(gold))


TABC = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>(.*?)</confidence>\s*\Z"


def tail_only_failure(text):
    """True if the completion is well-formed under RLCR's tabc check up to and including the last </confidence>,
    so its only fault is what follows (RLVR-hotpot often writes a junk token there instead of end of text)."""
    end = text.rfind("</confidence>")
    if end < 0:
        return False
    m = re.match(TABC, text[: end + len("</confidence>")], re.DOTALL | re.MULTILINE)
    if not m:
        return False
    try:
        return 0 <= float(m.group(1)) <= 1
    except ValueError:
        return False
