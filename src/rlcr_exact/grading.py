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
