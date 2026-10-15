"""RLCR's confidence strings and rewards, for exact scoring over the trie of candidate completions.

RLCR (reward_fns.py, weights 0.5 each) pays format + accuracy + Brier: a well-formed completion with stated
confidence v and correctness y earns 0.5 * (1 + y + 1 - (y - v)^2); a malformed one earns 0 for all three.
After the prefix "...</analysis>\\n<confidence>", the released math model writes the number as digit tokens,
then "</confidence>" and the end of turn.
"""
from dataclasses import dataclass

import torch

from .trie_score import Trie, build_trie

END_TOKENS = ("<|endoftext|>", "<|im_end|>")  # the checkpoint's eos; the chat template's end of turn


def confidence_strings() -> list[str]:
    """0, 1, one- and two-decimal values in [0, 1]: 114 strings covering what the released models write."""
    out = ["0", "1", "1.0", "1.00"]
    out += [f"0.{d}" for d in range(10)] + [f"0.{d:02d}" for d in range(100)]
    return out


@dataclass
class Candidates:
    strings: list[str]
    values: torch.Tensor  # [num_candidates] stated confidence of each candidate
    trie: Trie


def make_candidates(tokenizer) -> Candidates:
    strings, values, seqs = [], [], []
    for s in confidence_strings():
        body = tokenizer(s + "</confidence>", add_special_tokens=False).input_ids
        for end in END_TOKENS:
            strings.append(s + "</confidence>" + end)
            values.append(float(s))
            seqs.append(body + [tokenizer.convert_tokens_to_ids(end)])
    return Candidates(strings, torch.tensor(values), build_trie(seqs))


def rlcr_reward(values: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """RLCR's total reward for each well-formed candidate (malformed completions earn 0)."""
    y = correct.to(values.dtype)
    return 0.5 * (1 + y + 1 - (y - values) ** 2)


def exact_objective(logps: torch.Tensor, values: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Expected RLCR reward of the confidence completion: sum_j q_j R_j, with the unlisted mass earning 0."""
    return (logps.exp() * rlcr_reward(values, correct)).sum(-1)

