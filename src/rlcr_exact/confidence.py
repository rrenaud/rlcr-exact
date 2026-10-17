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


def hotpot_candidates(tokenizer) -> Candidates:
    """RLCR-hotpot's confidence strings, scored through the token that closes the number.

    The released hotpot model writes `Ġ` `0` `.` `7` `Ġ</` ... (94% of untrained rollouts) or the same without
    the spaces, `0` `.` `7` `</` ... (6%), and 16% of its values have a second decimal 5 (0.75, 0.25, ...). The
    candidates are 0.0 .. 0.9, 0.05 .. 0.95, the spellings 0.00 .. 0.90 (worth 0.d) and 1.0, in both forms (62).
    On 16,000 fresh-train rollouts they cover all but 0.04% of sampled tails (e.g. "0.33"). Scoring the closing
    `</` or `Ġ</` separates "0.7" from "0.75"; what follows it (`confidence` `>` end of turn) is not scored.
    """
    tid = tokenizer.convert_tokens_to_ids
    values = [d / 10 for d in range(10)] + [d / 10 + 0.05 for d in range(10)] + [d / 10 for d in range(10)] + [1.0]
    texts = [f"0.{d}" for d in range(10)] + [f"0.{d}5" for d in range(10)] + [f"0.{d}0" for d in range(10)] + ["1.0"]
    strings, vals, seqs = [], [], []
    for lead, close in ((["Ġ"], "Ġ</"), ([], "</")):
        for text, v in zip(texts, values):
            strings.append("".join(lead).replace("Ġ", " ") + text + close.replace("Ġ", " "))
            vals.append(v)
            seqs.append([tid(t) for t in lead] + [tid(c) for c in text] + [tid(close)])
    return Candidates(strings, torch.tensor(vals), build_trie(seqs))


def hotpot_stop_candidates(tokenizer) -> Candidates:
    """Hotpot confidence strings scored through `confidence` `>` and an end token, so the stop after
    </confidence> is part of the candidate. RLVR-hotpot writes a junk token there (mostly `assistant`) instead of
    <|endoftext|> in ~22-28% of rollouts, which RLCR's format check fails (reward 0): those, like any unlisted
    completion, earn 0. Values: 1, 1.0, 1.00, 0.d, 0.d5, 0.d0 and 0.9d (41). RLCR pays any number in [0, 1]; RLVR
    writes 29 distinct strings, all in this set, incl. " 1 ", "1.00" and "0.98", which hotpot_candidates misses
    (all 0.dd would be 113 values and ~3x the trie). Each spaced (`Ġ` ... `Ġ</`) or not, x 2 end tokens.
    """
    tid = tokenizer.convert_tokens_to_ids
    texts = ["1", "1.0", "1.00"] + [f"0.{d}" for d in range(10)] + sorted(
        {f"0.{d}5" for d in range(10)} | {f"0.{d}0" for d in range(10)} | {f"0.9{d}" for d in range(10)})
    tail = [tid("confidence"), tid(">")]
    strings, vals, seqs = [], [], []
    for lead, close in ((["Ġ"], "Ġ</"), ([], "</")):
        for text in texts:
            for end in END_TOKENS:
                strings.append("".join(lead).replace("Ġ", " ") + text + close.replace("Ġ", " ") + "confidence>" + end)
                vals.append(float(text))
                seqs.append([tid(t) for t in lead] + [tid(c) for c in text] + [tid(close)] + tail + [tid(end)])
    return Candidates(strings, torch.tensor(vals), build_trie(seqs))


def rlcr_reward(values: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """RLCR's total reward for each well-formed candidate (malformed completions earn 0)."""
    y = correct.to(values.dtype)
    return 0.5 * (1 + y + 1 - (y - values) ** 2)


def level_matrix(values: torch.Tensor, levels: int = 11) -> torch.Tensor:
    """[levels, K] 0/1: which candidates spell each of 0.0, 0.1, ..., 1.0 (all spellings of a value count)."""
    grid = torch.linspace(0, 1, levels, dtype=values.dtype, device=values.device)
    return ((values[None, :] - grid[:, None]).abs() < 1e-6).to(values.dtype)


def support_floor(logps: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """[rows] cross-entropy from a uniform target over the levels to each row's level marginal,
    -mean_l log sum_{j in l} q_j. Its gradient on a level's logit is about q_l - 1/L, which does not vanish when q
    is peaked (the expected reward's gradient on a candidate is proportional to its probability), so it keeps
    some probability on every level. The uniform-mixture idea of ReFIne's confidence remap, as a regularizer."""
    marginal = torch.logsumexp(logps[:, None, :] + levels.clamp_min(1e-30).log()[None], dim=-1)  # [rows, L]
    return -marginal.mean(-1)


def exact_objective(logps: torch.Tensor, values: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Expected RLCR reward of the confidence completion: sum_j q_j R_j, with the unlisted mass earning 0."""
    return (logps.exp() * rlcr_reward(values, correct)).sum(-1)

