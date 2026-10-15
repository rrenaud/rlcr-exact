"""Exact log-probabilities of RLCR's confidence on an 11-level one-decimal scale, 0.0 .. 1.0.

After "...</analysis>\\n<confidence>" the released math model writes `0` `.` `d` (or `1` ...) as separate
tokens. Appending `0` `.` to each prefix and running one ordinary causal forward gives everything needed:

  log p(0.d) = log p(`0` | x) + log p(`.` | x 0) + log p(d | x 0 .)    d = 0..9
  log p(1.0) = log p(`1` | x)

The rest of the mass (bare "0", "0," ..., anything else) is malformed and earns 0, as in RLCR. What follows
the first decimal digit (`</confidence>` and the end of turn, or a second digit) is not scored: "0.85" counts
as 0.8, and "1" or "1.00" count as 1.0.
"""
from dataclasses import dataclass

import torch

LEVELS = torch.tensor([d / 10 for d in range(11)])  # stated confidence of each level: 0.0, 0.1, ..., 1.0


@dataclass
class DigitIds:
    zero: int
    one: int
    dot: int
    digits: list[int]  # token of "0" .. "9"

    @classmethod
    def from_tokenizer(cls, tok):
        ids = [tok.convert_tokens_to_ids(c) for c in "0123456789."]
        return cls(zero=ids[0], one=ids[1], dot=ids[10], digits=ids[:10])


def level_logps(model, prefixes: list[list[int]], ids: DigitIds, greedy: bool = False):
    """[batch, 11] log-probabilities of 0.0 .. 1.0 after each prefix (differentiable), from one forward pass.

    With greedy=True also returns, per row, the level that greedy decoding reaches, or None when it leaves the
    format: the argmax first token over the vocabulary must be `1` (level 1.0) or `0`; after `0` the argmax
    must be `.`, and after `0.` it must be a digit.
    """
    model = model.get_base_model() if hasattr(model, "get_base_model") else model  # unwrap PEFT
    device = next(model.parameters()).device
    lens = [len(p) for p in prefixes]
    width = max(lens) + 2
    # Right padding: real tokens keep positions 0..L+1, and the mask hides the padding from them.
    input_ids = torch.full((len(prefixes), width), ids.zero, dtype=torch.long)
    mask = torch.zeros(len(prefixes), width, dtype=torch.long)
    for b, p in enumerate(prefixes):
        input_ids[b, : len(p) + 2] = torch.tensor(p + [ids.zero, ids.dot])
        mask[b, : len(p) + 2] = 1
    hidden = model.model(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state
    # Logits only where a confidence token is predicted: after the prefix, after `0`, after `0.`.
    rows = torch.arange(len(prefixes), device=device)[:, None]
    cols = torch.tensor(lens, device=device)[:, None] - 1 + torch.arange(3, device=device)
    logp = model.lm_head(hidden[rows, cols]).float().log_softmax(-1)  # [batch, 3, vocab]
    zero_dot = logp[:, 0, ids.zero] + logp[:, 1, ids.dot]
    out = torch.cat([zero_dot[:, None] + logp[:, 2, ids.digits], logp[:, 0, ids.one, None]], dim=-1)
    if not greedy:
        return out
    argmax = logp.argmax(-1).tolist()
    digit_level = {t: d for d, t in enumerate(ids.digits)}
    picks = []
    for first, second, third in argmax:
        if first == ids.one:
            picks.append(10)
        elif first == ids.zero and second == ids.dot:
            picks.append(digit_level.get(third))
        else:
            picks.append(None)
    return out, picks
