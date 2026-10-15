"""CPU check: 11-level digit scoring equals full-sequence scoring of each level (tiny random Qwen2, fp32)."""
import torch
from peft import LoraConfig, get_peft_model

from rlcr_exact.digit_score import DigitIds, level_logps
from test_trie_score import tiny_model

IDS = DigitIds(zero=5, one=6, dot=7, digits=[5, 6, 30, 31, 32, 33, 34, 35, 36, 37])
PREFIXES = [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [40, 41, 42], [21, 22, 23, 24, 25, 26]]


def full_logps(model, prefix, ids=IDS):
    """Score each level as its own sequence: prefix + 0 . d for 0.d, prefix + 1 for 1.0."""
    out = []
    for c in [[ids.zero, ids.dot, d] for d in ids.digits] + [[ids.one]]:
        logp = model(input_ids=torch.tensor([prefix + c])).logits[0].float().log_softmax(-1)
        out.append(sum(logp[len(prefix) - 1 + i, t] for i, t in enumerate(c)))
    return torch.stack(out)


def test_matches_full_sequences():
    model = tiny_model()
    with torch.no_grad():
        for p in PREFIXES:
            got = level_logps(model, [p], IDS)[0]
            assert torch.allclose(got, full_logps(model, p), atol=1e-4), (got, full_logps(model, p))
        # The 11 levels are disjoint events, so their mass is at most 1.
        assert level_logps(model, PREFIXES, IDS).exp().sum(-1).max() <= 1 + 1e-5


def test_batch_with_padding_matches_single_rows():
    model = tiny_model(seed=1)
    with torch.no_grad():
        batched = level_logps(model, PREFIXES, IDS)
        single = torch.cat([level_logps(model, [p], IDS) for p in PREFIXES])
    assert torch.allclose(batched, single, atol=1e-5), (batched - single).abs().max()


def test_gradients_match_through_lora():
    model = get_peft_model(tiny_model(), LoraConfig(r=4, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False))
    w = torch.randn(len(PREFIXES), 11)
    (level_logps(model, PREFIXES, IDS) * w).sum().backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    (torch.stack([full_logps(model, p) for p in PREFIXES]) * w).sum().backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(got[n], p.grad, atol=1e-5), n


def test_greedy_matches_generate():
    model = tiny_model(seed=0)
    for prefix in PREFIXES:
        with torch.no_grad():
            out = model.generate(torch.tensor([prefix]), max_new_tokens=3, do_sample=False)[0, len(prefix):].tolist()
        a, b, c = out
        assert len(set(out)) == 3, out  # this seed's greedy paths have distinct tokens, so the cases below are clean
        free = [t for t in range(50, 64) if t not in out]
        digits = [a, free[0]] + free[1:9]
        # Greedy writes "0" "." then its third token, here digit 7: the level 0.7.
        with_c = digits[:7] + [c] + digits[8:]
        assert level_logps(model, [prefix], DigitIds(a, free[0], b, with_c), greedy=True)[1] == [7]
        # Greedy's first token as "1" is the level 1.0.
        assert level_logps(model, [prefix], DigitIds(free[0], a, b, with_c), greedy=True)[1] == [10]
        # Off format: the first token is neither, the second is not ".", or the third is not a digit.
        assert level_logps(model, [prefix], DigitIds(free[0], free[1], b, with_c), greedy=True)[1] == [None]
        assert level_logps(model, [prefix], DigitIds(a, free[0], free[1], with_c), greedy=True)[1] == [None]
        assert level_logps(model, [prefix], DigitIds(a, free[0], b, digits), greedy=True)[1] == [None]
