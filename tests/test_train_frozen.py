"""CPU checks for train_frozen.py's metrics: ECE binning, completion log-probs, and evaluate's drift."""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import train_frozen
from train_frozen import answer_kl, ece, evaluate
from rlcr_exact.digit_score import LEVELS, DigitIds, level_logps
from test_digit_score import IDS
from test_trie_score import tiny_model


def test_centred_ece_is_stable_at_levels():
    # Half the rows say exactly 0.8 and are all right; half say 0.8 - 1e-5 and are right 60% of the time.
    # Together they are calibrated at 0.8, but floor bins put the halves in different bins (7 and 8).
    c = np.r_[np.full(50, 0.8), np.full(50, 0.8 - 1e-5)]
    y = np.r_[np.ones(50), np.ones(30), np.zeros(20)]
    assert abs(ece(c, y) - 0.2) < 1e-3
    assert ece(c, y, centred=True) < 1e-4
    # On exact levels the two binnings agree.
    levels = np.repeat([0.2, 0.5, 0.8], 20)
    yy = np.r_[np.zeros(16), np.ones(4), np.zeros(10), np.ones(10), np.ones(17), np.zeros(3)]
    assert abs(ece(levels, yy) - ece(levels, yy, centred=True)) < 1e-9


ROWS = [("a", [11, 12, 13, 14, 15, 16], 1.0, 2), ("b", [21, 22, 23, 24], 0.0, 1), ("c", [31, 32, 33, 34, 35, 36, 37], 1.0, 3)]
score = lambda m, prefixes, greedy=False, hidden=False: level_logps(m, prefixes, IDS, greedy, hidden)


def lora(init_zero):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(tiny_model(), LoraConfig(r=4, lora_alpha=4, init_lora_weights=init_zero,
                                                   target_modules=train_frozen.LINEAR_MODULES))


def brute_force_kl(model, prefix, start):
    """KL(current || adapter off) at each position predicting prefix[start:], summed, from two full forwards."""
    ids = torch.tensor([prefix])
    cur = model(input_ids=ids).logits[0].float().log_softmax(-1)[start - 1: len(prefix) - 1]
    with model.disable_adapter():
        ref = model(input_ids=ids).logits[0].float().log_softmax(-1)[start - 1: len(prefix) - 1]
    return (cur.exp() * (cur - ref)).sum()


def test_answer_kl_matches_brute_force_and_is_zero_untrained():
    for init_zero in (True, False):
        model = lora(init_zero).eval()
        with torch.no_grad():
            got = answer_kl(model, score(model, [r[1] for r in ROWS], hidden=True)[1], ROWS)
            want = torch.stack([brute_force_kl(model, r[1], r[3]) for r in ROWS])
        assert torch.allclose(got, want, atol=1e-5), (got, want)
        assert (got.abs().max() < 1e-6) if init_zero else (got.min() > 0)


def test_chunked_answer_kl_has_the_same_gradients(monkeypatch):
    def grads(chunk):
        monkeypatch.setattr(train_frozen, "KL_CHUNK", chunk)
        model = lora(False).train()
        answer_kl(model, score(model, [r[1] for r in ROWS], hidden=True)[1], ROWS).sum().backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    whole, chunked = grads(512), grads(2)
    assert all(torch.allclose(whole[n], chunked[n], atol=1e-5) for n in whole)


def test_evaluate_reports_zero_answer_kl_untrained():
    model = lora(True)
    metrics = evaluate(model, ROWS, LEVELS, score, 2, ROWS)
    assert abs(metrics["answer_kl"]) < 1e-4 and abs(metrics["answer_kl_token"]) < 1e-4
    assert metrics["rows"] == 3 and "greedy_sd" in metrics


def test_backward_stops_at_first_adapter_with_same_grads():
    from peft import LoraConfig, get_peft_model
    from train_frozen import LINEAR_MODULES, stop_backward_at_first_adapter

    def run(fix):
        model = tiny_model()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, init_lora_weights=False,
                                                 target_modules=LINEAR_MODULES, layers_to_transform=[1]))
        if fix:
            stop_backward_at_first_adapter(model)
        model.train()
        first_layer_needs_grad = []
        model.get_base_model().model.layers[0].register_forward_hook(
            lambda mod, inp, out: first_layer_needs_grad.append((out[0] if isinstance(out, tuple) else out).requires_grad))
        logps = level_logps(model, [[11, 12, 13, 14, 15], [21, 22, 23]], IDS)
        (logps.exp() * torch.arange(12.0)).sum().backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}, first_layer_needs_grad[0]

    before, needs_before = run(False)
    after, needs_after = run(True)
    assert needs_before and not needs_after  # layer 0 has no adapter: no backward through it after the fix
    assert before.keys() == after.keys() and all(torch.allclose(before[n], after[n], atol=1e-6) for n in before)
