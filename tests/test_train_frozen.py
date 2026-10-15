"""CPU checks for train_frozen.py's metrics: ECE binning, completion log-probs, and evaluate's drift."""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_frozen import completion_logprob, ece, evaluate
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


def test_completion_logprob_matches_brute_force():
    model = tiny_model()
    prefix, start = [11, 12, 13, 14, 15, 16, 17], 3
    with torch.no_grad():
        logp = model(input_ids=torch.tensor([prefix])).logits[0].float().log_softmax(-1)
        want = sum(logp[p - 1, prefix[p]].item() for p in range(start, len(prefix)))
        assert abs(completion_logprob(model, prefix, start) - want) < 1e-4


def test_evaluate_reports_zero_drift_for_the_same_model():
    model = tiny_model()
    rows = [("a", [11, 12, 13, 14, 15], 1.0, 2), ("b", [21, 22, 23, 24], 0.0, 2), ("c", [31, 32, 33, 34, 35, 36], 1.0, 3)]
    score = lambda m, prefixes, greedy=False: level_logps(m, prefixes, IDS, greedy)
    metrics, base = evaluate(model, rows, LEVELS, score, 2, rows)
    again = evaluate(model, rows, LEVELS, score, 2, rows, base)[0]
    assert abs(again["answer_drift_mean"]) < 1e-6 and abs(again["answer_drift_max"]) < 1e-6
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
