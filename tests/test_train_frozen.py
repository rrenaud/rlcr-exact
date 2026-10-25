"""CPU checks for train_frozen.py's metrics: ECE binning, completion log-probs, and evaluate's drift."""
import os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import train_frozen
from train_frozen import Row, RefTopk, answer_kl, ece, evaluate
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


ROWS = [Row("a", [11, 12, 13, 14, 15, 16], 1.0, 2, 0, 1.0), Row("b", [21, 22, 23, 24], 0.0, 1, 0, 1.0),
        Row("a", [31, 32, 33, 34, 35, 36, 37], 1.0, 3, 1, 0.0)]
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


def test_cached_topk_kl(tmp_path):
    from ref_topk import write_cache
    kls = {}
    for k in (64, 4):  # the tiny vocabulary has 64 tokens: k = 64 is the whole distribution
        with torch.no_grad():
            write_cache(tiny_model().eval(), ROWS, tmp_path / f"k{k}", k, {})
        cache = RefTopk(tmp_path / f"k{k}")
        for init_zero in (True, False):
            model = lora(init_zero).eval()
            with torch.no_grad():
                states = score(model, [r.prefix for r in ROWS], hidden=True)[1]
                kls[k, init_zero] = answer_kl(model, states, ROWS, cache), answer_kl(model, states, ROWS)
    cached, live = kls[64, False]
    assert torch.allclose(cached, live, atol=1e-3), (cached, live)  # fp16 cache, otherwise exact
    cached, live = kls[4, False]
    assert (cached <= live + 1e-4).all() and (cached > 0).all()  # coarse-graining never adds KL
    for k in (64, 4):
        assert kls[k, True][0].abs().max() < 1e-3  # untrained adapter: zero up to fp16 rounding


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


def test_trie_scorer_hidden_states_feed_answer_kl():
    from rlcr_exact.trie_score import build_trie, candidate_logps
    trie = build_trie([[5, 7, 30], [5, 7, 31, 8], [6, 8]])
    model = lora(False).eval()
    with torch.no_grad():
        states = [candidate_logps(model, r.prefix, trie, hidden=True)[1] for r in ROWS]
        for r, h in zip(ROWS, states):  # the prefix's rows equal a plain forward over the prefix
            plain = model.get_base_model().model(input_ids=torch.tensor([r.prefix])).last_hidden_state[0]
            assert torch.allclose(h[: len(r.prefix)], plain, atol=1e-5)
        trie_kl = answer_kl(model, states, ROWS)
        digit_kl = answer_kl(model, score(model, [r.prefix for r in ROWS], hidden=True)[1], ROWS)
    assert torch.allclose(trie_kl, digit_kl, atol=1e-5)


def test_support_floor_gradient_survives_peaked_q():
    import torch
    from rlcr_exact.confidence import level_matrix, support_floor
    values = torch.tensor([0.0, 0.1, 0.15, 0.5, 0.50, 1.0])
    levels = level_matrix(values)
    assert levels.shape == (11, 6) and levels[5].tolist() == [0, 0, 0, 1, 1, 0] and levels.sum() == 5
    logits = torch.tensor([[-20.0, -20.0, -20.0, -20.0, -20.0, 0.0]], requires_grad=True)
    support_floor(logits.log_softmax(-1), levels).sum().backward()
    assert logits.grad[0, 0] < -0.05  # pulls up an almost-empty level
    assert logits.grad[0, 5] > 0.05   # pushes down the peak


def test_adam_fraction_signal_vs_noise():
    import torch
    from train_frozen import adam_fraction
    for noise, lo, hi in ((0.0, 0.99, 1.01), (100.0, 0.0, 0.3)):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(10000))
        opt = torch.optim.Adam([p], lr=1e-3)
        for _ in range(50):
            p.grad = torch.ones_like(p) + noise * torch.randn_like(p)
            opt.step()
        assert lo <= adam_fraction(opt) <= hi, (noise, adam_fraction(opt))


def test_grad_report_parts():
    import torch
    from train_frozen import grad_report
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])
    r = grad_report({"objective": a, "kl": b}, {"objective": a})
    assert r["norm_objective"] == 1.0 and r["norm_kl"] == 2.0 and abs(r["norm_total"] - 5 ** 0.5) < 1e-4
    assert r["cos_kl_objective"] == 0.0 and r["cos_prev_objective"] == 1.0 and "cos_prev_kl" not in r


def test_sampled_k_draws_converge_to_exact_gradient():
    from rlcr_exact.confidence import exact_objective
    torch.manual_seed(0)
    values = torch.tensor([0.1, 0.5, 0.9, 1.0])
    correct = [1.0, 0.0, 1.0]
    logits = torch.randn(3, 5, requires_grad=True)  # last column = "other" (off-list) mass
    logps = logits.log_softmax(-1)[:, :4]
    (-exact_objective(logps, values, torch.tensor(correct)[:, None]).sum()).backward()
    g_exact, logits.grad = logits.grad.clone(), None

    def sampled_grad(k):
        logps = logits.log_softmax(-1)[:, :4]
        draws = train_frozen.sample_draws(logps.exp().detach(), correct, values, k)
        assert all(len(d) == k for d in draws)
        baseline = np.mean([r for d in draws for _, r in d])
        train_frozen.sampled_loss(logps, draws, baseline).backward()
        g, logits.grad = logits.grad.clone(), None
        return g

    err = lambda k: (sampled_grad(k) - g_exact).norm() / g_exact.norm()
    assert err(20000) < 0.05 and err(1) > err(20000)


def test_floor_weight_anneals_linearly():
    from types import SimpleNamespace
    a = SimpleNamespace(support_floor=0.05, floor_decay_from=None, steps=500)
    assert train_frozen.floor_weight(a, 400) == 0.05
    a.floor_decay_from = 0.6
    w = [train_frozen.floor_weight(a, s) for s in (1, 300, 400, 500)]
    assert w[0] == w[1] == 0.05 and abs(w[2] - 0.025) < 1e-12 and w[3] == 0


def test_adapter_and_optimizer_round_trip(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    model = get_peft_model(tiny_model(), LoraConfig(r=2, lora_alpha=2, target_modules=train_frozen.LINEAR_MODULES))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(2):
        model(torch.tensor([[1, 2, 3]])).logits.sum().backward()
        opt.step(); opt.zero_grad()
    model.save_pretrained(tmp_path); torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
    again = PeftModel.from_pretrained(tiny_model(), tmp_path, is_trainable=True)
    params2 = [p for p in again.parameters() if p.requires_grad]
    assert len(params2) == len(params) and all(torch.equal(a, b) for a, b in zip(params, params2))
    opt2 = torch.optim.Adam(params2, lr=1e-2)
    opt2.load_state_dict(torch.load(tmp_path / "optimizer.pt"))
    assert all(torch.equal(opt.state[a]["exp_avg_sq"], opt2.state[b]["exp_avg_sq"]) for a, b in zip(params, params2))


def test_token_parts_caps_rows_and_tokens():
    from train_frozen import token_parts
    assert token_parts([], 8, 100) == []
    assert token_parts([10] * 5, 2, 100) == [(0, 2), (2, 4), (4, 5)]
    assert token_parts([60, 50, 1536, 10, 10], 8, 100) == [(0, 1), (1, 2), (2, 3), (3, 5)]  # a long row alone
