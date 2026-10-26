import os

import pytest
import torch

from rlcr_exact.confidence import level_matrix, rlcr11_candidates, rlcr11_math_candidates


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("mehuldamani/hotpot-v2-correctness-7b")
    except OSError:
        pytest.skip("hotpot tokenizer not cached")


def test_rlcr11_is_the_levels_in_the_models_spelling(tok):
    c = rlcr11_candidates(tok)
    assert len(c.strings) == 11 and len(c.trie.tokens) == 60
    assert torch.equal(level_matrix(c.values), torch.eye(11))  # every candidate is exactly one support-floor level
    for s, path in zip(c.strings, c.trie.leaf_paths):
        ids = [c.trie.tokens[n] for n in path]
        # the trie path is how the tokenizer spells the text after "<confidence>", ending in the checkpoint's eos
        assert ids == tok(s, add_special_tokens=False).input_ids
        assert ids[-1] == tok.convert_tokens_to_ids("<|endoftext|>") == 151643


def test_rlcr11_math_spells_each_level_twice():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("mehuldamani/big-math-digits-v2-correctness")
    except OSError:
        pytest.skip("math tokenizer not cached")
    c = rlcr11_math_candidates(tok)
    assert len(c.strings) == 22 and len(c.trie.tokens) == 93
    assert torch.equal(level_matrix(c.values).sum(1), torch.full((11,), 2.0))  # two closings per level
    for s, path in zip(c.strings, c.trie.leaf_paths):
        ids = [c.trie.tokens[n] for n in path]
        assert ids == tok(s, add_special_tokens=False).input_ids and ids[-1] == 151643
        assert tok.decode(ids[:1]) == ">\n"  # the token cut_point drops after "<confidence"


def test_correctness_target_is_the_binned_beta_posterior():
    from scipy.stats import beta
    from rlcr_exact.confidence import correctness_target
    k, n = torch.tensor([0, 3, 8]), torch.tensor([8, 8, 8])
    t = correctness_target(k, n)
    assert t.shape == (3, 11) and torch.allclose(t.sum(-1), torch.ones(3))
    edges = [0] + [(i + 0.5) / 10 for i in range(10)] + [1]
    for row, kk in zip(t, k.tolist()):
        want = [beta.cdf(edges[i + 1], kk + 1, 8 - kk + 1) - beta.cdf(edges[i], kk + 1, 8 - kk + 1) for i in range(11)]
        assert torch.allclose(row, torch.tensor(want, dtype=torch.float32), atol=1e-6)
    assert t[2, 10] < 0.6 and t[2, 7] > 0.05  # all 8 right still leaves real mass below 0.85: the target is wide


def test_rps_is_proper_and_ignores_unlisted_mass():
    from rlcr_exact.confidence import correctness_target, rps_objective
    torch.manual_seed(0)
    levels = level_matrix(torch.linspace(0, 1, 11))
    target = correctness_target(torch.tensor([5]), torch.tensor([8]))
    rps_t, mass = rps_objective(target.log(), levels, target)
    assert rps_t.abs().max() < 1e-12 and (mass - 1).abs().max() < 1e-6
    for _ in range(20):  # any other forecast scores worse
        q = torch.randn(1, 11).softmax(-1)
        assert rps_objective(q.log(), levels, target)[0] > 1e-6
    # half the mass off the list: the same RPS (renormalised), mass 0.5
    rps_h, mass_h = rps_objective((target * 0.5).log(), levels, target)
    assert rps_h.abs().max() < 1e-12 and torch.allclose(mass_h, torch.tensor([0.5]))
    # gradient descent on free logits reaches the target, a spread distribution rather than a point mass
    logits = torch.zeros(1, 11, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(3000):
        opt.zero_grad(); rps_objective(logits.log_softmax(-1), levels, target)[0].sum().backward(); opt.step()
    assert (logits.softmax(-1) - target).abs().max() < 0.01


def test_format_loss_softens_toward_linear_at_low_mass():
    from rlcr_exact.confidence import format_loss
    mass = torch.tensor([0.19, 0.99], requires_grad=True)
    for eps, want in ((0.0, 1 / 0.19), (0.5, 1 / 0.69)):  # gradient relative to the linear term's (-1)
        g, = torch.autograd.grad(format_loss(mass, eps).sum(), mass)
        assert torch.allclose(-g[0], torch.tensor(want), rtol=1e-5)
    assert torch.isfinite(format_loss(torch.zeros(1))).all()


def test_level_ce_is_proper_and_counts_unlisted_mass():
    from rlcr_exact.confidence import correctness_target, level_ce, support_floor
    levels = level_matrix(torch.linspace(0, 1, 11))
    target = correctness_target(torch.tensor([6]), torch.tensor([8]))
    best = level_ce(target.log(), levels, target)
    torch.manual_seed(0)
    for _ in range(20):
        assert level_ce(torch.randn(1, 11).log_softmax(-1), levels, target) > best
    assert torch.allclose(level_ce((target * 0.5).log(), levels, target), best + torch.log(torch.tensor(2.0)))
    q = torch.randn(3, 11).log_softmax(-1)  # the floor is the uniform-target case
    assert torch.allclose(support_floor(q, levels), level_ce(q, levels, torch.full((1, 11), 1 / 11)))
