import os

import pytest
import torch

from rlcr_exact.confidence import level_matrix, rlcr11_candidates


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
