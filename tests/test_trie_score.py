"""CPU check: trie scoring equals scoring each candidate as its own full sequence (tiny random Qwen2, fp32)."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from rlcr_exact.trie_score import build_trie, candidate_logps


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128, attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval()


def full_logps(model, prefix, cands):
    out = []
    for c in cands:
        ids = torch.tensor([prefix + c])
        logp = model(input_ids=ids).logits[0].float().log_softmax(-1)
        out.append(sum(logp[len(prefix) - 1 + i, t] for i, t in enumerate(c)))
    return torch.stack(out)


CANDS = [[5, 7, 9, 3, 1], [5, 7, 9, 4, 3, 1], [5, 7, 8, 3, 1], [6, 3, 1], [6, 7, 9, 3, 1], [5, 7, 9, 4, 4, 3, 1]]
PREFIX = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def test_matches_full_sequences():
    model = tiny_model()
    with torch.no_grad():
        got = candidate_logps(model, PREFIX, build_trie(CANDS))
        want = full_logps(model, PREFIX, CANDS)
    assert torch.allclose(got, want, atol=1e-4), (got, want)


def test_gradients_match_through_lora():
    model = get_peft_model(tiny_model(), LoraConfig(r=4, target_modules=["q_proj", "v_proj", "down_proj"], init_lora_weights=False))
    w = torch.randn(len(CANDS))
    (candidate_logps(model, PREFIX, build_trie(CANDS)) * w).sum().backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    (full_logps(model, PREFIX, CANDS) * w).sum().backward()
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(got[n], p.grad, atol=1e-5), n


def test_greedy_matches_generate():
    model = tiny_model(seed=3)
    with torch.no_grad():
        out = model.generate(torch.tensor([PREFIX]), max_new_tokens=6, do_sample=False)[0, len(PREFIX):].tolist()
    # The greedy continuation is found among distractors...
    _, k = candidate_logps(model, PREFIX, build_trie(CANDS + [out[:6]]), greedy=True)
    assert k == len(CANDS)
    # ...and when every candidate leaves it at the third token, greedy decoding exits the trie.
    off = (out[2] + 1) % 64
    _, k = candidate_logps(model, PREFIX, build_trie([out[:2] + [off, 1], out[:2] + [off, 2]]), greedy=True)
    assert k is None
