"""Packed shared-prefix scoring vs per-row scoring on the real checkpoint (bf16): values, greedy, gradients, time.

usage: check_packed.py ROLLOUTS_JSONL [--groups 4] [--scorer hotpot] [--model NAME] [--dtype bfloat16|float32]
--model swaps in another Qwen2.5 (same tokenizer), e.g. Qwen/Qwen2.5-0.5B in float32, to separate bf16 noise
from a packing error on real sequence lengths. Writes $GPUQ_RUN_DIR/check_packed_<model>_<dtype>.json.
"""
import argparse, collections, json, os, sys, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_frozen import LINEAR_MODULES, load_rows, make_token_bytes, stop_backward_at_first_adapter
from rlcr_exact.confidence import hotpot_candidates, make_candidates
from rlcr_exact.trie_score import candidate_logps, packed_candidate_logps

ap = argparse.ArgumentParser()
ap.add_argument("rollouts")
ap.add_argument("--scorer", default="hotpot", choices=["hotpot", "trie"])
ap.add_argument("--groups", type=int, default=4)
ap.add_argument("--model", default=None)
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()
name = "mehuldamani/hotpot-v2-brier-7b-no-split" if a.scorer == "hotpot" else "mehuldamani/big-math-digits-v2-brier-base-tabc"
tok = AutoTokenizer.from_pretrained(name)  # the rows' tokenizer; --model must share it
name = a.model or name
trie = (hotpot_candidates(tok) if a.scorer == "hotpot" else make_candidates(tok)).trie
groups = collections.defaultdict(list)
for r in load_rows(a.rollouts, make_token_bytes(tok), 4608)[0]:
    groups[r.qid].append(r)
groups = [g for g in groups.values() if len(g) == 8][: a.groups]

model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, a.dtype), attn_implementation="sdpa", device_map={"": 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=LINEAR_MODULES, init_lora_weights=False,
                                         layers_to_transform=list(range(model.config.num_hidden_layers // 2, model.config.num_hidden_layers))))
stop_backward_at_first_adapter(model)


def timed(fn):
    torch.cuda.synchronize(); t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t


report = dict(model=name, dtype=a.dtype, groups=len(groups), rows=sum(len(g) for g in groups))
diffs, greedy_agree, t_row, t_pack = [], 0, 0.0, 0.0
model.eval()
with torch.no_grad():
    for g in groups:
        prefixes = [r.prefix for r in g]
        (row_out), dt = timed(lambda: [candidate_logps(model, p, trie, greedy=True) for p in prefixes])
        t_row += dt
        (pack_logps, pack_picks), dt = timed(lambda: packed_candidate_logps(model, prefixes, trie, greedy=True))
        t_pack += dt
        for (lp, pick), plp, ppick in zip(row_out, pack_logps, pack_picks):
            diffs.append((lp.exp() - plp.exp()).abs().max().item())
            greedy_agree += pick == ppick
report |= dict(prob_abs_diff_max=max(diffs), prob_abs_diff_mean=sum(diffs) / len(diffs),
               greedy_agree=greedy_agree / report["rows"], seconds_no_grad_per_row=dict(per_row=t_row / report["rows"],
                                                                                         packed=t_pack / report["rows"]))
# Gradients of the same loss through both paths, on one group.
model.train()
g = groups[0]
w = torch.randn(len(g), len(trie.leaf_paths), device="cuda")
grads, seconds = {}, {}
for path in ("per_row", "packed"):
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    def step():
        logps = (torch.stack([candidate_logps(model, r.prefix, trie) for r in g]) if path == "per_row"
                 else packed_candidate_logps(model, [r.prefix for r in g], trie))
        (logps.exp() * w).sum().backward()
    _, seconds[path] = timed(step)
    grads[path] = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.requires_grad])
    report[f"peak_gb_{path}"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
a_, b_ = grads["per_row"], grads["packed"]
report |= dict(grad_cosine=torch.nn.functional.cosine_similarity(a_, b_, dim=0).item(),
               grad_rel_diff=((a_ - b_).norm() / a_.norm()).item(),
               seconds_fwd_bwd_group=seconds)
print(json.dumps(report, indent=1), flush=True)
with open(os.path.join(a.out, f"check_packed_{name.split('/')[-1]}_{a.dtype}.json"), "w") as f:
    json.dump(report, f, indent=1)
