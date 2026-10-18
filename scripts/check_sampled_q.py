"""Why do the one-pass and two-pass sampled paths diverge? At a fixed adapter, per training group:
q from the no-grad pass (old) vs q from the training forward (grad + hidden states, new), the draws each gives
from the same RNG seed, and run-to-run determinism of each forward and of the backward.

usage: check_sampled_q.py ADAPTER [--groups N]   (rlcr11 rows, RLVR-hotpot)
Writes $GPUQ_RUN_DIR/check_sampled_q.json.
"""
import argparse, collections, json, os, sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_frozen import STOP_SCORERS, load_rows, make_scorer, make_token_bytes, sample_draws, sampled_loss, stop_backward_at_first_adapter

C = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"
ap = argparse.ArgumentParser()
ap.add_argument("adapter")
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--groups", type=int, default=40)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
values, score, _ = make_scorer("rlcr11", tok)
values = values.to("cuda")
rows = load_rows(f"{C}/rollouts_hotpot_fresh_train.jsonl", make_token_bytes(tok), 4608, tail_ok="rlcr11" in STOP_SCORERS)[0]
groups = collections.defaultdict(list)
for r in rows:
    groups[r.qid].append(r)
groups = [groups[g] for g in sorted(groups)[: a.groups]]

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
stop_backward_at_first_adapter(model)
model.train()
params = [p for p in model.parameters() if p.requires_grad]

stats = collections.defaultdict(list)
for gi, group in enumerate(groups):
    prefixes = [r.prefix for r in group]
    correct = [r.correct for r in group]
    with torch.no_grad():
        q_ng = score(model, prefixes).exp().float()
        q_ng2 = score(model, prefixes).exp().float()
    logps, _ = score(model, prefixes, hidden=True)
    q_g = logps.detach().exp().float()
    logps2, _ = score(model, prefixes, hidden=True)
    stats["dq_nograd_vs_grad"].append((q_ng - q_g).abs().max().item())
    stats["dq_nograd_repeat"].append((q_ng - q_ng2).abs().max().item())
    stats["dq_grad_repeat"].append((q_g - logps2.detach().exp().float()).abs().max().item())
    torch.manual_seed(gi); d_ng = sample_draws(q_ng, correct, values, a.k)
    torch.manual_seed(gi); d_g = sample_draws(q_g, correct, values, a.k)
    same = [x == y for rn, rg in zip(d_ng, d_g) for x, y in zip(rn, rg)]
    stats["draws_identical_frac"].append(sum(same) / len(same))
    if gi < 8:  # backward determinism: the same loss twice
        base = sum(r for rd in d_g for _, r in rd) / (len(d_g) * a.k)
        grads = []
        for lp in (logps, logps2):
            g = torch.autograd.grad(sampled_loss(lp, d_g, base), params, retain_graph=False)
            grads.append(torch.cat([x.flatten().float() for x in g]))
        stats["grad_repeat_rel_diff"].append(((grads[0] - grads[1]).norm() / grads[0].norm()).item())
    del logps, logps2

report = {k: dict(mean=sum(v) / len(v), max=max(v), zero_frac=sum(x == 0 for x in v) / len(v)) for k, v in stats.items()}
report.update(adapter=a.adapter, groups=len(groups), k=a.k)
print(json.dumps(report, indent=1), flush=True)
json.dump(report, open(os.path.join(a.out, "check_sampled_q.json"), "w"), indent=1)
