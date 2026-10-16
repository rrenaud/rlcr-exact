"""Why does full-data training stall from RLVR-hotpot? Gradient signal vs noise at the untrained start.

For N training batches drawn as train_frozen.py draws them (4 problems x their rollouts), the LoRA gradient of each
loss term separately: the exact objective (-J), the answer-KL penalty (W=1, cached top-64 reference), and the
support floor (beta=0.03). For each term: the mean per-batch norm; the consistency ||mean g||^2 / mean ||g||^2
(1 = every batch points the same way, about 1/N = no shared direction); the mean pairwise cosine between batches;
and an Adam proxy, the median over coordinates of |mean g_i| / sqrt(mean g_i^2), roughly the fraction of the
learning rate Adam moves each coordinate along its consistent direction. Also the same for the sums J+KL and
J+KL+floor, and cos(mean g_J, mean g_KL). Fresh LoRA has B = 0, so all gradient is on B.

usage: grad_check.py [--batches 24] [--model M] (gpuq payload; TASK paths as in run_frozen.sh hotpot-rlvr)
Writes $GPUQ_RUN_DIR/grad_check.json.
"""
import argparse, collections, json, os, random, sys, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_frozen import LINEAR_MODULES, RefTopk, answer_kl, load_rows, make_scorer, make_token_bytes, stop_backward_at_first_adapter
from rlcr_exact.confidence import exact_objective, level_matrix, support_floor

C = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"
REF = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen/run-20261016T144115Z-73d8ea5b8b/ref_topk_train"

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=24)
ap.add_argument("--problems-per-step", type=int, default=4)
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--train", default=f"{C}/rollouts_hotpot_fresh_train.jsonl")
ap.add_argument("--ref-topk", default=REF)
ap.add_argument("--beta", type=float, default=0.03)
ap.add_argument("--adapter", default=None, help="start from a trained train_frozen.py adapter instead of a fresh LoRA")
ap.add_argument("--tag", default="start")
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()
random.seed(0); torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
values, score, _ = make_scorer("hotpot", tok)
train, _ = load_rows(a.train, make_token_bytes(tok), 4608)
groups = collections.defaultdict(list)
for r in train:
    groups[r.qid].append(r)
ids = random.sample(sorted(groups), a.batches * a.problems_per_step)
ref = RefTopk(a.ref_topk)

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=LINEAR_MODULES,
                                             layers_to_transform=list(range(14, model.config.num_hidden_layers))))
stop_backward_at_first_adapter(model)
model.train()
params = [p for p in model.parameters() if p.requires_grad]
values_gpu, levels = values.cuda(), level_matrix(values.cuda())


def flat_grad():
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in params]).cpu()
    for p in params:
        p.grad = None
    return g


grads = collections.defaultdict(list)  # term -> [batch grads]
t = time.time()
for b in range(a.batches):
    batch = [groups[g] for g in ids[b * a.problems_per_step: (b + 1) * a.problems_per_step]]
    n_rows = sum(len(g) for g in batch)
    for term in ("J", "Jcal", "KL", "floor"):
        for group in batch:  # one packed forward per group, as in training; recomputed per term to separate grads
            logps, states = score(model, [r.prefix for r in group], hidden=True)
            if term == "J":
                y = torch.tensor([r.correct for r in group], device=logps.device)
                loss = -exact_objective(logps, values_gpu, y[:, None]).sum()
            elif term == "Jcal":  # calibration only: expected Brier term under q renormalized over the list
                y = torch.tensor([r.correct for r in group], device=logps.device, dtype=values_gpu.dtype)
                q = (logps - torch.logsumexp(logps, -1, keepdim=True)).exp()
                loss = -(q * 0.5 * (1 - (y[:, None] - values_gpu[None]) ** 2)).sum()
            elif term == "KL":
                loss = answer_kl(model, states, group, ref).sum()
            else:
                loss = a.beta * support_floor(logps, levels).sum()
            (loss / n_rows).backward()
        grads[term].append(flat_grad())
    print(json.dumps(dict(batch=b, seconds=round(time.time() - t, 1))), flush=True)


def stats(gs):
    G = torch.stack(gs)  # [N, P]
    mean = G.mean(0)
    norms = G.norm(dim=1)
    unit = G / norms[:, None].clamp_min(1e-30)
    cos = unit @ unit.T
    n = len(gs)
    adam = (mean.abs() / G.pow(2).mean(0).sqrt().clamp_min(1e-30))
    nz = G.pow(2).mean(0) > 0
    return dict(mean_norm=norms.mean().item(), norm_of_mean=mean.norm().item(),
                consistency=(mean.norm() ** 2 / norms.pow(2).mean()).item(), chance_consistency=1 / n,
                mean_pairwise_cos=((cos.sum() - n) / (n * (n - 1))).item(),
                adam_fraction_median=adam[nz].median().item(), adam_fraction_mean=adam[nz].mean().item())


report = dict(tag=a.tag, adapter=a.adapter, batches=a.batches, rows_per_batch=sum(len(groups[g]) for g in ids) / a.batches, params=len(grads["J"][0]),
              beta=a.beta)
for term in ("J", "Jcal", "KL", "floor"):
    report[term] = stats(grads[term])
report["Jcal+KL"] = stats([j + k for j, k in zip(grads["Jcal"], grads["KL"])])
report["J+KL"] = stats([j + k for j, k in zip(grads["J"], grads["KL"])])
report["J+KL+floor"] = stats([j + k + f for j, k, f in zip(grads["J"], grads["KL"], grads["floor"])])
mj, mk, mf, mc = (torch.stack(grads[t]).mean(0) for t in ("J", "KL", "floor", "Jcal"))
cos = lambda u, v: (u @ v / (u.norm() * v.norm()).clamp_min(1e-30)).item()
report["cos_mean_J_KL"], report["cos_mean_J_floor"], report["cos_mean_J_Jcal"] = cos(mj, mk), cos(mj, mf), cos(mj, mc)
print(json.dumps(report, indent=1), flush=True)
with open(os.path.join(a.out, f"grad_check_{a.tag}.json"), "w") as f:
    json.dump(report, f, indent=1)
