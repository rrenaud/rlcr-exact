"""Continued training of RLCR's confidence on cached rollouts: exact expected reward vs sampled policy gradient.

Each training row is a cached rollout of the released checkpoint, cut after "<confidence>", with RLCR's
correctness label for its answer. The answer and analysis are fixed; only the distribution of the
confidence completion (114 strings x 2 end tokens, scored exactly through a trie) is trained.

  exact:   maximize sum_j q_j R_j (the unlisted mass earns 0, as a malformed completion does in RLCR)
  sampled: GRPO on the confidence tokens alone: sample one completion per row from q (or "other"),
           advantage = reward minus its problem group's mean (no std division, as in RLCR), loss -A log q

usage: train_frozen.py --train ROLLOUTS --dev ROLLOUTS [--test ROLLOUTS] --objective exact|sampled ...
"""
import argparse, collections, json, math, os, random, sys, time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlcr_exact.confidence import exact_objective, make_candidates, rlcr_reward
from rlcr_exact.trie_score import candidate_logps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rlcr-reward-conf-weighted-tt-scaling"))
from logprob_spans import cut_point, make_token_bytes


def load_rows(path, token_bytes, max_prefix):
    """Well-formed rollouts as (problem id, prefix token ids, correctness); counts what was dropped.

    The prefix is the sampled token ids themselves (prompt_ids + completion_ids), cut through the <confidence>
    tag that opens the final confidence, so training scores exactly the tokens the policy produced.
    """
    rows, dropped = [], collections.Counter()
    for line in open(path):
        q = json.loads(line)
        for text, ids, ok, correct in zip(q["outputs"], q["completion_ids"], q["format_ok"], q["correct"]):
            cut = cut_point(text, ids, "confidence", token_bytes) if ok else None
            if cut is None:
                dropped["format"] += 1; continue
            prefix = q["prompt_ids"] + ids[:cut]
            if len(prefix) > max_prefix:
                dropped["long"] += 1; continue
            rows.append((q["id"], prefix, float(correct)))
    return rows, dict(dropped)


def ece(c, y, bins=10):
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in range(bins) if (b == k).any()))


@torch.no_grad()
def evaluate(model, rows, cands):
    model.eval()
    greedy, expected, mass, y = [], [], [], []
    for _, ids, correct in rows:
        logps, k = candidate_logps(model, ids, cands.trie, greedy=True)
        q = logps.exp().cpu()
        greedy.append(np.nan if k is None else cands.values[k].item())
        mass.append(q.sum().item())
        expected.append((q * cands.values).sum().item() / max(q.sum().item(), 1e-9))
        y.append(correct)
    model.train()
    g, e, y = np.array(greedy), np.array(expected), np.array(y)
    ok = ~np.isnan(g)
    out = {"rows": len(y), "accuracy": y.mean(), "mass": float(np.mean(mass)), "greedy_off_list": float(1 - ok.mean())}
    for name, c, yy in (("greedy", g[ok], y[ok]), ("expected", e, y)):
        out |= {f"{name}_mean": c.mean(), f"{name}_brier": np.mean((c - yy) ** 2), f"{name}_ece": ece(c, yy),
                f"{name}_auroc": roc_auc_score(yy, c) if len(set(yy)) > 1 else float("nan")}
    return {k: round(float(v), 4) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-brier-base-tabc")
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test")
    ap.add_argument("--objective", choices=["exact", "sampled"], required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-first-layer", type=int, default=0, help="adapt only layers >= this one")
    ap.add_argument("--problems-per-step", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--max-prefix", type=int, default=4608)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    log_path = os.path.join(args.out, f"train_{args.objective}_s{args.seed}.jsonl")

    def log(**rec):
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    cands = make_candidates(tok)
    token_bytes = make_token_bytes(tok)
    train, dropped = load_rows(args.train, token_bytes, args.max_prefix)
    dev, dev_dropped = load_rows(args.dev, token_bytes, args.max_prefix)
    dev = dev[: args.dev_limit]
    test = load_rows(args.test, token_bytes, args.max_prefix)[0] if args.test else None
    groups = collections.defaultdict(list)
    for row in train:
        groups[row[0]].append(row)
    group_ids = sorted(groups)
    log(event="data", train_rows=len(train), train_problems=len(groups), dropped=dropped, dev_rows=len(dev),
        dev_dropped=dev_dropped, test_rows=len(test) if test else 0, candidates=len(cands.strings),
        trie_nodes=len(cands.trie.tokens), args=vars(args))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layers = list(range(args.lora_first_layer, model.config.num_hidden_layers))
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0,
                                             target_modules="all-linear", layers_to_transform=layers))
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    values = cands.values.cuda()

    log(event="eval", step=0, split="dev", **evaluate(model, dev, cands))
    order, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        if len(order) < args.problems_per_step:
            order += random.sample(group_ids, len(group_ids))
        batch = [groups[g] for g in order[: args.problems_per_step]]
        del order[: args.problems_per_step]
        n_rows = sum(len(g) for g in batch)
        stats = collections.defaultdict(float)
        for group in batch:
            # Sampled: draw every row's completion first (no grad) so the group baseline is known.
            if args.objective == "sampled":
                draws = []
                for _, ids, correct in group:
                    with torch.no_grad():
                        q = candidate_logps(model, ids, cands.trie).exp().float()
                    probs = torch.cat([q, (1 - q.sum()).clamp_min(0)[None]])
                    j = torch.multinomial(probs / probs.sum(), 1).item()
                    reward = 0.0 if j == len(q) else rlcr_reward(values[j], torch.tensor(correct, device=values.device)).item()
                    draws.append((j, reward))
                baseline = np.mean([r for _, r in draws])
            for k, (_, ids, correct) in enumerate(group):
                logps = candidate_logps(model, ids, cands.trie)
                y = torch.tensor(correct, device=logps.device)
                expected = exact_objective(logps, values, y)
                if args.objective == "exact":
                    loss = -expected
                else:
                    j, reward = draws[k]
                    logq = torch.log1p(-logps.exp().sum().clamp(max=1 - 1e-6)) if j == len(logps) else logps[j]
                    loss = -(reward - baseline) * logq
                (loss / n_rows).backward()
                stats["expected_reward"] += expected.item() / n_rows
                stats["mass"] += logps.exp().sum().item() / n_rows
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            log(event="train", step=step, seconds=round(time.time() - t0, 1), rows=n_rows, **{k: round(v, 4) for k, v in stats.items()})
        if step % args.eval_every == 0 or step == args.steps:
            log(event="eval", step=step, split="dev", **evaluate(model, dev, cands))
    if test:
        log(event="eval", step=args.steps, split="test", **evaluate(model, test, cands))
    model.save_pretrained(os.path.join(args.out, "checkpoints", f"{args.objective}_s{args.seed}"))


if __name__ == "__main__":
    main()
