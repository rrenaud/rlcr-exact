"""Continued training of RLCR's confidence on cached rollouts: exact expected reward vs sampled policy gradient.

Each training row is a cached rollout of the released checkpoint, cut after "<confidence>", with RLCR's
correctness label for its answer. The answer and analysis are fixed; only the distribution of the
confidence completion is trained, over one of two candidate sets scored exactly:

  --scorer digit (default): the one-decimal levels 0.0 .. 1.0 plus a bare "0", from "0" "." appended to the
                  prefix (digit_score.py); the tail after the digit is not scored
  --scorer trie:  114 strings x 2 end tokens, through a token trie (trie_score.py); the reference

  exact:   maximize sum_j q_j R_j (the unlisted mass earns 0, as a malformed completion does in RLCR)
  sampled: GRPO on the confidence tokens alone: sample one completion per row from q (or "other"),
           advantage = reward minus its problem group's mean (no std division, as in RLCR), loss -A log q

Defaults follow the rrenaud/rewarding_doubt sweeps (docs/sweeps): Adam at a constant 3e-4 with one update per
32 rows, LoRA rank 4 (alpha = rank) on the late half of the layers, one epoch.

Evaluation reports greedy and expected confidence. Greedy ECE uses the paper's 10 floor bins. Expected ECE uses
bins centred on the levels (nearest 0.1), because an expected confidence of 0.79998 vs 0.8 otherwise flips a bin.
It also reports answer drift: on the first --drift-rows dev rows, the summed log p_untrained - log p_current over
the cached completion tokens before the confidence. The cache freezes the answers, so this is the warning that
regenerated answers would change.

usage: train_frozen.py --train ROLLOUTS --dev ROLLOUTS [--test ROLLOUTS] --objective exact|sampled
                       [--scorer digit|trie] ...
"""
import argparse, collections, json, math, os, random, sys, time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlcr_exact.confidence import exact_objective, make_candidates, rlcr_reward
from rlcr_exact.digit_score import LEVELS, DigitIds, level_logps
from rlcr_exact.trie_score import candidate_logps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rlcr-reward-conf-weighted-tt-scaling"))
from logprob_spans import cut_point, make_token_bytes


# Every linear layer of a Qwen2 block, as "all-linear" selects; peft refuses a string with layers_to_transform.
LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path, token_bytes, max_prefix):
    """Well-formed rollouts as (problem id, prefix token ids, correctness, prompt length); counts what was dropped.

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
            rows.append((q["id"], prefix, float(correct), len(q["prompt_ids"])))
    return rows, dict(dropped)


def make_scorer(name, tok):
    """(values [K], score, description) where score(model, prefixes, greedy) gives [len(prefixes), K] logps,
    plus with greedy=True the index of the greedy candidate per row (None when greedy leaves the set)."""
    if name == "digit":
        ids = DigitIds.from_tokenizer(tok)
        return LEVELS, lambda model, prefixes, greedy=False: level_logps(model, prefixes, ids, greedy), dict(levels=len(LEVELS))
    cands = make_candidates(tok)

    def score(model, prefixes, greedy=False):
        outs = [candidate_logps(model, p, cands.trie, greedy=greedy) for p in prefixes]
        return (torch.stack([o[0] for o in outs]), [o[1] for o in outs]) if greedy else torch.stack(outs)
    return cands.values, score, dict(candidates=len(cands.strings), trie_nodes=len(cands.trie.tokens))


def chunks(xs, n):
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def ece(c, y, bins=10, centred=False):
    """Expected calibration error. Floor bins [k/10, (k+1)/10) as in the paper, or, with centred=True, one bin per
    level (nearest 0.1), for confidences that sit on or next to the levels."""
    b = np.rint(c * bins).astype(int) if centred else np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in np.unique(b)))


def stop_backward_at_first_adapter(model):
    """Remove the hooks that make the input embeddings require grad.

    transformers' gradient_checkpointing_enable and PEFT each register one (`make_inputs_require_grads`). They are
    only needed for reentrant checkpointing. With them, backward recomputes and differentiates every layer below
    the first adapted one for gradients nobody uses. Without them, the untrained early layers run forward only,
    and the LoRA gradients are unchanged.
    """
    emb = (model.get_base_model() if hasattr(model, "get_base_model") else model).get_input_embeddings()
    for key, hook in list(emb._forward_hooks.items()):
        if getattr(hook, "__name__", "") == "make_inputs_require_grads":
            del emb._forward_hooks[key]


def completion_logprob(model, prefix, start):
    """Summed log p of prefix[start:] given what precedes it (the cached reasoning and answer tokens)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids = torch.tensor([prefix], device=next(base.parameters()).device)
    hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, start - 1: -1]
    logp = base.lm_head(hidden).float().log_softmax(-1)
    return logp.gather(-1, ids[0, start:, None]).sum().item()


@torch.no_grad()
def evaluate(model, rows, values, score, batch, drift_rows=(), drift_base=None):
    """Dev metrics; with drift_rows, also answer drift against drift_base (their untrained completion log-probs)."""
    model.eval()
    greedy, expected, mass, y = [], [], [], []
    for part in chunks(rows, batch):
        logps, picks = score(model, [r[1] for r in part], greedy=True)
        for q, k, (_, _, correct, _) in zip(logps.exp().cpu(), picks, part):
            greedy.append(np.nan if k is None else values[k].item())
            mass.append(q.sum().item())
            expected.append((q * values).sum().item() / max(q.sum().item(), 1e-9))
            y.append(correct)
    drift = [completion_logprob(model, ids, start) for _, ids, _, start in drift_rows]
    model.train()
    g, e, y = np.array(greedy), np.array(expected), np.array(y)
    ok = ~np.isnan(g)
    out = {"rows": len(y), "accuracy": y.mean(), "mass": float(np.mean(mass)), "greedy_off_list": float(1 - ok.mean())}
    for name, c, yy in (("greedy", g[ok], y[ok]), ("expected", e, y)):
        out |= {f"{name}_mean": c.mean(), f"{name}_sd": c.std(), f"{name}_brier": np.mean((c - yy) ** 2),
                f"{name}_ece": ece(c, yy, centred=name == "expected"),
                f"{name}_auroc": roc_auc_score(yy, c) if len(set(yy)) > 1 else float("nan")}
    if drift_rows and drift_base is not None:
        gap = np.array(drift_base) - np.array(drift)
        out |= {"answer_drift_mean": gap.mean(), "answer_drift_max": gap.max()}
    out = {k: round(float(v), 4) for k, v in out.items()}
    return (out, drift) if drift_rows else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-brier-base-tabc")
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test")
    ap.add_argument("--objective", choices=["exact", "sampled"], required=True)
    ap.add_argument("--scorer", choices=["digit", "trie"], default="digit")
    ap.add_argument("--micro-batch", type=int, default=1, help="rows per forward/backward when training")
    ap.add_argument("--eval-batch", type=int, default=1, help="rows per no-grad forward (evaluation, sampling)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-first-layer", type=int, default=14, help="adapt only layers >= this one (Qwen2.5-7B has 28)")
    ap.add_argument("--problems-per-step", type=int, default=4, help="problems per update (x their rollouts each)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--drift-rows", type=int, default=200, help="dev rows for the answer-drift metric (0 = off)")
    ap.add_argument("--max-prefix", type=int, default=4608)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    run_name = f"{args.scorer}_{args.objective}_lr{args.lr:g}_s{args.seed}"
    log_path = os.path.join(args.out, f"train_{run_name}.jsonl")

    def log(**rec):
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    values, score, scorer_info = make_scorer(args.scorer, tok)
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
        dev_dropped=dev_dropped, test_rows=len(test) if test else 0, **scorer_info, args=vars(args))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layers = list(range(args.lora_first_layer, model.config.num_hidden_layers))
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0,
                                             target_modules=LINEAR_MODULES, layers_to_transform=layers))
    stop_backward_at_first_adapter(model)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    evaluate_on = lambda rows: evaluate(model, rows, values, score, args.eval_batch)
    values_gpu = values.cuda()

    # LoRA starts at zero, so step 0 is the untrained model: its completion log-probs are the drift baseline.
    drift_rows = dev[: args.drift_rows]
    metrics, drift_base = evaluate(model, dev, values, score, args.eval_batch, drift_rows)
    log(event="eval", step=0, split="dev", **metrics)
    evaluate_dev = lambda: evaluate(model, dev, values, score, args.eval_batch, drift_rows, drift_base)[0]
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
                with torch.no_grad():
                    qs = torch.cat([score(model, [r[1] for r in part]).exp().float()
                                    for part in chunks(group, args.eval_batch)])
                for q, (_, _, correct, _) in zip(qs, group):
                    probs = torch.cat([q, (1 - q.sum()).clamp_min(0)[None]])
                    j = torch.multinomial(probs / probs.sum(), 1).item()
                    reward = 0.0 if j == len(q) else rlcr_reward(values_gpu[j], torch.tensor(correct, device=values_gpu.device)).item()
                    draws.append((j, reward))
                baseline = np.mean([r for _, r in draws])
            for start in range(0, len(group), args.micro_batch):
                part = group[start: start + args.micro_batch]
                logps = score(model, [r[1] for r in part])  # [rows, K]
                y = torch.tensor([r[2] for r in part], device=logps.device)
                expected = exact_objective(logps, values_gpu, y[:, None])  # [rows]
                if args.objective == "exact":
                    loss = -expected.sum()
                else:
                    loss = 0
                    for row, (j, reward) in zip(logps, draws[start: start + len(part)]):
                        logq = torch.log1p(-row.exp().sum().clamp(max=1 - 1e-6)) if j == len(row) else row[j]
                        loss = loss - (reward - baseline) * logq
                (loss / n_rows).backward()
                stats["expected_reward"] += expected.sum().item() / n_rows
                stats["mass"] += logps.exp().sum().item() / n_rows
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            log(event="train", step=step, seconds=round(time.time() - t0, 1), rows=n_rows, **{k: round(v, 4) for k, v in stats.items()})
        if step % args.eval_every == 0 or step == args.steps:
            log(event="eval", step=step, split="dev", **evaluate_dev())
    if test:
        log(event="eval", step=args.steps, split="test", **evaluate_on(test))
    model.save_pretrained(os.path.join(args.out, "checkpoints", run_name))


if __name__ == "__main__":
    main()
