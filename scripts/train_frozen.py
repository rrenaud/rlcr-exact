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
Answer drift: LoRA acts at every position, so training only the confidence also moves the reasoning and answer
(rewarding_doubt, sweep 11). answer_kl is KL(current || untrained) over the vocabulary, summed over each row's
cached completion tokens before the confidence; the untrained model is the same one with the adapter off.
--answer-kl W adds W times it to each row's loss (rewarding_doubt's LoRA setting is W = 1). With --ref-topk DIR
(from ref_topk.py) the training penalty compares against the untrained model's cached top-k log-probs instead
of a live pass: both distributions coarse-grained to those k tokens plus one bucket for the rest, which is
never more than the exact KL (rewarding_doubt's topk_answer_kl). Dev metrics always report the exact KL, per
completion and per token, on the first --drift-rows rows. The cache freezes the answers, so this is
the only signal that regenerated answers would change.

usage: train_frozen.py --train ROLLOUTS --dev ROLLOUTS [--test ROLLOUTS] --objective exact|sampled
                       [--scorer digit|trie] ...
"""
import argparse, collections, json, math, os, random, sys, time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
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

# A training row: one cached rollout, cut after "<confidence>". j is the output's index within its question.
Row = collections.namedtuple("Row", "qid prefix correct start j")


def load_rows(path, token_bytes, max_prefix):
    """Well-formed rollouts as Rows (problem id, prefix token ids, correctness, prompt length, output index);
    counts what was dropped.

    The prefix is the sampled token ids themselves (prompt_ids + completion_ids), cut through the <confidence>
    tag that opens the final confidence, so training scores exactly the tokens the policy produced.
    """
    rows, dropped = [], collections.Counter()
    for line in open(path):
        q = json.loads(line)
        for j, (text, ids, ok, correct) in enumerate(zip(q["outputs"], q["completion_ids"], q["format_ok"], q["correct"])):
            cut = cut_point(text, ids, "confidence", token_bytes) if ok else None
            if cut is None:
                dropped["format"] += 1; continue
            prefix = q["prompt_ids"] + ids[:cut]
            if len(prefix) > max_prefix:
                dropped["long"] += 1; continue
            rows.append(Row(q["id"], prefix, float(correct), len(q["prompt_ids"]), j))
    return rows, dict(dropped)


def make_scorer(name, tok):
    """(values [K], score, description) where score(model, prefixes, greedy) gives [len(prefixes), K] logps,
    plus with greedy=True the index of the greedy candidate per row (None when greedy leaves the set). The digit
    scorer also takes hidden=True and then returns the forward's hidden states last (for answer_kl)."""
    if name == "digit":
        ids = DigitIds.from_tokenizer(tok)
        score = lambda model, prefixes, greedy=False, hidden=False: level_logps(model, prefixes, ids, greedy, hidden)
        return LEVELS, score, dict(levels=len(LEVELS))
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


KL_CHUNK = 512  # completion positions per KL chunk; each chunk is recomputed in backward to bound memory


def _chunk_kl(lm_head, h_cur, h_ref):
    cur = lm_head(h_cur).float().log_softmax(-1)
    ref = lm_head(h_ref).float().log_softmax(-1)
    return (cur.exp() * (cur - ref)).sum()


def _chunk_kl_topk(lm_head, h_cur, ref_ids, ref_logp):
    """KL with both distributions coarse-grained to the reference's top-k tokens plus one bucket for the rest."""
    top = lm_head(h_cur).float().log_softmax(-1).gather(-1, ref_ids)
    ref = ref_logp.float()
    rest_cur = torch.log1p(-top.exp().sum(-1).clamp(max=1 - 1e-6))
    rest_ref = torch.log1p(-ref.exp().sum(-1).clamp(max=1 - 1e-6))
    return (top.exp() * (top - ref)).sum() + (rest_cur.exp() * (rest_cur - rest_ref)).sum()


class RefTopk:
    """The untrained model's top-k log-probs at every cached completion position (written by ref_topk.py)."""

    def __init__(self, root):
        self.meta = json.load(open(os.path.join(root, "meta.json")))
        n, k = self.meta["positions"], self.meta["k"]
        self.ids = np.memmap(os.path.join(root, "ids.int32"), dtype=np.int32, mode="r", shape=(n, k))
        self.logp = np.memmap(os.path.join(root, "logp.float16"), dtype=np.float16, mode="r", shape=(n, k))
        self.index = {(qid, j): (off, length) for qid, j, off, length in self.meta["index"]}

    def get(self, row, device):
        off, length = self.index[(row.qid, row.j)]
        assert length == len(row.prefix) - row.start, "reference cache does not match these rows"
        return (torch.from_numpy(np.array(self.ids[off: off + length], dtype=np.int64)).to(device),
                torch.from_numpy(np.array(self.logp[off: off + length])).to(device))


def answer_kl(model, states, rows, ref_topk=None):
    """[len(rows)] KL(current || untrained) over the vocabulary, summed over each row's cached completion tokens
    before the confidence. states: the current model's last hidden states [rows, width, d] (right-padded, as
    level_logps returns them). The untrained side is a no-grad pass with the adapter off, or with ref_topk the
    cached top-k log-probs (coarse-grained KL, never more than the exact one)."""
    base = model.get_base_model()
    out = []
    for b, row in enumerate(rows):
        cur = states[b, row.start - 1: len(row.prefix) - 1]  # position t predicts token t + 1
        if ref_topk is None:
            with torch.no_grad(), model.disable_adapter():
                ids = torch.tensor([row.prefix], device=states.device)
                ref = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, row.start - 1: len(row.prefix) - 1]
            chunk = lambda c: (_chunk_kl, cur[c: c + KL_CHUNK], ref[c: c + KL_CHUNK])
        else:
            ref_ids, ref_logp = ref_topk.get(row, states.device)
            chunk = lambda c: (_chunk_kl_topk, cur[c: c + KL_CHUNK], ref_ids[c: c + KL_CHUNK], ref_logp[c: c + KL_CHUNK])
        total = states.new_zeros((), dtype=torch.float32)
        for c in range(0, len(cur), KL_CHUNK):
            fn, *tensors = chunk(c)
            total = total + checkpoint(fn, base.lm_head, *tensors, use_reentrant=False)
        out.append(total)
    return torch.stack(out)


@torch.no_grad()
def evaluate(model, rows, values, score, batch, drift_rows=()):
    """Dev metrics; with drift_rows, also their answer KL (per completion and per token)."""
    model.eval()
    greedy, expected, mass, y = [], [], [], []
    for part in chunks(rows, batch):
        logps, picks = score(model, [r[1] for r in part], greedy=True)
        for q, k, row in zip(logps.exp().cpu(), picks, part):
            greedy.append(np.nan if k is None else values[k].item())
            mass.append(q.sum().item())
            expected.append((q * values).sum().item() / max(q.sum().item(), 1e-9))
            y.append(row.correct)
    kls = [answer_kl(model, score(model, [r[1]], hidden=True)[1], [r]).item() for r in drift_rows]
    tokens = sum(len(r.prefix) - r.start for r in drift_rows)
    model.train()
    g, e, y = np.array(greedy), np.array(expected), np.array(y)
    ok = ~np.isnan(g)
    out = {"rows": len(y), "accuracy": y.mean(), "mass": float(np.mean(mass)), "greedy_off_list": float(1 - ok.mean())}
    for name, c, yy in (("greedy", g[ok], y[ok]), ("expected", e, y)):
        out |= {f"{name}_mean": c.mean(), f"{name}_sd": c.std(), f"{name}_brier": np.mean((c - yy) ** 2),
                f"{name}_ece": ece(c, yy, centred=name == "expected"),
                f"{name}_auroc": roc_auc_score(yy, c) if len(set(yy)) > 1 else float("nan")}
    if drift_rows:
        out |= {"answer_kl": np.mean(kls), "answer_kl_max": np.max(kls), "answer_kl_token": sum(kls) / tokens}
    return {k: round(float(v), 4) for k, v in out.items()}


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
    ap.add_argument("--drift-rows", type=int, default=200, help="dev rows for the answer-KL metric (0 = off)")
    ap.add_argument("--answer-kl", type=float, default=1.0, help="W: weight of the answer-KL penalty (0 = off)")
    ap.add_argument("--ref-topk", default=None, help="ref_topk.py cache of --train for the penalty (default: live pass)")
    ap.add_argument("--max-prefix", type=int, default=4608)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    run_name = f"{args.scorer}_{args.objective}_lr{args.lr:g}_W{args.answer_kl:g}_s{args.seed}"
    assert args.answer_kl == 0 or args.scorer == "digit", "the answer-KL penalty needs the digit scorer's hidden states"
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
    ref_topk = RefTopk(args.ref_topk) if args.ref_topk else None
    if ref_topk:
        m = ref_topk.meta
        assert (m["rollouts"], m["max_prefix"], m["model"]) == (os.path.abspath(args.train), args.max_prefix, args.model), \
            f"--ref-topk was built for {m['rollouts']} (max_prefix {m['max_prefix']}, {m['model']})"
        assert all((r.qid, r.j) in ref_topk.index for r in train), "--ref-topk is missing training rows"

    drift_rows = dev[: args.drift_rows] if args.scorer == "digit" else []
    evaluate_dev = lambda: evaluate(model, dev, values, score, args.eval_batch, drift_rows)
    log(event="eval", step=0, split="dev", **evaluate_dev())
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
                for q, row in zip(qs, group):
                    probs = torch.cat([q, (1 - q.sum()).clamp_min(0)[None]])
                    j = torch.multinomial(probs / probs.sum(), 1).item()
                    reward = 0.0 if j == len(q) else rlcr_reward(values_gpu[j], torch.tensor(row.correct, device=values_gpu.device)).item()
                    draws.append((j, reward))
                baseline = np.mean([r for _, r in draws])
            for start in range(0, len(group), args.micro_batch):
                part = group[start: start + args.micro_batch]
                if args.answer_kl:
                    logps, states = score(model, [r[1] for r in part], hidden=True)  # [rows, K], hidden states
                else:
                    logps = score(model, [r[1] for r in part])
                y = torch.tensor([r[2] for r in part], device=logps.device)
                expected = exact_objective(logps, values_gpu, y[:, None])  # [rows]
                if args.objective == "exact":
                    loss = -expected.sum()
                else:
                    loss = 0
                    for row, (j, reward) in zip(logps, draws[start: start + len(part)]):
                        logq = torch.log1p(-row.exp().sum().clamp(max=1 - 1e-6)) if j == len(row) else row[j]
                        loss = loss - (reward - baseline) * logq
                if args.answer_kl:
                    kl = answer_kl(model, states, part, ref_topk)
                    loss = loss + args.answer_kl * kl.sum()
                    stats["answer_kl"] += kl.sum().item() / n_rows
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
