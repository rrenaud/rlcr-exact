"""Continued training of RLCR's confidence on cached rollouts: exact expected reward vs sampled policy gradient.

Each training row is a cached rollout of the released checkpoint, cut after "<confidence>", with RLCR's
correctness label for its answer. The answer and analysis are fixed; only the distribution of the
confidence completion is trained, over one of two candidate sets scored exactly:

  --scorer digit (default): the one-decimal levels 0.0 .. 1.0 plus a bare "0", from "0" "." appended to the
                  prefix (digit_score.py); the tail after the digit is not scored
  --scorer trie:  114 strings x 2 end tokens, through a token trie (trie_score.py); the reference
  --scorer hotpot: RLCR-hotpot's 21 values x {spaced, unspaced}, through a trie up to the closing `</`
                  (confidence.hotpot_candidates)

  exact:   maximize sum_j q_j R_j (the unlisted mass earns 0, as a malformed completion does in RLCR)
  sampled: GRPO on the confidence tokens alone: sample --samples-per-row k completions per row from q (or "other"),
           advantage = reward minus the mean over the problem group's rows x k draws (no std division, as in
           RLCR), loss -(1/k) sum A log q per row. Exact is its k -> infinity limit (a baseline changes only variance).

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
import argparse, collections, gzip, json, math, os, random, sys, time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlcr_exact.confidence import exact_objective, hotpot_candidates, hotpot_stop_candidates, level_matrix, make_candidates, rlcr11_candidates, rlcr11_math_candidates, rlcr_reward, support_floor
from rlcr_exact.digit_score import LEVELS, DigitIds, level_logps
from rlcr_exact.grading import em_correct, f1_correct, tail_only_failure
from rlcr_exact.trie_score import packed_candidate_logps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rlcr-reward-conf-weighted-tt-scaling"))
from logprob_spans import cut_point, make_token_bytes


# Every linear layer of a Qwen2 block, as "all-linear" selects; peft refuses a string with layers_to_transform.
LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# A training row: one cached rollout, cut after "<confidence>". j is the output's index within its question.
# correct is RLCR's label (exact match for HotpotQA), which training uses; correct_f1 (F1 >= 0.5) is reported.
STOP_SCORERS = ("hotpot-stop", "rlcr11")  # score through the end token, so tail-fault rows are kept (earn 0 off-list)
Row = collections.namedtuple("Row", "qid prefix correct start j correct_f1")


def load_rows(path, token_bytes, max_prefix, tail_ok=False):
    """Well-formed rollouts as Rows (problem id, prefix token ids, correctness, prompt length, output index);
    counts what was dropped.

    The prefix is the sampled token ids themselves (prompt_ids + completion_ids), cut through the <confidence>
    tag that opens the final confidence, so training scores exactly the tokens the policy produced.
    With tail_ok, rows whose only format fault is after </confidence> (a junk token instead of end of text) are
    kept too, labelled by exact match of their last <answer> (RLCR's own label is 0 for them, gated on format).
    """
    rows, dropped = [], collections.Counter()
    for line in open(path):
        rows += question_rows(json.loads(line), token_bytes, max_prefix, tail_ok, dropped)
    return rows, dict(dropped)


def question_rows(q, token_bytes, max_prefix, tail_ok, dropped, qid=None):
    """load_rows for one question's rollouts (a cache line, or a fresh rollout dict from online.Roller)."""
    rows = []
    for j, (text, ids, ok, correct) in enumerate(zip(q["outputs"], q["completion_ids"], q["format_ok"], q["correct"])):
        tail_fault = tail_ok and not ok and tail_only_failure(text)
        cut = cut_point(text, ids, "confidence", token_bytes) if ok or tail_fault else None
        if cut is None:
            dropped["format"] += 1; continue
        if tail_fault:
            correct = em_correct(text, q["answer"])
            dropped["kept_tail_fault"] += 1
        prefix = q["prompt_ids"] + ids[:cut]
        if len(prefix) > max_prefix:
            dropped["long"] += 1; continue
        rows.append(Row(qid or q["id"], prefix, float(correct), len(q["prompt_ids"]), j, f1_correct(text, q["answer"])))
    return rows


def make_scorer(name, tok):
    """(values [K], score, description) where score(model, prefixes, greedy) gives [len(prefixes), K] logps,
    plus with greedy=True the index of the greedy candidate per row (None when greedy leaves the set). The digit
    and trie scorers also take hidden=True and then return the forward's hidden states last (for answer_kl): a
    [rows, width, d] tensor for digit, a list of per-row [width, d] tensors for the tries."""
    if name == "digit":
        ids = DigitIds.from_tokenizer(tok)
        score = lambda model, prefixes, greedy=False, hidden=False: level_logps(model, prefixes, ids, greedy, hidden)
        return LEVELS, score, dict(levels=len(LEVELS))
    cands = (hotpot_candidates(tok) if name == "hotpot" else hotpot_stop_candidates(tok) if name == "hotpot-stop"
             else rlcr11_candidates(tok) if name == "rlcr11" else rlcr11_math_candidates(tok) if name == "rlcr11-math"
             else make_candidates(tok))

    def score(model, prefixes, greedy=False, hidden=False):
        # One packed forward: the rows' common start (a question's prompt, shared by its rollouts) runs once.
        return packed_candidate_logps(model, prefixes, cands.trie, greedy=greedy, hidden=hidden)
    return cands.values, score, dict(candidates=len(cands.strings), trie_nodes=len(cands.trie.tokens))


def chunks(xs, n):
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def ece(c, y, bins=10, centred=False):
    """Expected calibration error. Floor bins [k/10, (k+1)/10) as in the paper, or, with centred=True, one bin per
    level (nearest 0.1), for confidences that sit on or next to the levels."""
    b = np.rint(c * bins).astype(int) if centred else np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in np.unique(b)))


def floor_weight(args, step):
    """Support-floor beta at this step: constant, or linear to 0 from --floor-decay-from (a fraction of the steps)."""
    if args.floor_decay_from is None:
        return args.support_floor
    start = args.floor_decay_from * args.steps
    return args.support_floor * min(1.0, max(0.0, (args.steps - step) / max(args.steps - start, 1)))


def sample_draws(qs, correct, values, k):
    """k draws per row from its candidate distribution plus "other" (index K, reward 0): [[(j, reward)] * k] per row."""
    out = []
    for q, y in zip(qs, correct):
        probs = torch.cat([q, (1 - q.sum()).clamp_min(0)[None]])
        js = torch.multinomial(probs / probs.sum(), k, replacement=True).tolist()
        out.append([(j, 0.0 if j == len(q) else rlcr_reward(values[j], torch.tensor(y, device=values.device)).item()) for j in js])
    return out


def rlcr_draws(draws, group, zero_reward, n_bad):
    """RLCR's k = 1 reward: a tail fault earns 0 for its drawn confidence too, and the group baseline is the mean over
    all the question's rollouts (malformed ones earn 0), the same baseline the answer term uses."""
    draws = [[(j, 0.0 if (row.qid, row.j) in zero_reward else rew) for j, rew in d] for row, d in zip(group, draws)]
    return draws, sum(rew for d in draws for _, rew in d) / (len(group) + n_bad)


def sampled_loss(logps, draws, baseline):
    """GRPO loss on the confidence: -(1/k) sum_draws (reward - baseline) log q_j per row, summed over rows."""
    loss = 0
    for row, row_draws in zip(logps, draws):
        for j, reward in row_draws:
            logq = torch.log1p(-row.exp().sum().clamp(max=1 - 1e-6)) if j == len(row) else row[j]
            loss = loss - (reward - baseline) * logq / len(row_draws)
    return loss


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
    before the confidence. states: the current model's last hidden states per row ([rows, width, d] right-padded,
    as level_logps returns them, or a list of per-row [width, d] from the trie scorers). The untrained side is a no-grad pass with the adapter off, or with ref_topk the
    cached top-k log-probs (coarse-grained KL, never more than the exact one)."""
    base = model.get_base_model()
    out = []
    for b, row in enumerate(rows):
        cur = states[b][row.start - 1: len(row.prefix) - 1]  # position t predicts token t + 1
        if ref_topk is None:
            with torch.no_grad(), model.disable_adapter():
                ids = torch.tensor([row.prefix], device=cur.device)
                ref = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, row.start - 1: len(row.prefix) - 1]
            chunk = lambda c: (_chunk_kl, cur[c: c + KL_CHUNK], ref[c: c + KL_CHUNK])
        else:
            ref_ids, ref_logp = ref_topk.get(row, cur.device)
            chunk = lambda c: (_chunk_kl_topk, cur[c: c + KL_CHUNK], ref_ids[c: c + KL_CHUNK], ref_logp[c: c + KL_CHUNK])
        total = cur.new_zeros((), dtype=torch.float32)
        for c in range(0, len(cur), KL_CHUNK):
            fn, *tensors = chunk(c)
            total = total + checkpoint(fn, base.lm_head, *tensors, use_reentrant=False)
        out.append(total)
    return torch.stack(out)


def _chunk_logp(lm_head, h, ids):
    return lm_head(h).float().log_softmax(-1).gather(-1, ids[:, None]).sum()


def answer_logp(model, states, prefixes, starts):
    """[rows] summed log-probs of each row's tokens prefix[start:] (with grad), from the forward's hidden states;
    chunked and recomputed in backward like the KL."""
    head = model.get_base_model().lm_head
    out = []
    for h_all, p, s in zip(states, prefixes, starts):
        h, ids = h_all[s - 1: len(p) - 1], torch.tensor(p[s:], device=h_all.device)
        total = h.new_zeros((), dtype=torch.float32)
        for c in range(0, len(h), KL_CHUNK):
            total = total + checkpoint(_chunk_logp, head, h[c: c + KL_CHUNK], ids[c: c + KL_CHUNK], use_reentrant=False)
        out.append(total)
    return torch.stack(out)


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().float() for p in params])


def grad_report(parts, previous):
    """Per-term gradient norms (of the step's mean loss), cosines between terms, and each term's cosine with its
    previous measurement (a cheap consistency proxy over time)."""
    cos = lambda u, v: (u @ v / (u.norm() * v.norm()).clamp_min(1e-30)).item()
    out = {f"norm_{n}": g.norm().item() for n, g in parts.items()}
    out["norm_total"] = sum(parts.values()).norm().item()
    names = sorted(parts)
    out |= {f"cos_{a}_{b}": cos(parts[a], parts[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    out |= {f"cos_prev_{n}": cos(g, previous[n]) for n, g in parts.items() if n in previous}
    return {k: round(v, 5) for k, v in out.items()}


@torch.no_grad()
def adam_fraction(opt):
    """Median over coordinates of |m_hat| / sqrt(v_hat): Adam's step along each coordinate's running-mean
    direction, as a fraction of the learning rate (1 = consistent gradient, ~0.1-0.2 = noise)."""
    beta1, beta2 = opt.param_groups[0]["betas"]
    ratios = []
    for p in opt.param_groups[0]["params"]:
        st_ = opt.state.get(p)
        if not st_:
            continue
        t = float(st_["step"])
        m, v = st_["exp_avg"] / (1 - beta1 ** t), st_["exp_avg_sq"] / (1 - beta2 ** t)
        nz = v > 0
        ratios.append((m.abs()[nz] / v[nz].sqrt()).flatten().float())
    return round(torch.cat(ratios).median().item(), 4) if ratios else None


@torch.no_grad()
def evaluate(model, rows, values, score, batch, drift_rows=()):
    """Dev metrics; with drift_rows, also their answer KL (per completion and per token).

    Support-floor diagnostics, on each row's level distribution p (level marginals renormalized over the list):
    entropy (nats; ln 11 = 2.40 is uniform), the mean p per level (all / correct / wrong rows), the spread cost
    (reward of p's top level minus p's expected reward: what the spread gives up), and both terms' ascent gradients
    in level-logit space, treating p as a softmax over the 11 levels: objective p_l (R_l - sum p R), floor (beta = 1)
    1/L - p_l. Their norm ratio times beta is what the floor weighs against the objective there; cos < 0 means the
    floor opposes it. This view ignores the LoRA geometry (the grad events measure that, every --grad-decomp-every)."""
    model.eval()
    greedy, expected, mass, y, y_f1, level_min = [], [], [], [], [], []
    ent, level_p, spread_cost, g_obj, g_floor, g_cos = [], [], [], [], [], []
    levels = level_matrix(values)
    grid = torch.linspace(0, 1, levels.shape[0], dtype=values.dtype)
    for part in chunks(rows, batch):
        logps, picks = score(model, [r[1] for r in part], greedy=True)
        for q, k, row in zip(logps.exp().cpu(), picks, part):
            greedy.append(np.nan if k is None else values[k].item())
            mass.append(q.sum().item())
            level_min.append((levels @ q).min().item())
            p = (levels @ q).clamp_min(1e-12); p = p / p.sum()
            R = rlcr_reward(grid, torch.tensor(float(row.correct)))
            go, gf = p * (R - (p * R).sum()), 1 / len(p) - p
            ent.append(-(p * p.log()).sum().item()); level_p.append(p.float().numpy())
            spread_cost.append((R[p.argmax()] - (p * R).sum()).item())
            g_obj.append(go.norm().item()); g_floor.append(gf.norm().item())
            g_cos.append((go @ gf / (go.norm() * gf.norm()).clamp_min(1e-12)).item())
            expected.append((q * values).sum().item() / max(q.sum().item(), 1e-9))
            y.append(row.correct)
            y_f1.append(row.correct_f1)
    kls = [answer_kl(model, score(model, [r[1]], hidden=True)[1], [r]).item() for r in drift_rows]
    tokens = sum(len(r.prefix) - r.start for r in drift_rows)
    model.train()
    g, e, y, y_f1 = np.array(greedy), np.array(expected), np.array(y), np.array(y_f1)
    ok = ~np.isnan(g)
    out = {"rows": len(y), "accuracy": y.mean(), "accuracy_f1": y_f1.mean(), "mass": float(np.mean(mass)),
           "level_min_median": float(np.median(level_min)), "greedy_off_list": float(1 - ok.mean())}
    for name, c, yy, yf in (("greedy", g[ok], y[ok], y_f1[ok]), ("expected", e, y, y_f1)):
        out |= {f"{name}_mean": c.mean(), f"{name}_sd": c.std(),
                f"{name}_mean_correct": c[yy == 1].mean() if (yy == 1).any() else np.nan,
                f"{name}_mean_wrong": c[yy == 0].mean() if (yy == 0).any() else np.nan}
        for suffix, labels in (("", yy), ("_f1", yf)):  # RLCR's labels, then F1 >= 0.5
            out |= {f"{name}_brier{suffix}": np.mean((c - labels) ** 2),
                    f"{name}_ece{suffix}": ece(c, labels, centred=name == "expected"),
                    f"{name}_auroc{suffix}": roc_auc_score(labels, c) if len(set(labels)) > 1 else float("nan")}
    ent, spread_cost, g_obj, g_floor, g_cos, level_p = map(np.array, (ent, spread_cost, g_obj, g_floor, g_cos, level_p))
    for suffix, m in (("", np.ones(len(y), bool)), ("_correct", y == 1), ("_wrong", y == 0)):
        if m.any():
            out |= {f"level_entropy{suffix}": ent[m].mean(), f"spread_cost{suffix}": spread_cost[m].mean(),
                    f"grad_obj_level{suffix}": np.median(g_obj[m]), f"grad_floor_level{suffix}": np.median(g_floor[m]),
                    f"grad_ratio_level{suffix}": np.median(g_floor[m] / np.maximum(g_obj[m], 1e-12)),
                    f"grad_cos_level{suffix}": np.median(g_cos[m])}
    out |= {"level_entropy_median": np.median(ent)}
    if drift_rows:
        out |= {"answer_kl": np.mean(kls), "answer_kl_max": np.max(kls), "answer_kl_token": sum(kls) / tokens}
    out = {k: round(float(v), 4) for k, v in out.items()}
    for suffix, m in (("", np.ones(len(y), bool)), ("_correct", y == 1), ("_wrong", y == 0)):
        if m.any():
            out[f"level_p{suffix}"] = [round(float(v), 5) for v in level_p[m].mean(0)]
    return out


def reward_snr(q, values, y, zero, n_bad):
    """One group's reward signal and confidence-sampling noise, by component. Per rollout, the exact reward
    r = sum_c q(c) R(c) splits into format (0.5), correctness (0.5 y) and calibration (the rest: the Brier term and
    the unlisted mass's loss, everything that depends on c); malformed rollouts and tail faults are 0 in all three.
    Signal = each component's variance across the group's rollouts (what the advantage r - mean sees); noise = the
    mean over rollouts of Var_q[R], the variance a sampled c (RLCR, k = 1) adds to each rollout's reward. cal_within:
    the calibration component's pooled variance within rollouts of the same correctness. All variances divide by the
    group's rollout count, so they add up (with covariances) to the advantage's variance."""
    R = rlcr_reward(values, y[:, None]).float()
    e = (q * R).sum(-1)
    keep = ~zero
    comps = {"format": torch.where(keep, 0.5, 0.0), "correct": torch.where(keep, 0.5 * y, 0.0),
             "cal": torch.where(keep, e - 0.5 - 0.5 * y, 0.0)}
    comps["total"] = comps["format"] + comps["correct"] + comps["cal"]
    noise = torch.where(keep, (q * R ** 2).sum(-1) - e ** 2, 0.0)
    pad = torch.zeros(n_bad, device=q.device)
    n = len(y) + n_bad
    out = {k: torch.cat([v, pad]).var(unbiased=False).item() for k, v in comps.items()}
    within = sum(((comps["cal"][keep & (y == v)] - comps["cal"][keep & (y == v)].mean()) ** 2).sum().item()
                 for v in (0.0, 1.0) if (keep & (y == v)).any())
    out |= {"cal_within": within / n, "noise": noise.sum().item() / n}
    return out, bool((y[keep] == 1).any() and (y[keep] == 0).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: the released checkpoint for the scorer's task")
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test")
    ap.add_argument("--objective", choices=["exact", "sampled"], required=True)
    ap.add_argument("--scorer", choices=["digit", "trie", "hotpot", "hotpot-stop", "rlcr11", "rlcr11-math"], default="digit",
                    help="hotpot-stop: the hotpot candidates through `confidence>` + end token, and rows whose only fault is the "
                         "tail after </confidence> are kept (needs a reference cache built with ref_topk.py --tail-ok); rlcr11: only "
                         "0.0 .. 1.0 in the spaced spelling, each through the end token, same rows as hotpot-stop; rlcr11-math: "
                         "0.0 .. 1.0 in RLVR-math's newline spelling, closed by > or >\\n, then <|endoftext|>")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="rows per forward/backward when training (default: a whole problem group for the trie "
                         "scorers, whose rows share the prompt and are packed into one forward; 1 for digit)")
    ap.add_argument("--eval-batch", type=int, default=1, help="rows per no-grad forward (evaluation, sampling)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-first-layer", type=int, default=14, help="adapt only layers >= this one (Qwen2.5-7B has 28)")
    ap.add_argument("--problems-per-step", type=int, default=4, help="problems per update (x their rollouts each)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--init-adapter", default=None,
                    help="continue from a saved adapter dir (and its optimizer.pt, if there) instead of a fresh LoRA")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save adapter + optimizer state every N steps, to checkpoints/RUN/stepN")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--skip-initial-eval", action="store_true",
                    help="no step-0 dev eval (identical for every run from the same base; saves ~3.5 min on hotpot)")
    ap.add_argument("--dev-limit", type=int, default=None)
    ap.add_argument("--drift-rows", type=int, default=200, help="dev rows for the answer-KL metric (0 = off)")
    ap.add_argument("--answer-kl", type=float, default=1.0, help="W: weight of the answer-KL penalty (0 = off)")
    ap.add_argument("--samples-per-row", type=int, default=1,
                    help="sampled: confidence draws per cached prefix per step (k); one scoring pass serves all k")
    ap.add_argument("--floor-decay-from", type=float, default=None,
                    help="anneal the support floor linearly to 0 from this fraction of the steps to the end (default: constant)")
    ap.add_argument("--support-floor", type=float, default=0.0,
                    help="beta: weight of the cross-entropy from a uniform target over the levels 0.0..1.0 to each "
                         "row's level marginal (0 = off); keeps probability on every level so training cannot collapse")
    ap.add_argument("--grad-decomp-every", type=int, default=0,
                    help="diagnostic, off by default: every N steps, backpropagate each loss term separately and log its "
                         "gradient norm, the cosines between terms and with the term's previous measurement (about 2 "
                         "extra backwards per measured step; 25 costs ~5%%). Use while tuning loss weights; leave it "
                         "off once training is stable (adam_fraction is always logged and costs nothing)")
    ap.add_argument("--sampled-prepass", action="store_true",
                    help="draw sampled completions from a separate no-grad forward even when the group is one forward "
                         "(the old path; same draws, slower; for checking)")
    ap.add_argument("--ref-topk", default=None, help="ref_topk.py cache of --train for the penalty (default: live pass)")
    ap.add_argument("--max-prefix", type=int, default=4608)
    ap.add_argument("--train-problems", type=int, default=None,
                    help="overfit check: train on only the first N problems (sorted id) with mixed correctness, "
                         "and also evaluate on their rows (split 'train')")
    ap.add_argument("--train-band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="with --train-problems: only problems whose cached accuracy is in [LO, HI] (default: in (0, 1))")
    ap.add_argument("--conf-weight", type=float, default=1.0,
                    help="weight of the confidence objective (0 = off: Stage B answer-side checks without it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="vllm: in-process vLLM holds the base weights and the trained model is bound to them (one copy; "
                         "src/rlcr_exact/shared_vllm.py, vLLM env); training is otherwise identical")
    ap.add_argument("--vllm-kv-gb", type=float, default=2.0, help="vLLM KV-cache size with --backend vllm")
    ap.add_argument("--vllm-eager", action="store_true", help="no CUDA graphs in vLLM (saves memory, slower generation)")
    ap.add_argument("--refresh-every", type=int, default=0,
                    help="N > 0 (needs --backend vllm): every N steps, fresh rollouts of the next N steps' questions from the "
                         "current adapter (k per question, as the cache: T 0.7, RLCR's tabc labels), and the answer-KL "
                         "reference from the adapter-off model (top --ref-k). N = 1 is on-policy; 0 trains on the cache")
    ap.add_argument("--gen-dev-every", type=int, default=0,
                    help="with --backend vllm: greedy generation of the dev questions (--dev-limit) with the current "
                         "adapter every N steps and at the start (event gen_dev: accuracy, format, length, stated-"
                         "confidence Brier/AUROC, answers changed vs the cached base greedy rollouts); 0 = off")
    ap.add_argument("--rollouts-per-question", type=int, default=8)
    ap.add_argument("--gen-temperature", type=float, default=0.7)
    ap.add_argument("--gen-max-new", type=int, default=1536)
    ap.add_argument("--ref-k", type=int, default=64)
    ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR", help="RLCR checkout (reward_fns) for online grading")
    ap.add_argument("--answer-pg", type=float, default=0.0,
                    help="Stage B (needs --refresh-every): weight alpha of REINFORCE on the answer-side tokens (completion "
                         "through <confidence>) with group-mean advantages and RLCR's bnpo normalization (summed token "
                         "terms / answer tokens in the step); malformed rollouts and tail faults get reward 0 (RLCR's rule)")
    ap.add_argument("--pg-balance", choices=["fixed", "rlcr"], default="fixed",
                    help="fixed: the answer term is --answer-pg x (its token-mean); rlcr: both terms are averaged over the "
                         "step's rollouts, which is RLCR's answer:confidence balance in expectation (its token-level "
                         "loss with the confidence tokens' REINFORCE replaced by the exact gradient); --answer-pg is then "
                         "1/kappa, the inverse confidence boost over RLCR (1 = RLCR). Logged as alpha_eff in fixed units")
    ap.add_argument("--pg-reward", choices=["exact", "accuracy", "sampled"], default="exact",
                    help="exact: r = sum_c q(c|a) R(a, c) from the same forward (the confidence integrated out); "
                         "accuracy: r = RLCR's correctness label (validation: can the answers learn at all); "
                         "sampled: RLCR (k = 1) — r = R(a, c) of the confidence drawn for the sampled objective, one "
                         "advantage per rollout for answer and confidence alike (needs --objective sampled; tail faults "
                         "and malformed rollouts earn 0, and both terms share the group baseline over all rollouts)")
    ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
    ap.add_argument("--no-log-q", action="store_true",
                    help="skip q_<run>.jsonl.gz: every training rollout's confidence distribution over the 11 levels at each step (before the update), with its label, expected reward and Var_q[R]")
    args = ap.parse_args()
    packed = args.scorer in ("trie", "hotpot", "hotpot-stop", "rlcr11", "rlcr11-math")
    args.model = args.model or ("mehuldamani/hotpot-v2-brier-7b-no-split" if args.scorer.startswith("hotpot") or args.scorer == "rlcr11"
                                else "mehuldamani/big-math-digits-v2-correctness" if args.scorer == "rlcr11-math"
                                else "mehuldamani/big-math-digits-v2-brier-base-tabc")
    random.seed(args.seed); torch.manual_seed(args.seed)
    assert args.pg_reward != "sampled" or (args.objective == "sampled" and args.samples_per_row == 1 and args.answer_pg), \
        "--pg-reward sampled is RLCR's k = 1 arm: --objective sampled, --samples-per-row 1, --answer-pg"
    run_name = f"{args.scorer}_{args.objective}_lr{args.lr:g}_W{args.answer_kl:g}_s{args.seed}"
    if args.support_floor:
        run_name += f"_F{args.support_floor:g}"
        if args.floor_decay_from is not None:
            run_name += f"-anneal{args.floor_decay_from:g}"
    if args.init_adapter:
        run_name += "_cont"
    if args.samples_per_row > 1:
        run_name += f"_k{args.samples_per_row}"
    if args.train_problems:
        run_name += f"_overfit{args.train_problems}"
    if args.conf_weight != 1:
        run_name += f"_C{args.conf_weight:g}"
    if args.answer_pg:
        run_name += f"_pg{args.answer_pg:g}{args.pg_reward[:3]}" + ("_bRLCR" if args.pg_balance == "rlcr" else "")
    log_path = os.path.join(args.out, f"train_{run_name}.jsonl")

    def log(**rec):
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    values, score, scorer_info = make_scorer(args.scorer, tok)
    token_bytes = make_token_bytes(tok)
    tail_ok = args.scorer in STOP_SCORERS
    train, dropped = load_rows(args.train, token_bytes, args.max_prefix, tail_ok)
    dev, dev_dropped = load_rows(args.dev, token_bytes, args.max_prefix, tail_ok)
    dev = dev[: args.dev_limit]
    test = load_rows(args.test, token_bytes, args.max_prefix, tail_ok)[0] if args.test else None
    groups = collections.defaultdict(list)
    for row in train:
        groups[row[0]].append(row)
    group_ids = sorted(groups)
    if args.train_problems:
        lo, hi = args.train_band or (1e-9, 1 - 1e-9)
        group_ids = [g for g in group_ids if lo <= np.mean([r.correct for r in groups[g]]) <= hi][: args.train_problems]
        train = [r for g in group_ids for r in groups[g]]
    log(event="data", train_rows=len(train), train_problems=len(groups), dropped=dropped, dev_rows=len(dev),
        dev_dropped=dev_dropped, test_rows=len(test) if test else 0, **scorer_info, args=vars(args))

    llm = None
    if args.backend == "vllm":
        from rlcr_exact.shared_vllm import bind_hf, start_vllm
        llm = start_vllm(args.model, kv_gb=args.vllm_kv_gb, max_model_len=args.max_prefix + 1536,
                         max_lora_rank=max(8, args.lora_r), seed=args.seed, eager=args.vllm_eager)
        model = bind_hf(llm, args.model)
        free, total = torch.cuda.mem_get_info()
        log(event="backend", backend="vllm", gpu_used_gb=round((total - free) / 2 ** 30, 2))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layers = list(range(args.lora_first_layer, model.config.num_hidden_layers))
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0,
                                                 target_modules=LINEAR_MODULES, layers_to_transform=layers))
    stop_backward_at_first_adapter(model)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    opt_path = os.path.join(args.init_adapter or "", "optimizer.pt")
    if args.init_adapter and os.path.exists(opt_path):
        opt.load_state_dict(torch.load(opt_path))
        for g in opt.param_groups:
            g["lr"] = args.lr
    def save(path):  # adapter + Adam state, so a run can be continued (--init-adapter path)
        model.save_pretrained(path)
        torch.save(opt.state_dict(), os.path.join(path, "optimizer.pt"))
    evaluate_on = lambda rows: evaluate(model, rows, values, score, args.eval_batch)
    values_gpu = values.cuda()
    levels_gpu = level_matrix(values_gpu)
    ref_topk = RefTopk(args.ref_topk) if args.ref_topk and not args.refresh_every else None
    if args.refresh_every or args.gen_dev_every:
        assert llm is not None, "--refresh-every and --gen-dev-every need --backend vllm"
    assert not args.answer_pg or args.refresh_every, "--answer-pg needs --refresh-every (fresh rollouts)"
    if args.refresh_every or args.gen_dev_every:
        from rlcr_exact.online import OnlineRefTopk, Roller, gen_metrics, load_questions, rollout_stats
        from rlcr_exact.shared_vllm import AdapterPublisher
        questions = {q["id"]: q for q in load_questions(args.train) if q["id"] in groups}
        roller = Roller(llm, args.rlcr, k=args.rollouts_per_question, temperature=args.gen_temperature,
                        max_new=args.gen_max_new, seed=args.seed)
        publisher = AdapterPublisher(f"/dev/shm/{os.environ.get('USER', 'u')}_{os.path.basename(args.out.rstrip('/'))}_{run_name}")
    if args.gen_dev_every:
        dev_base = [json.loads(l) for l in open(args.dev)][: args.dev_limit]  # the cached base greedy rollouts
        dev_qs = [dict(id=q["id"], prompt_ids=q["prompt_ids"], answer=q["answer"], source=q.get("source")) for q in dev_base]
        dev_base = {q["id"]: q for q in dev_base}

        def gen_dev():
            t = time.time()
            lora = publisher.publish(model) if step > 0 or args.init_adapter else None
            m = gen_metrics(roller(dev_qs, lora, greedy=True), dev_base)
            log(event="gen_dev", step=step, seconds=round(time.time() - t, 1), **m)
    if args.refresh_every:
        ref_topk = OnlineRefTopk(args.ref_k) if args.answer_kl else None
        fresh, refreshes = {}, 0
    if ref_topk and not args.refresh_every:
        m = ref_topk.meta
        assert (m["rollouts"], m["max_prefix"], m["model"]) == (os.path.abspath(args.train), args.max_prefix, args.model), \
            f"--ref-topk was built for {m['rollouts']} (max_prefix {m['max_prefix']}, {m['model']})"
        assert all((r.qid, r.j) in ref_topk.index for r in train), "--ref-topk is missing training rows"

    drift_rows = dev[: args.drift_rows]
    def evaluate_dev():
        if args.train_problems:
            log(event="eval", step=step, split="train", **evaluate_on(train))
        return evaluate(model, dev, values, score, args.eval_batch, drift_rows)
    step = 0
    if not args.skip_initial_eval:
        log(event="eval", step=0, split="dev", **evaluate_dev())
    if args.gen_dev_every:
        gen_dev()
    order, t0, previous_parts = [], time.time(), {}
    # Progressive readout: the last 10 steps' rows scored before their update (expected / argmax confidence, label).
    # Held out: fresh rollouts, or cached questions on their first pass (seen = questions already trained on).
    fresh_c, fresh_g, fresh_y, fresh_p, seen = [], [], [], [], set()
    snr = collections.defaultdict(float)  # reward_snr sums over the last 10 steps' groups (all / mixed / uniform)  # fresh_p: level marginals, as evaluate's level_p
    zero_reward = set()  # tail-fault rows (format fails only after the confidence); refreshed runs rebind it per refresh
    qlog = None if args.no_log_q else gzip.open(os.path.join(args.out, f"q_{run_name}.jsonl.gz"), "at")
    trainable = [p for p in model.parameters() if p.requires_grad]
    for step in range(1, args.steps + 1):
        if len(order) < args.problems_per_step:
            order += random.sample(group_ids, len(group_ids))
        if args.refresh_every and (step - 1) % args.refresh_every == 0:
            need = args.refresh_every * args.problems_per_step
            while len(order) < need:
                order += random.sample(group_ids, len(group_ids))
            tg = time.time()
            lora = publisher.publish(model) if step > 1 or args.init_adapter else None  # a fresh LoRA is the base (B = 0)
            refreshes += 1
            qs = roller([questions[g] for g in order[:need]], lora)
            gen_s = time.time() - tg
            dropped_fresh = collections.Counter()
            fresh, bad, zero_reward = {}, {}, set()
            for q in qs:
                rows_q = question_rows(q, token_bytes, args.max_prefix, tail_ok, dropped_fresh, qid=f"{q['id']}#r{refreshes}")
                if rows_q:
                    fresh[q["id"]] = rows_q
                    kept = {r.j for r in rows_q}
                    # Stage B: rollouts without a confidence row (format failures) still get reward 0 in their group
                    bad[q["id"]] = [(q["prompt_ids"] + ids)[: args.max_prefix] for j, ids in enumerate(q["completion_ids"])
                                    if j not in kept and ids]
                    zero_reward |= {(r.qid, r.j) for r in rows_q if not q["format_ok"][r.j]}  # tail faults
            tr = time.time()
            if ref_topk is not None:
                ref_topk.drop({r.qid for g in fresh.values() for r in g[:1]})
                for g in fresh.values():
                    ref_topk.add(model, g)
            model.train()
            log(event="refresh", step=step, refresh=refreshes, adapter_version=publisher.version if lora else 0,
                questions=len(qs), groups=len(fresh), rows=sum(map(len, fresh.values())), gen_s=round(gen_s, 1),
                ref_s=round(time.time() - tr, 1), dropped=dict(dropped_fresh), **rollout_stats(qs),
                per_q={q["id"]: [int(sum(q["correct"])), sum(em_correct(t, q["answer"]) for t in q["outputs"])] for q in qs}
                if len(qs) <= 16 else None,
                mixed=sum(0 < sum(q["correct"]) < len(q["correct"]) for q in qs))
        if args.refresh_every:
            batch = [fresh[g] for g in order[: args.problems_per_step] if g in fresh]
        else:
            batch = [groups[g] for g in order[: args.problems_per_step]]
        del order[: args.problems_per_step]
        n_rows = sum(len(g) for g in batch)
        stats = collections.defaultdict(float)
        if args.answer_pg:  # bnpo: the answer-side term is divided by the step's answer tokens (good + malformed rollouts)
            n_tok = sum(len(r.prefix) - r.start for g in batch for r in g) + \
                sum(len(p) - len(questions[g[0].qid.split("#")[0]]["prompt_ids"]) for g in batch for p in bad[g[0].qid.split("#")[0]])
            stats["answer_tokens"] = n_tok / max(len(batch), 1)  # per question
        decompose = args.grad_decomp_every and step % args.grad_decomp_every == 0
        parts = {}
        for group in batch:
            unseen = bool(args.refresh_every) or group[0].qid.split("#")[0] not in seen
            seen.add(group[0].qid.split("#")[0])
            micro = args.micro_batch or (len(group) if packed else 1)
            # Sampled: every row's draws are needed for the group baseline before any backward. With the group in
            # one forward (packed), draw from that forward's own q; split into parts, draw first in a no-grad pass.
            prepass = args.objective == "sampled" and (micro < len(group) or args.sampled_prepass)
            if prepass:
                with torch.no_grad():
                    qs = torch.cat([score(model, [r[1] for r in part]).exp().float()
                                    for part in chunks(group, len(group) if packed else args.eval_batch)])
                draws = sample_draws(qs, [row.correct for row in group], values_gpu, args.samples_per_row)
                baseline = np.mean([r for row_draws in draws for _, r in row_draws])
                if args.pg_reward == "sampled":
                    draws, baseline = rlcr_draws(draws, group, zero_reward, len(bad[group[0].qid.split("#")[0]]))
            for start in range(0, len(group), micro):
                part = group[start: start + micro]
                if args.answer_kl or args.answer_pg:
                    logps, states = score(model, [r[1] for r in part], hidden=True)  # [rows, K], hidden states
                else:
                    logps = score(model, [r[1] for r in part])
                y = torch.tensor([r[2] for r in part], device=logps.device)
                expected = exact_objective(logps, values_gpu, y[:, None])  # [rows]
                if qlog is not None:  # every rollout's q over the levels, before this step's update
                    with torch.no_grad():
                        q_ = logps.detach().exp().float(); R_ = rlcr_reward(values_gpu.float(), y[:, None].float())
                        e_ = (q_ * R_).sum(-1); v_ = (q_ * R_ ** 2).sum(-1) - e_ ** 2; lp_ = q_ @ levels_gpu.T.float()
                        for row, p_, m_, e1, v1 in zip(part, lp_.tolist(), q_.sum(-1).tolist(), e_.tolist(), v_.tolist()):
                            qlog.write(json.dumps(dict(step=step, qid=row.qid, j=row.j, y=row[2], tail_fault=(row.qid, row.j) in zero_reward,
                                                       p=[round(x, 5) for x in p_], mass=round(m_, 5), r=round(e1, 5), var_q=round(v1, 6))) + "\n")
                if args.objective == "exact":
                    loss = -expected.sum()
                else:
                    if not prepass:  # the whole group is this part
                        draws = sample_draws(logps.detach().exp().float(), [row.correct for row in group], values_gpu, args.samples_per_row)
                        baseline = np.mean([r for row_draws in draws for _, r in row_draws])
                        if args.pg_reward == "sampled":
                            draws, baseline = rlcr_draws(draws, group, zero_reward, len(bad[group[0].qid.split("#")[0]]))
                    loss = sampled_loss(logps, draws[start: start + len(part)], baseline)
                terms = [("objective", args.conf_weight * loss)] if args.conf_weight else []
                if args.answer_pg:
                    assert len(part) == len(group), "--answer-pg needs the whole group in one forward"
                    qid0 = group[0].qid.split("#")[0]
                    r = expected.detach() if args.pg_reward == "exact" else y if args.pg_reward == "accuracy" else \
                        torch.tensor([d[0][1] for d in draws[start: start + len(part)]], device=y.device, dtype=expected.dtype)
                    r = torch.where(torch.tensor([(row.qid, row.j) in zero_reward for row in part], device=r.device), 0.0, r)
                    bad_p = bad[qid0]
                    mean_r = r.sum() / (len(part) + len(bad_p))  # group mean over all its rollouts (malformed earn 0)
                    adv = r - mean_r
                    lp = answer_logp(model, states, [row.prefix for row in part], [row.start for row in part])
                    pg = -(adv * lp).sum()
                    if bad_p:
                        start_b = len(questions[qid0]["prompt_ids"])
                        states_b = packed_candidate_logps(model, bad_p, None)
                        pg = pg + mean_r * answer_logp(model, states_b, bad_p, [start_b] * len(bad_p)).sum()  # adv = -mean_r
                    scale = 1.0 if args.pg_balance == "rlcr" else n_rows / max(n_tok, 1)
                    terms.append(("pg", args.answer_pg * pg * scale))
                    stats["alpha_eff"] = args.answer_pg * (n_tok / n_rows if args.pg_balance == "rlcr" else 1.0)
                    stats["pg_reward"] += r.sum().item() / n_rows
                    stats["pg_adv_abs"] += adv.abs().sum().item() / n_rows
                    acc = torch.where(r == 0, 0.0, 0.5 * y)  # the correctness part of the reward; the rest is format + calibration
                    acc_c = acc - acc.sum() / (len(part) + len(bad_p))
                    stats["pg_adv_acc_abs"] += acc_c.abs().sum().item() / n_rows
                    stats["pg_adv_cal_abs"] += (adv - acc_c).abs().sum().item() / n_rows
                    stats["pg_malformed"] += len(bad_p) / n_rows
                    with torch.no_grad():
                        zero = torch.tensor([(row.qid, row.j) in zero_reward for row in part], device=y.device)
                        parts_snr, mixed = reward_snr(logps.exp().float(), values_gpu.float(), y.float(), zero, len(bad_p))
                        for kind in ("all", "mixed" if mixed else "uniform"):
                            snr[f"{kind}_groups"] += 1
                            for k, v in parts_snr.items():
                                snr[f"{kind}_{k}"] += v
                if args.support_floor:
                    floor = support_floor(logps, levels_gpu)
                    terms.append(("floor", floor_weight(args, step) * floor.sum()))
                    stats["support_floor"] += floor.sum().item() / n_rows
                if args.answer_kl:
                    kl = answer_kl(model, states, part, ref_topk)
                    terms.append(("kl", args.answer_kl * kl.sum()))
                    stats["answer_kl"] += kl.sum().item() / n_rows
                if decompose:  # one backward per term (graph kept), each term's gradient read off as a difference
                    for k, (name, term) in enumerate(terms):
                        before = flat_grad(trainable)
                        (term / n_rows).backward(retain_graph=k < len(terms) - 1)
                        parts[name] = parts.get(name, 0) + flat_grad(trainable) - before
                else:
                    (sum(t for _, t in terms) / n_rows).backward()
                stats["expected_reward"] += expected.sum().item() / n_rows
                if unseen:  # rows not trained on yet, before this step's update: a progressive (held-out) readout
                    with torch.no_grad():
                        q_ = logps.exp().float()
                        fresh_c += ((q_ * values_gpu).sum(-1) / q_.sum(-1).clamp_min(1e-9)).tolist()
                        fresh_g += values_gpu[logps.argmax(-1)].tolist(); fresh_y += y.tolist()
                        lp_ = (q_ @ levels_gpu.T.float()).clamp_min(1e-12); fresh_p += (lp_ / lp_.sum(-1, keepdim=True)).tolist()
                stats["mass"] += logps.exp().sum().item() / n_rows
        if decompose:
            log(event="grad", step=step, **grad_report(parts, previous_parts))
            previous_parts = parts
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            levels_log = {}
            for kind in ("all", "mixed", "uniform"):  # SNR = component signal variance / confidence-sampling noise
                if snr[f"{kind}_groups"]:
                    sfx = "" if kind == "all" else f"_{kind}"
                    levels_log[f"snr_groups{sfx}"] = int(snr[f"{kind}_groups"])
                    levels_log[f"snr_noise_var{sfx}"] = round(snr[f"{kind}_noise"] / snr[f"{kind}_groups"], 5)
                    for k in ("total", "format", "correct", "cal", "cal_within"):
                        levels_log[f"snr_{k}{sfx}"] = round(snr[f"{kind}_{k}"] / max(snr[f"{kind}_noise"], 1e-12), 4)
                        levels_log[f"snr_sig_{k}{sfx}"] = round(snr[f"{kind}_{k}"] / snr[f"{kind}_groups"], 5)
            snr.clear()
            if qlog is not None:
                qlog.flush()
            if fresh_y:
                c_, g_, y_ = map(np.array, (fresh_c, fresh_g, fresh_y))
                both = 0 < y_.mean() < 1
                stats |= dict(fresh_expected_brier=np.mean((c_ - y_) ** 2), fresh_argmax_brier=np.mean((g_ - y_) ** 2),
                              fresh_auroc=roc_auc_score(y_, c_) if both else float("nan"),
                              fresh_argmax_auroc=roc_auc_score(y_, g_) if both else float("nan"), fresh_rows=len(y_),
                              fresh_accuracy=y_.mean(), fresh_expected_mean=c_.mean(), fresh_argmax_mean=g_.mean(),
                              fresh_expected_mean_correct=c_[y_ == 1].mean() if both else float("nan"),
                              fresh_expected_mean_wrong=c_[y_ == 0].mean() if both else float("nan"),
                              fresh_argmax_mean_correct=g_[y_ == 1].mean() if both else float("nan"),
                              fresh_argmax_mean_wrong=g_[y_ == 0].mean() if both else float("nan"))
                p_ = np.array(fresh_p)
                levels_log |= {f"fresh_level_p{sfx}": [round(float(v), 5) for v in p_[m].mean(0)]
                              for sfx, m in (("", np.ones(len(y_), bool)), ("_correct", y_ == 1), ("_wrong", y_ == 0)) if m.any()}
                fresh_c, fresh_g, fresh_y, fresh_p = [], [], [], []
            log(event="train", step=step, seconds=round(time.time() - t0, 1), rows=n_rows, lr=opt.param_groups[0]["lr"], floor_beta=floor_weight(args, step),
                adam_fraction=adam_fraction(opt), **{k: round(v, 4) for k, v in stats.items()}, **levels_log)
        if step % args.eval_every == 0 or step == args.steps:
            log(event="eval", step=step, split="dev", **evaluate_dev())
        if args.gen_dev_every and (step % args.gen_dev_every == 0 or step == args.steps):
            gen_dev()
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save(os.path.join(args.out, "checkpoints", run_name, f"step{step}"))
    if test:
        log(event="eval", step=args.steps, split="test", **evaluate_on(test))
    save(os.path.join(args.out, "checkpoints", run_name))
    if qlog is not None:
        qlog.close()
    if args.refresh_every:
        publisher.close()
    if llm is not None:  # in-process vLLM's shutdown can hang holding the GPU
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
