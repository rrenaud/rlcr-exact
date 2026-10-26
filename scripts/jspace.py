"""J-space readouts for rows the confidence head gets right and wrong: what the Jacobian lens reads out at the
<confidence> position and over the <analysis> span, for test rows with low and high Brier error, for the RLVR
baseline and the best exact adapter.

Lenses (all one linear map per layer into the final-layer basis, then the model's own final norm + unembedding):
  logit: the logit lens (identity map), the baseline.
  j:     the Jacobian lens, J_l = E[dh_final / dh_l] over web text (Gurnee et al. 2026, "Verbalizable Representations
         Form a Global Workspace in Language Models"; fitted with anthropics/jacobian-lens).
  r:     the R-lens (Blank, Bhatia & Nanda 2026): the same fit with layerwise-relevance rules in the backward pass, so
         noise does not compound through the norms and gated MLPs on the way to early layers. Rules as released in
         safety-research/jpp_lens for dense Qwen2 (lrp_mode "rlens"): every RMSNorm's scale detached (LN rule), the
         SiLU gate detached (identity rule), the gate x up product's gradient halved to each branch (half rule).
         Attention and linear layers are unchanged.

  fit:     python scripts/jspace.py fit --kind j|r --out LENS.pt [--n-prompts 48]
  readout: python scripts/jspace.py readout --lens j=J.pt --lens r=R.pt --test ROLLOUTS [--adapter DIR] --out DIR
  over training: python scripts/jspace.py readout-steps --lens j=J.pt --adapter base=- --adapter s50=DIR ... --test ROLLOUTS --out DIR

Readout groups (per model, from its own stated confidence c and RLCR's exact-match label y): low (c - y)^2 <= 0.04;
over: y = 0 and c >= 0.7; under: y = 1 and c <= 0.3. The baseline's c is the number in its cached greedy text (it
mostly leaves rlcr11's list); the adapter's is its greedy rlcr11 level on the cached prefix, as the trainer reports.
Saved per (model, lens, layer, site, group): the mean log-softmax over the vocabulary (fp16), plus each row's top-20
tokens at the confidence site.
"""
import argparse, collections, json, math, os, re, sys, time
from contextlib import contextmanager, nullcontext
from types import MethodType

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

MODEL = "mehuldamani/hotpot-v2-correctness-7b"
LAYERS = list(range(6, 27, 2))  # block outputs 6, 8, ..., 26; the target is block 27 (the final residual)
GROUPS = ("low", "over", "under", "all")


def load(model_name, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    if adapter:
        from peft import PeftModel
        hf = PeftModel.from_pretrained(hf, adapter).merge_and_unload()
    return tok, hf.eval()


# --- R-lens rules (dense Qwen2), after safety-research/jpp_lens src/workspace_lens/lrp/lrp.py -----------------------

def _rmsnorm_ln_rule(self, hidden_states):
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    return self.weight * (x * scale).to(input_dtype)


def _mlp_identity_half_rule(self, x):
    z = self.gate_proj(x)
    product = z * torch.sigmoid(z).detach() * self.up_proj(x)
    return self.down_proj(0.5 * product + 0.5 * product.detach())


@contextmanager
def rlens_rules(hf):
    """Shadow the forward of every Qwen2RMSNorm and Qwen2MLP instance with its LRP-rule version (same values, the
    relevance backward); restored on exit."""
    patched = []
    for name, m in hf.named_modules():
        cls = type(m).__name__
        if cls == "Qwen2RMSNorm":
            m.forward = MethodType(_rmsnorm_ln_rule, m); patched.append(m)
        elif cls == "Qwen2MLP":
            assert getattr(m.act_fn, "__class__").__name__ in ("SiLU", "SiLUActivation"), m.act_fn
            m.forward = MethodType(_mlp_identity_half_rule, m); patched.append(m)
    assert patched, "no Qwen2 modules found"
    try:
        yield len(patched)
    finally:
        for m in patched:
            del m.forward


def web_prompts(n, min_chars=1500, seed=0):
    """n documents from FineWeb-Edu (sample-10BT), the pretraining-like text the J-lens paper fits on."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
    out = []
    for ex in ds:
        if len(ex["text"]) >= min_chars:
            out.append(ex["text"])
        if len(out) == n:
            return out


def cmd_fit(args):
    import jlens
    tok, hf = load(args.model)
    lm = jlens.from_hf(hf, tok)
    prompts = web_prompts(args.n_prompts)
    t0 = time.time()
    with (rlens_rules(hf) if args.kind == "r" else nullcontext()) as n:
        print(f"fit {args.kind}: {args.n_prompts} prompts, layers {LAYERS}, patched {n}", flush=True)
        lens = jlens.fit(lm, prompts, source_layers=LAYERS, dim_batch=args.dim_batch, max_seq_len=128,
                         checkpoint_path=args.out + ".ckpt", checkpoint_every=4)
    lens.save(args.out)
    print(f"saved {args.out} ({time.time() - t0:.0f} s)", flush=True)


# --- readout ---------------------------------------------------------------------------------------------------------

def stated_confidence(text):
    m = re.findall(r"<confidence>(.*?)</confidence>", text, re.DOTALL)
    try:
        return min(max(float(m[-1].strip()), 0.0), 1.0) if m else None
    except ValueError:
        return None


def analysis_span(tok, prefix, start):
    """Token positions of the last <analysis> ... </analysis> body in prefix[start:] (empty if absent)."""
    pieces = [tok.decode([t]) for t in prefix[start:]]
    ends = np.cumsum([len(p) for p in pieces])
    text = "".join(pieces)
    a, b = text.rfind("<analysis>"), text.rfind("</analysis>")
    if a < 0 or b < a:
        return []
    a += len("<analysis>")
    return [start + i for i, e in enumerate(ends) if a < e and e - len(pieces[i]) < b]


def group_of(c, y):
    out = ["all"]
    if (c - y) ** 2 <= 0.04:
        out.append("low")
    if y == 0 and c >= 0.7:
        out.append("over")
    if y == 1 and c <= 0.3:
        out.append("under")
    return out


@torch.no_grad()
def cmd_readout(args):
    import jlens
    from jlens.hooks import ActivationRecorder
    from train_frozen import load_rows, make_scorer  # puts the conf-vote repo's logprob_spans on the path
    from logprob_spans import make_token_bytes
    lenses = {"logit": None}
    for spec in args.lens:
        k, path = spec.split("=", 1)
        lenses[k] = jlens.JacobianLens.load(path)
    tag = "exact" if args.adapter else "base"
    tok, hf = load(args.model, args.adapter)
    lm = jlens.from_hf(hf, tok)
    texts = {}
    for line in open(args.test):
        q = json.loads(line)
        for j, t in enumerate(q["outputs"]):
            texts[(q["id"], j)] = t
    rows = load_rows(args.test, make_token_bytes(tok), 4608, tail_ok=True)[0][: args.limit]
    values, score, _ = make_scorer("rlcr11", tok)
    Js = {k: {l: L.jacobians[l].to("cuda") for l in LAYERS} for k, L in lenses.items() if L is not None}
    sums = collections.defaultdict(lambda: torch.zeros(hf.config.vocab_size, device="cuda"))
    counts = collections.Counter()
    per_row = []
    t0 = time.time()
    for i, row in enumerate(rows):
        if args.adapter:
            logps, picks = score(hf, [row.prefix], greedy=True)
            q = logps[0].exp().float()
            c = values[picks[0]].item() if picks[0] is not None else ((q * values.to(q.device)).sum() / q.sum()).item()
        else:
            c = stated_confidence(texts[(row.qid, row.j)])
            if c is None:
                continue
        y = float(row.correct)
        groups = group_of(c, y)
        span = analysis_span(tok, row.prefix, row.start)
        ids = torch.tensor([row.prefix], device="cuda")
        with ActivationRecorder(lm.layers, at=LAYERS) as rec:
            lm.forward(ids)
            acts = {l: rec.activations[l][0].float() for l in LAYERS}
        rec_top = {}
        for kind in lenses:
            for l in LAYERS:
                sites = {"conf": acts[l][-1:]}
                if span:
                    sites["analysis"] = acts[l][span]
                for site, h in sites.items():
                    if kind != "logit":
                        h = h @ Js[kind][l].T
                    lp = lm.unembed(h).float().log_softmax(-1).mean(0)
                    for g in groups:
                        sums[(kind, l, site, g)] += lp
                        counts[(kind, l, site, g)] += 1
                    if site == "conf":
                        top = lp.topk(20)
                        rec_top[f"{kind}/{l}"] = top.indices.tolist()
        per_row.append(dict(qid=row.qid, j=row.j, y=y, c=c, groups=groups[1:], n_analysis=len(span), top=rec_top))
        if i % 50 == 0:
            print(f"{tag} row {i}/{len(rows)} {time.time() - t0:.0f} s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"means": {k: (v / counts[k]).half().cpu() for k, v in sums.items()}, "counts": dict(counts),
                "layers": LAYERS, "lenses": list(lenses)}, os.path.join(args.out, f"means_{tag}.pt"))
    with open(os.path.join(args.out, f"rows_{tag}.jsonl"), "w") as f:
        for r in per_row:
            f.write(json.dumps(r) + "\n")
    c_, y_ = np.array([r["c"] for r in per_row]), np.array([r["y"] for r in per_row])
    print(json.dumps(dict(model=tag, rows=len(per_row), brier=float(np.mean((c_ - y_) ** 2)), accuracy=float(y_.mean()),
                          groups={g: counts[("logit", LAYERS[0], "conf", g)] for g in GROUPS})), flush=True)


# Scalar probes at the confidence site: logsumexp of the lens log-probs over one token set minus another (oriented so
# higher = more confident), each set = the first token of every word with and without a leading space, lower and
# capitalised. cert-hedge compares the mean log-prob of the two lists instead.
PROBES = {"right-wrong": (["right"], ["wrong"]), "correct-incorrect": (["correct"], ["incorrect"]),
          "yes-no": (["yes"], ["no"]), "sure-unsure": (["sure"], ["unsure"]), "true-false": (["true"], ["false"])}
CERT_WORDS = ["sure", "certain", "confident", "definitely", "absolutely", "clearly", "correct", "yes", "true", "verified"]
HEDGE_WORDS = ["unsure", "uncertain", "unlikely", "maybe", "perhaps", "unknown", "doubt", "unclear", "partial", "possibly"]


def probe_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()):
            ids.add(tok(v, add_special_tokens=False).input_ids[0])
    return sorted(ids)


def probe_sets(tok):
    sets = {k: (probe_ids(tok, a), probe_ids(tok, b), "lse") for k, (a, b) in PROBES.items()}
    sets["cert-hedge"] = (probe_ids(tok, CERT_WORDS), probe_ids(tok, HEDGE_WORDS), "mean")
    return sets


def probe_values(lp, sets):
    """lp [vocab] log-probs -> {probe: value}."""
    out = {}
    for k, (a, b, how) in sets.items():
        f = (lambda ix: lp[ix].logsumexp(0)) if how == "lse" else (lambda ix: lp[ix].mean())
        out[k] = round((f(torch.tensor(a, device=lp.device)) - f(torch.tensor(b, device=lp.device))).item(), 4)
    return out

@torch.no_grad()
def cmd_readout_steps(args):
    """Confidence-site readout of several checkpoints of one run on the same rows: each TAG=DIR adapter (TAG "base"
    with DIR "-" is the untrained model) is merged into the base weights, read out, and unmerged. Every model's c is
    its greedy rlcr11 level (the listed-mass expectation where greedy leaves the list), so steps compare like for like.
    Writes rows_TAG.jsonl per checkpoint: c, its expected level, y, the top-20 tokens per lens at --layers, and the
    PROBES values per lens at every LAYERS layer (probe_sets.json lists each probe's tokens)."""
    import jlens
    from jlens.hooks import ActivationRecorder
    from peft import PeftModel
    from train_frozen import load_rows, make_scorer
    from logprob_spans import make_token_bytes
    layers = [int(l) for l in args.layers.split(",")]  # top-20 tokens here; probes at every LAYERS layer
    read_at = sorted(set(layers) | set(LAYERS))
    lenses = {"logit": None}
    for spec in args.lens:
        k, path = spec.split("=", 1)
        lenses[k] = jlens.JacobianLens.load(path)
    tok, hf = load(args.model)
    rows = load_rows(args.test, make_token_bytes(tok), 4608, tail_ok=True)[0][: args.limit]
    values, score, _ = make_scorer("rlcr11", tok)
    Js = {k: {l: L.jacobians[l].to("cuda") for l in read_at} for k, L in lenses.items() if L is not None}
    sets = probe_sets(tok)
    os.makedirs(args.out, exist_ok=True)
    json.dump({k: [[tok.decode([i]) for i in a], [tok.decode([i]) for i in b], how] for k, (a, b, how) in sets.items()},
              open(os.path.join(args.out, "probe_sets.json"), "w"), indent=1)
    original = {n: p.detach().to("cpu", copy=True) for n, p in hf.named_parameters()}
    for spec in args.adapter:
        tag, path = spec.split("=", 1)
        model = hf if path == "-" else PeftModel.from_pretrained(hf, path).merge_and_unload()
        lm = jlens.from_hf(model, tok)
        t0, out = time.time(), []
        for row in rows:
            logps, picks = score(model, [row.prefix], greedy=True)
            q = logps[0].exp().float()
            expected = ((q * values.to(q.device)).sum() / q.sum().clamp_min(1e-9)).item()
            c = values[picks[0]].item() if picks[0] is not None else expected
            with ActivationRecorder(lm.layers, at=read_at) as rec:
                lm.forward(torch.tensor([row.prefix], device="cuda"))
                acts = {l: rec.activations[l][0][-1:].float() for l in read_at}
            top, probe = {}, {}
            for kind in lenses:
                for l in read_at:
                    h = acts[l] if kind == "logit" else acts[l] @ Js[kind][l].T
                    lp = lm.unembed(h).float()[0].log_softmax(-1)
                    if l in layers:
                        top[f"{kind}/{l}"] = lp.topk(20).indices.tolist()
                    probe[f"{kind}/{l}"] = probe_values(lp, sets)
            out.append(dict(qid=row.qid, j=row.j, y=float(row.correct), c=c, expected=expected, mass=q.sum().item(), top=top, probe=probe))
        with open(os.path.join(args.out, f"rows_{tag}.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        c_, y_ = np.array([r["c"] for r in out]), np.array([r["y"] for r in out])
        print(json.dumps(dict(tag=tag, rows=len(out), brier=float(np.mean((c_ - y_) ** 2)), seconds=round(time.time() - t0))), flush=True)
        for n, p in hf.named_parameters():  # unmerge: back to the untrained weights for the next checkpoint
            p.copy_(original[n].to(p.device))
        if hasattr(hf, "peft_config"):
            del hf.peft_config


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--kind", choices=["j", "r"], required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--n-prompts", type=int, default=48)
    f.add_argument("--dim-batch", type=int, default=32)
    r = sub.add_parser("readout")
    r.add_argument("--lens", action="append", default=[], help="KIND=PATH")
    r.add_argument("--test", required=True)
    r.add_argument("--adapter", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", required=True)
    s = sub.add_parser("readout-steps")
    s.add_argument("--lens", action="append", default=[], help="KIND=PATH")
    s.add_argument("--adapter", action="append", required=True, help="TAG=DIR, or base=- for the untrained model")
    s.add_argument("--layers", default="18,20,22,24,26")
    s.add_argument("--test", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--out", required=True)
    for p in (f, r, s):
        p.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    {"fit": cmd_fit, "readout": cmd_readout, "readout-steps": cmd_readout_steps}[args.cmd](args)


if __name__ == "__main__":
    main()
