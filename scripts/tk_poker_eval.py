"""Zero-shot poker equity eval on Tinker (sampling only, no training client): Inkling-2-Mini free by default.

Same spots and grading as scripts/poker_eval.py, but in rlcr11-tinker's setup: tml_v0 renderer, native thinking at
effort 0.0 standing in for RLCR's <think>, the relaxed tabc format (tk/cands.py FORMAT), RLCR's tabc_long system
prompt and "\\n\\nPROBLEM: ...\\n\\n" user turn. Text confidence only (no candidate logprob scoring; Inkling's
confidences are mostly off the rlcr11 grid anyway). Rollouts are NOT seeded: SamplingParams(seed=...) makes all
num_samples identical on Tinker.

Writes OUT/poker_eval/<tag>.jsonl.gz with the same per-completion fields as poker_eval.py (no q / conf_exact).
usage (tinker env): tk_poker_eval.py --eval data/poker_v1_claim/dev.jsonl --tag inkling_claim --out DIR [--k 4]
"""
import argparse, gzip, json, os, re, sys, time, types as pytypes
import numpy as np
import tinker
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer

FORMAT = re.compile(r"\s*(<think>.*?</think>\s*)?<answer>(.*?)</answer>\s*<analysis>.*?</analysis>\s*<confidence>(.*?)</confidence>\s*\Z", re.S)

ap = argparse.ArgumentParser()
ap.add_argument("--eval", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="thinkingmachines/Inkling-2-Mini:peft:65536:free")
ap.add_argument("--k", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--effort", type=float, default=0.0)
ap.add_argument("--think-budget", type=int, default=0,
                help="budget forcing (knob, 0 = off): stop native thinking at N tokens, append a commit line, close the "
                     "thinking channel (<|end_message|><|message_model|><|content_text|>) and sample the answer")
ap.add_argument("--answer-tokens", type=int, default=1024)
ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
a = ap.parse_args()
sys.path.insert(0, a.rlcr)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from system_prompts import get_sys_prompt
from reward_fns import accuracy_reward
from rlcr_exact.grading import normalize_answer

out = os.path.join(a.out, "poker_eval"); os.makedirs(out, exist_ok=True)
base = a.model.split(":")[0]
tok = get_tokenizer(base); ren = get_renderer("tml_v0", tok, model_name=base)
sysp = get_sys_prompt("tabc_long")
spots = [json.loads(l) for l in open(a.eval)][: a.limit]
sampler = tinker.ServiceClient().create_sampling_client(base_model=a.model)
params = types.SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, stop=ren.get_stop_sequences())
t0 = time.time()
futs, prompts = [], []
if a.think_budget:
    params = types.SamplingParams(max_tokens=a.think_budget, temperature=a.temperature, stop=ren.get_stop_sequences())
for s in spots:
    p = ren.build_generation_prompt([dict(role="system", content=sysp), dict(role="user", content=f"\n\nPROBLEM: {s['problem']}\n\n")],
                                    effort=a.effort).to_ints()
    prompts.append(p)
    futs.append(sampler.sample(prompt=types.ModelInput.from_ints(p), num_samples=a.k, sampling_params=params))
END_THINK = [200010, 200001, 200004]  # <|end_message|><|message_model|><|content_text|> (Inkling tml_v0)
COMMIT = tok.encode("\n\nI'm out of time, so I'll commit to my best estimate now.", add_special_tokens=False)
aparams = types.SamplingParams(max_tokens=a.answer_tokens, temperature=a.temperature, stop=ren.get_stop_sequences())


def finish(prompt, seq):
    """(tokens, stop_reason, forced): the sequence, or for a budget-cut thinking trace, trace + commit + answer."""
    toks = list(seq.tokens)
    if not a.think_budget or seq.stop_reason == "stop" or 200004 in toks:  # finished, or already in the text channel
        return toks, seq.stop_reason, False
    head = toks + COMMIT + END_THINK
    r = sampler.sample(prompt=types.ModelInput.from_ints(prompt + head), num_samples=1, sampling_params=aparams).result()
    return head + list(r.sequences[0].tokens), r.sequences[0].stop_reason, True


recs = []
for i, (s, f, prompt) in enumerate(zip(spots, futs, prompts)):  # forced answers run serially per spot (small)
    for j, seq0 in enumerate(f.result().sequences):
        toks, stop, forced = finish(prompt, seq0)
        seq = pytypes.SimpleNamespace(tokens=toks, stop_reason=stop) if forced else seq0
        msg, _ = ren.parse_response(toks)
        content = msg["content"] if isinstance(msg["content"], str) else "".join(
            x.get("text", "") for x in msg["content"] if isinstance(x, dict))
        m = FORMAT.match(content) if seq.stop_reason == "stop" else None
        ans = normalize_answer(m.group(2)) if m else ""
        conf = None
        if m:
            try:
                conf = float(m.group(3)); conf = conf if 0 <= conf <= 1 else None
            except ValueError:
                pass
        ok = m is not None and conf is not None
        y = float(accuracy_reward("tabc", [[{"role": "assistant", "content": content}]], [s["answer"]], ["hotpot"])[0]) if ok else 0.0
        # claim: confidence is defined as P(win) whatever the answer; rlcr: P(answer right)
        eq = None if conf is None else conf if s.get("fmt") == "claim" else None if ans not in ("yes", "no") else (conf if ans == "yes" else 1 - conf)
        recs.append(dict(id=s["id"], j=j, street=s["street"], villain_kind=s["villain_kind"], p_win=s["p_win"], p_tie=s["p_tie"],
                         outcome=s["outcome"], gold=s["answer"], answer=ans[:40], format_ok=float(ok), correct=y,
                         truncated=seq.stop_reason != "stop", conf_text=conf, conf_exact=None, equity_text=eq, equity_exact=None,
                         tokens=len(toks), forced=forced, text=content if j == 0 else None))
    if i % 50 == 0:
        print(json.dumps(dict(event="progress", spots=i + 1, elapsed=round(time.time() - t0))), flush=True)
with gzip.open(os.path.join(out, f"{a.tag}.jsonl.gz"), "wt") as fo:
    for r in recs:
        fo.write(json.dumps(r) + "\n")
g = [r for r in recs if r["equity_text"] is not None]
summ = dict(tag=a.tag, model=a.model, k=a.k, completions=len(recs), format_ok=float(np.mean([r["format_ok"] for r in recs])),
            truncated=float(np.mean([r["truncated"] for r in recs])), tokens=float(np.mean([r["tokens"] for r in recs])),
            mse_equity_text_vs_oracle=float(np.mean([(r["equity_text"] - r["p_win"]) ** 2 for r in g])) if g else None,
            elapsed=round(time.time() - t0))
json.dump(summ, open(os.path.join(out, f"{a.tag}.json"), "w"), indent=1)
print(json.dumps(dict(event="done", **summ)), flush=True)
os._exit(0)
