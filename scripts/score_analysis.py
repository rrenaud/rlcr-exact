"""Score regenerated analyses (regen_analysis.py output) at the confidence tag with the trie scorer.

For each row whose generation contains "<confidence", the prefix is the prompt + cached answer prefix + generated
tokens through that tag's `confidence` token. Candidates continue `>\\n` d `.` d `\\n</confidence>` then an end
token (<|im_end|> or <|endoftext|>): 11 values x 2 ends, plus each value without its end (so p(end | value) is
known). Writes per-row JSON (log-probs of all candidates) and the final hidden state at the prefix (fp16) for a
linear probe.

usage: score_analysis.py --model M --regen R.jsonl --out OUT_PREFIX      (pytorch env, gpuq payload)
writes OUT_PREFIX.jsonl and OUT_PREFIX.hidden.npy (rows in the same order)
"""
import argparse, collections, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rlcr_exact.trie_score import build_trie, packed_candidate_logps

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-correctness")
ap.add_argument("--regen", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tid = tok.convert_tokens_to_ids
VALUES = [f"0.{d}" for d in range(10)] + ["1.0"]
ENDS = ["<|im_end|>", "<|endoftext|>"]
seqs, meta = [], []  # meta: (value, end or None)
for v in VALUES:
    body = tok.encode(">\n" + v + "\n</confidence>", add_special_tokens=False)
    assert tok.decode(body[:1]) == ">\n" and len(body) == 8, (v, [tok.decode([t]) for t in body])
    for e in ENDS:
        seqs.append(body + [tid(e)]); meta.append((float(v), e))
    seqs.append(body); meta.append((float(v), None))
trie = build_trie(seqs)
CONF = tid("confidence")

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()


def tag_cut(gen_ids):
    """Index just past the `confidence` token of the last opening <confidence tag in gen_ids, or None."""
    for i in range(len(gen_ids) - 1, 0, -1):
        if gen_ids[i] == CONF:
            prev = tok.decode([gen_ids[i - 1]])
            if prev.endswith("<") and not prev.endswith("</"):
                return i + 1
    return None


rows = [json.loads(l) for l in open(a.regen)]
groups = collections.defaultdict(list)  # rows sharing a prompt head score in one packed forward
for k, r in enumerate(rows):
    groups[(r["prompt"], r["split"], r["id"])].append(k)
hidden = np.zeros((len(rows), model.config.hidden_size), np.float16)
scored = [None] * len(rows)
t0 = time.time()
with torch.no_grad():
    for gi, ks in enumerate(groups.values()):
        todo = []
        for k in ks:
            r = rows[k]
            c = tag_cut(r["gen_ids"])
            if c is None:
                scored[k] = dict(has_tag=False)
                continue
            todo.append((k, r["prompt_ids"] + r["gen_ids"][:c]))
        for i in range(0, len(todo), 8):
            part = todo[i:i + 8]
            logps, hs = packed_candidate_logps(model, [p for _, p in part], trie, hidden=True)
            for (k, p), lp, h in zip(part, logps.float().cpu().tolist(), hs):
                scored[k] = dict(has_tag=True, logps=[round(x, 5) for x in lp])
                hidden[k] = h[-1].float().cpu().numpy().astype(np.float16)
        if gi % 200 == 0:
            print(json.dumps(dict(event="progress", groups=gi, of=len(groups), seconds=round(time.time() - t0))), flush=True)

with open(a.out + ".jsonl", "w") as f:
    for r, s in zip(rows, scored):
        f.write(json.dumps(dict(prompt=r["prompt"], split=r["split"], id=r["id"], j=r["j"], correct=r["correct"],
                                gen=r["gen"], gen_tokens=len(r["gen_ids"]), finish=r["finish"], **s)) + "\n")
json.dump(dict(values=[m[0] for m in meta], ends=[m[1] for m in meta]), open(a.out + ".candidates.json", "w"))
np.save(a.out + ".hidden.npy", hidden)
print(json.dumps(dict(event="done", rows=len(rows), tagged=sum(bool(s and s.get("has_tag")) for s in scored),
                      seconds=round(time.time() - t0))), flush=True)
