"""Turn regen_analysis.py output for one prompt into the cached-rollout schema train_frozen.py and ref_topk.py read.

Each completion is the cached <think>...</answer> prefix plus the regenerated analysis and confidence (generation
stopped at </confidence>; no end token, so the scorer's candidates supply the stop). format_ok is RLCR's tabc check
on that text; correct is the prompt-independent label regen_analysis.py computed. Rows are grouped per question.

usage: regen_to_rollouts.py REGEN.jsonl PROMPT OUT.jsonl
"""
import collections, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rlcr_exact.grading import TABC
from transformers import AutoTokenizer

regen, prompt, out = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
answers = {}
for split in ("train", "dev", "test"):
    for line in open(os.path.join(HERE, "..", "data", f"math_fresh_{split}.jsonl")):
        q = json.loads(line); answers[q["id"]] = q["answer"]
tok = None
groups = collections.OrderedDict()
for line in open(regen):
    r = json.loads(line)
    if r["prompt"] != prompt:
        continue
    head = r["prompt_ids"][: len(r["prompt_ids"]) - r["prefix_len"]]
    g = groups.setdefault(r["id"], dict(id=r["id"], answer=answers[r["id"]], prompt_ids=head, outputs=[], completion_ids=[],
                                        format_ok=[], correct=[], truncated=[]))
    assert g["prompt_ids"] == head, r["id"]
    ids = r["prompt_ids"][len(head):] + r["gen_ids"]
    if tok is None:
        tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL", "mehuldamani/big-math-digits-v2-correctness"))
    text = tok.decode(ids)
    g["outputs"].append(text); g["completion_ids"].append(ids)
    g["format_ok"].append(1.0 if re.match(TABC, text, re.DOTALL | re.MULTILINE) else 0.0)
    g["correct"].append(float(r["correct"])); g["truncated"].append(r["finish"] == "length")
with open(out, "w") as f:
    for g in groups.values():
        f.write(json.dumps(g) + "\n")
n = sum(len(g["outputs"]) for g in groups.values())
print(json.dumps(dict(prompt=prompt, questions=len(groups), rows=n,
                      format_ok=sum(sum(g["format_ok"]) for g in groups.values()) / max(n, 1))))
