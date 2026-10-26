"""Which tag name for a pre-answer solve-rate head? Step-0 readout of the untrained model for each candidate tag.

For each train question, insert `[instruction] <tag>` right after the prompt (where the assistant turn starts) and
score the 11 values 0.0 .. 1.0 through `</tag>` in three spellings (` 0.7 </tag>`, `0.7</tag>`, `\\n0.7\\n</tag>`)
with one trie per tag. The cached rollouts are not involved beyond their labels: k of n correct per question (exact
match of each rollout's last <answer>, format not checked) is the solve rate the head should forecast.

Per tag and instruction: mean mass on the 33 candidates, entropy of the renormalised level distribution, mean
expected value, and its Spearman correlation with k/n and AUROC for k/n >= 0.5. Train questions only (dev stays
clean). Writes OUT/tag_probe.json and prints a table.

usage: tag_probe.py ROLLOUTS_JSONL [--questions N] [--model M] [--out DIR]
"""
import argparse, json, os, sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rlcr_exact.grading import em_correct
from rlcr_exact.trie_score import build_trie, packed_candidate_logps

TAGS = ["solve_rate", "p_correct", "expected_accuracy", "success_probability", "prior_confidence", "confidence",
        "difficulty"]
INSTRUCTION = "Before answering, I estimate the probability that my final answer will be correct.\n"
VALUES = [f"0.{d}" for d in range(10)] + ["1.0"]

ap = argparse.ArgumentParser()
ap.add_argument("rollouts")
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--questions", type=int, default=500)
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
enc = lambda s: tok(s, add_special_tokens=False).input_ids

questions = []
for line in open(a.rollouts):
    q = json.loads(line)
    questions.append((q["prompt_ids"], sum(em_correct(t, q["answer"]) for t in q["outputs"]) / len(q["outputs"])))
    if len(questions) == a.questions:
        break
rate = np.array([r for _, r in questions])
print(f"{len(questions)} questions, mean solve rate {rate.mean():.3f}, all-wrong {np.mean(rate == 0):.2f}, all-right {np.mean(rate == 1):.2f}")
print("prompt tail:", repr(tok.decode(questions[0][0][-12:])))

tries, levels = {}, {}
for tag in TAGS:
    seqs, lv = [], []
    for spell in (" {v} </{t}>", "{v}</{t}>", "\n{v}\n</{t}>"):
        for i, v in enumerate(VALUES):
            seqs.append(enc(spell.format(v=v, t=tag))); lv.append(i)
    tries[tag], levels[tag] = build_trie(seqs), torch.tensor(lv)
variants = [(tag, instr) for tag in TAGS for instr in (False, True)]
tag_ids = {(tag, instr): enc((INSTRUCTION if instr else "") + f"<{tag}>") for tag, instr in variants}
print("inserted:", {f"{t}{'+i' if i else ''}": len(ids) for (t, i), ids in tag_ids.items()})

grid = torch.linspace(0, 1, 11)
stats = {v: {"mass": [], "entropy": [], "expected": []} for v in variants}
with torch.no_grad():
    for n, (prompt, _) in enumerate(questions):
        for tag in TAGS:
            rows = [prompt + tag_ids[(tag, False)], prompt + tag_ids[(tag, True)]]
            lp = packed_candidate_logps(model, rows, tries[tag]).float().cpu()  # [2, 33]
            for instr, row in zip((False, True), lp):
                p = torch.zeros(11).index_add_(0, levels[tag], row.exp())
                mass = p.sum().item(); pn = p / max(mass, 1e-12)
                s = stats[(tag, instr)]
                s["mass"].append(mass); s["entropy"].append(-(pn * pn.clamp_min(1e-12).log()).sum().item())
                s["expected"].append((pn * grid).sum().item())
        if n % 100 == 0:
            print(f"question {n}", flush=True)

out, majority = [], rate >= 0.5
print(f"\n{'tag':<22}{'instr':<7}{'mass':>8}{'entropy':>9}{'E[v]':>7}{'spearman':>10}{'auroc':>7}")
for (tag, instr), s in stats.items():
    e = np.array(s["expected"])
    rec = dict(tag=tag, instruction=instr, mass=float(np.mean(s["mass"])), mass_median=float(np.median(s["mass"])),
               entropy=float(np.mean(s["entropy"])), expected_mean=float(e.mean()), expected_sd=float(e.std()),
               spearman=float(spearmanr(e, rate).statistic), auroc=float(roc_auc_score(majority, e)) if 0 < majority.mean() < 1 else None)
    out.append(rec)
    print(f"{tag:<22}{'yes' if instr else 'no':<7}{rec['mass']:>8.3f}{rec['entropy']:>9.2f}{rec['expected_mean']:>7.2f}"
          f"{rec['spearman']:>10.3f}{rec['auroc']:>7.3f}")
os.makedirs(a.out, exist_ok=True)
json.dump(dict(questions=len(questions), mean_rate=float(rate.mean()), instruction=INSTRUCTION, results=out),
          open(os.path.join(a.out, "tag_probe.json"), "w"), indent=1)
