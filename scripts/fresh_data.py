"""Count source-dataset examples RLCR never trained or evaluated on (HotpotQA distractor, Big-Math)."""
import glob, re, sys
from datasets import Dataset, load_dataset

rlcr = sys.argv[1]
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()

# Problems in RLCR's eval dumps, by text, plus HotpotQA ids (their problem text embeds the paragraphs).
eval_q, eval_ids = set(), set()
for p in glob.glob(f"{rlcr}/eval_outputs/*/*/*.arrow"):
    d = Dataset.from_file(p)
    eval_q |= {norm(q) for q in d["question" if "question" in d.column_names else "problem"]}
    if "supporting_facts" in d.column_names:
        eval_ids |= set(d["id"])

hp = load_dataset("mehuldamani/hotpot_qa")
used_ids = set(hp["train"]["id"]) | set(hp["test"]["id"]) | eval_ids
src = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
fresh = src.filter(lambda ex: ex["id"] not in used_ids and norm(ex["question"]) not in eval_q)
print(f"HotpotQA distractor train: {len(src)}; RLCR train+test+eval ids: {len(used_ids)}; "
      f"fresh (not in RLCR train/test/eval): {len(fresh)}")
print("  fresh by level:", {l: fresh["level"].count(l) for l in set(fresh["level"])})

bm = load_dataset("mehuldamani/big-math-digits")
used = {norm(p) for p in bm["train"]["problem"]} | {norm(p) for p in bm["test"]["problem"]}
src = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
numeric = re.compile(r"^-?\d+(\.\d+)?$")
cand = src.filter(lambda ex: ex["llama8b_solve_rate"] is not None and 0 < ex["llama8b_solve_rate"] < 0.75
                  and bool(numeric.match(ex["answer"].strip())))
fresh = cand.filter(lambda ex: norm(ex["problem"]) not in used and norm(ex["problem"]) not in eval_q)
print(f"Big-Math: {len(src)}; solve rate in (0, 0.75) with plain numeric answer (big-math-digits rule): {len(cand)}; "
      f"big-math-digits train+test: {len(used)}; fresh: {len(fresh)}")
print("  fresh by source:", sorted({s: fresh["source"].count(s) for s in set(fresh["source"])}.items(), key=lambda x: -x[1]))
