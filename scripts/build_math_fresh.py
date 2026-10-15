"""Fresh Big-Math problems for continued training of RLCR-math, drawn with big-math-digits' own rule.

big-math-digits is a uniform sample of Big-Math problems with 0 < llama8b_solve_rate < 0.75 and a plain
numeric answer (source mix and mean solve rate match the pool). We sample from the remainder: problems not
in big-math-digits train/test and not in any RLCR eval set. Writes data/math_fresh_{train,dev,test}.jsonl;
test is big-math-digits' own test split, for comparison with the released evaluation.
"""
import glob, json, re, sys
from datasets import Dataset, load_dataset

rlcr, n_train, n_dev = sys.argv[1], 4000, 1000
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
numeric = re.compile(r"^-?\d+(\.\d+)?$")

eval_q = set()
for p in glob.glob(f"{rlcr}/eval_outputs/*/*/*.arrow"):
    d = Dataset.from_file(p)
    eval_q |= {norm(q) for q in d["question" if "question" in d.column_names else "problem"]}
bm = load_dataset("mehuldamani/big-math-digits")
used = {norm(p) for split in bm.values() for p in split["problem"]} | eval_q

src = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
fresh = src.filter(lambda ex: ex["llama8b_solve_rate"] is not None and 0 < ex["llama8b_solve_rate"] < 0.75
                   and bool(numeric.match(ex["answer"].strip())) and norm(ex["problem"]) not in used)
seen, rows = set(), []
for ex in fresh.shuffle(seed=0):  # drop duplicate problems within the pool
    if norm(ex["problem"]) not in seen:
        seen.add(norm(ex["problem"])); rows.append(ex)
splits = {"train": rows[:n_train], "dev": rows[n_train:n_train + n_dev], "test": list(bm["test"])}
for name, split in splits.items():
    with open(f"data/math_fresh_{name}.jsonl", "w") as f:
        for i, ex in enumerate(split):
            f.write(json.dumps({"id": f"{name}-{i}", "problem": ex["problem"], "answer": ex["answer"],
                                "source": ex["source"], "llama8b_solve_rate": ex["llama8b_solve_rate"]}) + "\n")
    print(name, len(split))
print("fresh pool after dedup:", len(rows))
