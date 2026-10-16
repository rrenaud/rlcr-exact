"""Fresh HotpotQA-Modified questions for continued training of RLCR-hotpot, built like mehuldamani/hotpot_qa.

HotpotQA-Modified (paper, appendix B.1) starts from the HotpotQA distractor setting (10 paragraphs, 2 of them
gold) and removes two paragraphs: both gold, one gold and one distractor, or two distractors, in about equal
thirds, so every question keeps 8. The prompt lists the kept paragraphs in their original order.

`check` rebuilds every mehuldamani/hotpot_qa train row's problem string from the original question, its
context and its removed_titles, and reports how many match byte for byte. `build` draws fresh questions
(distractor-train ids not in mehuldamani/hotpot_qa train/test or any RLCR eval set), picks gold_removed in
{0, 1, 2} uniformly and the removed paragraphs at random (seed 0), and writes data/hotpot_fresh_{train,dev}.jsonl,
plus data/hotpot_fresh_test.jsonl: mehuldamani/hotpot_qa's own 500-question test split (the paper's in-domain
HotpotQA-Modified evaluation), for comparison with the released numbers.

Their rows remove gold_removed gold paragraphs plus 2 - gold_removed distractors. Quirks we do not copy: in 9%
of the gold_removed = 2 rows removed_titles repeats gold titles (one per supporting sentence; same removal),
and 79 rows come from questions with fewer than 10 paragraphs. We keep 10-paragraph, 2-gold questions only.

usage: build_hotpot_fresh.py check | build RLCR_DIR [N_TRAIN N_DEV]
"""
import glob, json, random, sys

from datasets import Dataset, load_dataset

INSTRUCTION = ("Your answer will be verified with exact match score. To ensure correct verification, only provide "
               "the answer within the <answer> </answer> tags. Do not put any sentences or reasoning process within "
               "the <answer> </answer> tags.")
SEPARATOR = "--------------------------------"


def problem_text(question, titles, sentences, removed):
    """The prompt's problem string: the question, the instruction, then the kept paragraphs renumbered from 0."""
    kept = [s for t, s in zip(titles, sentences) if t not in removed]
    paragraphs = "".join(f"Paragraph {k} \n\n {chr(10).join(s)} \n\n{SEPARATOR}\n\n" for k, s in enumerate(kept))
    return f"Question: {question} \n\n {INSTRUCTION} \n\nSupporting Information: {paragraphs} \n\n"


def gold_titles(example):
    return list(dict.fromkeys(example["supporting_facts"]["title"]))


def check():
    src = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    question = dict(zip(src["id"], src["question"]))
    mod = load_dataset("mehuldamani/hotpot_qa", split="train")
    ok = 0
    for r in mod:
        removed = set(r["removed_titles"])
        ok += problem_text(question[r["id"]], r["context"]["title"], r["context"]["sentences"], removed) == r["problem"]
    print(json.dumps(dict(event="check", rows=len(mod), identical=ok)), flush=True)


def build(rlcr, n_train=4000, n_dev=1000):
    used = set()
    for split in load_dataset("mehuldamani/hotpot_qa").values():
        used |= set(split["id"])
    for p in glob.glob(f"{rlcr}/eval_outputs/*/*/*.arrow"):
        d = Dataset.from_file(p)
        if "supporting_facts" in d.column_names:
            used |= set(d["id"])
    src = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    rng = random.Random(0)
    rows = []
    for ex in src.shuffle(seed=0):
        titles = ex["context"]["title"]
        gold = gold_titles(ex)
        if ex["id"] in used or len(titles) != 10 or len(gold) != 2 or not set(gold) <= set(titles):
            continue
        g = rng.randrange(3)  # number of gold paragraphs removed
        distractors = [t for t in titles if t not in gold]
        removed = rng.sample(gold, g) + rng.sample(distractors, 2 - g)
        rows.append(dict(id=ex["id"], problem=problem_text(ex["question"], titles, ex["context"]["sentences"], set(removed)),
                         answer=ex["answer"], source="hotpot", type=ex["type"], level=ex["level"],
                         gold_removed=g, removed_titles=removed))
        if len(rows) == n_train + n_dev:
            break
    test = [dict(id=r["id"], problem=r["problem"], answer=r["answer"], source=r["source"], type=r["type"],
                 level=r["level"], gold_removed=int(r["gold_removed"]), removed_titles=r["removed_titles"])
            for r in load_dataset("mehuldamani/hotpot_qa", split="test")]
    for name, part in (("train", rows[:n_train]), ("dev", rows[n_train:]), ("test", test)):
        with open(f"data/hotpot_fresh_{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(json.dumps(dict(event="split", split=name, rows=len(part),
                              gold_removed={g: sum(r["gold_removed"] == g for r in part) for g in range(3)})), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        build(sys.argv[2], *map(int, sys.argv[3:]))
