"""RLCR-hotpot's own training questions (mehuldamani/hotpot_qa train, 20k rows) as a prompt file for online runs.

Writes data/hotpot_rlcr_train_prompts.jsonl: one line per question with id, answer, source and prompt_ids, the
tabc_long chat prompt tokenized as rollout_vllm.py builds it (so --train-questions can feed the Roller without
cached rollouts). The tokenization is checked against a rollout cache's prompt_ids on the 500 test questions,
which are mehuldamani/hotpot_qa's test split built through the same path.

usage: build_rlcr_train.py RLCR_DIR MODEL CHECK_ROLLOUTS.jsonl
"""
import json, sys

from datasets import load_dataset
from transformers import AutoTokenizer

rlcr, model, check = sys.argv[1:4]
sys.path.insert(0, rlcr)
from system_prompts import get_sys_prompt

tok = AutoTokenizer.from_pretrained(model)
sys_prompt = get_sys_prompt("tabc_long")


def prompt_ids(problem):
    text = tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": f"\n\nPROBLEM: {problem}\n\n"}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


ds = load_dataset("mehuldamani/hotpot_qa")
test = {r["id"]: r for r in ds["test"]}
cached = [json.loads(line) for line in open(check)]
bad = sum(prompt_ids(test[q["id"]]["problem"]) != q["prompt_ids"] for q in cached)
print(f"tokenization check: {len(cached) - bad} / {len(cached)} cached test prompts match")
assert bad == 0
ids = set()
with open("data/hotpot_rlcr_train_prompts.jsonl", "w") as f:
    for r in ds["train"]:
        assert r["id"] not in ids; ids.add(r["id"])
        f.write(json.dumps(dict(id=r["id"], answer=r["answer"], source=r["source"], prompt_ids=prompt_ids(r["problem"]))) + "\n")
print(f"wrote {len(ids)} questions")
