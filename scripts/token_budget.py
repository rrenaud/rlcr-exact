"""Prompt-token lengths of RLCR's training sets and completion lengths in the released eval dumps."""
import glob, sys
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, sys.argv[1])  # RLCR checkout
from system_prompts import get_sys_prompt

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
q = lambda a: np.percentile(a, [50, 90, 99, 100]).astype(int).tolist()

for name, sp in [("mehuldamani/hotpot_qa", "tabc_long"), ("mehuldamani/big-math-digits", "tabc")]:
    ds = load_dataset(name, split="train").shuffle(seed=0).select(range(2000))
    sys_prompt = get_sys_prompt(sp)
    lens = [len(tok(tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                         {"role": "user", "content": f"\n\nPROBLEM: {ex['problem']}\n\n"}],
                                        tokenize=False, add_generation_prompt=True)).input_ids) for ex in ds]
    print(f"{name} prompt tokens p50/p90/p99/max: {q(lens)}")

for fam in ["Hotpot-models", "Math-models"]:
    for p in sorted(glob.glob(f"{sys.argv[1]}/eval_outputs/{fam}/*")):
        d = Dataset.from_file(glob.glob(p + "/*.arrow")[0])
        outs = d.select(range(min(len(d), 1000)))["RLCR-output_0"]
        c = [len(ids) for ids in tok([o if isinstance(o, str) else o[0] for o in outs]).input_ids]
        print(f"{fam:14s} {p.split('/')[-1]:24s} RLCR completion tokens p50/p90/p99/max: {q(c)}")
