"""Every rollout's confidence distribution under the untrained model and under each trained adapter.

For each question of ROLLOUTS (tts_gen.py output, or any cache file), its rollouts are cut after <confidence> as in
training (train_frozen.question_rows, tail faults kept) and scored by the rlcr11 trie in one packed forward per model:
"base" with the adapters off, then each NAME=DIR (an adapter, or a run directory holding one under checkpoints/).
Writes one line per rollout to OUT/scores.jsonl: id, split, j, gold
correctness, the normalised last answer, length and mean sampled log-prob, agreement (k of m others with the same
answer), and conf[model] = the 11 level probabilities (unrenormalised; null where the rollout has no cut point).
--prior MODEL,... also scores the pre-answer head (PRIOR_INSTRUCTION <expected_accuracy> after the prompt, as
train_frozen --prior-head) under those models, once per question: prior[model] = its 11 level probabilities.

usage: tts_score.py ROLLOUTS.jsonl NAME=DIR [NAME=DIR ...] --out OUT_DIR [--prior base,NAME,...]
"""
import argparse, collections, glob, json, os, sys, time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_frozen import make_scorer, make_token_bytes, question_rows
from rlcr_exact.confidence import PRIOR_INSTRUCTION, level_matrix, prior_candidates
from rlcr_exact.trie_score import packed_candidate_logps
from rlcr_exact.grading import agreement, last_answer, normalize_answer

ap = argparse.ArgumentParser()
ap.add_argument("rollouts")
ap.add_argument("adapters", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--max-prefix", type=int, default=6144)
ap.add_argument("--prior", default="", help="comma-separated models (base and/or adapter names) to score the pre-answer head under")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
values, score, _ = make_scorer("rlcr11", tok)
levels = level_matrix(values).float()
token_bytes = make_token_bytes(tok)
prior_c = prior_candidates(tok)
prior_ids = tok(PRIOR_INSTRUCTION + "<expected_accuracy>", add_special_tokens=False).input_ids
prior_levels = level_matrix(prior_c.values).float()
prior_models = [m for m in a.prior.split(",") if m]
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
names = []
for spec in a.adapters:
    name, path = spec.split("=", 1)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        path = sorted(glob.glob(os.path.join(path, "checkpoints", "*", "")))[0]
    if names:
        model.load_adapter(path, adapter_name=name.replace(".", "_"))
    else:
        model = PeftModel.from_pretrained(model, path, adapter_name=name.replace(".", "_"))
    names.append(name)
model.eval()

t0, dropped = time.time(), collections.Counter()
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "scores.jsonl"), "w") as f, torch.no_grad():
    for n, line in enumerate(open(a.rollouts)):
        q = json.loads(line)
        rows = question_rows(q, token_bytes, a.max_prefix, True, dropped)
        conf = {j: {} for j in range(len(q["outputs"]))}
        if rows:
            prefixes = [r.prefix for r in rows]
            for name in ["base"] + names:
                if name == "base":
                    with model.disable_adapter():
                        lp = score(model, prefixes)
                else:
                    model.set_adapter(name.replace(".", "_"))  # peft module names cannot hold "."
                    lp = score(model, prefixes)
                p = lp.float().exp().cpu() @ levels.T  # [rows, 11]
                for r, pr in zip(rows, p.tolist()):
                    conf[r.j][name] = [round(x, 5) for x in pr]
        prior = {}
        for name in prior_models:
            if name == "base":
                with model.disable_adapter():
                    lp = packed_candidate_logps(model, [q["prompt_ids"] + prior_ids], prior_c.trie)
            else:
                model.set_adapter(name.replace(".", "_"))
                lp = packed_candidate_logps(model, [q["prompt_ids"] + prior_ids], prior_c.trie)
            prior[name] = [round(x, 5) for x in (lp.float().exp().cpu() @ prior_levels.T)[0].tolist()]
        agree = agreement(q["outputs"])
        for j, text in enumerate(q["outputs"]):
            lps = q.get("sampled_logprobs", [[]] * len(q["outputs"]))[j]
            f.write(json.dumps(dict(id=q["id"], split=q.get("split"), j=j, correct=q["correct"][j], format_ok=q["format_ok"][j],
                                    answer=normalize_answer(last_answer(text)), length=len(q["completion_ids"][j]),
                                    mean_logprob=sum(lps) / max(len(lps), 1), agree=agree[j],
                                    conf={m: conf[j].get(m) for m in ["base"] + names}, prior=prior)) + "\n")
        if n % 100 == 0:
            print(f"question {n} {time.time() - t0:.0f}s dropped {dropped}", flush=True)
print("done", time.time() - t0, dropped)
