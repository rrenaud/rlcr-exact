"""Does the packed forward's numerics depend on the trie? Decides how to build a packed answer-KL reference.

For real RLVR-hotpot training groups (bf16, the trainer's model), compute each row's next-token distributions at
its completion positions (what answer_kl compares) under several layouts of the same tokens:
  inter133 / inter133b  [shared][suffix_1][trie_1]...  with the old 62-candidate hotpot trie (133 nodes), twice
  inter499              same, hotpot-stop trie (499 nodes)
  last133 / last499     [shared][suffix_1]..[suffix_B][trie_1]..  (tries_last)
  none                  [shared][suffix_1]..[suffix_B], no trie (a reference forward)
  perrow                one plain causal forward per row (how ref_topk.py builds today's reference)
In exact arithmetic all are equal. Per pair: mean / max over rows of the summed KL(a || b) over completion
positions (nats per row, the unit of answer_kl) and the fraction of positions whose hidden state is bitwise equal.

usage: check_trie_layout.py [--groups 12]   (gpuq payload; writes $GPUQ_RUN_DIR/check_trie_layout.json)
"""
import argparse, collections, json, os, random, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_frozen import load_rows, make_token_bytes
from rlcr_exact.confidence import hotpot_candidates, hotpot_stop_candidates
from rlcr_exact.trie_score import packed_candidate_logps

C = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"
ap = argparse.ArgumentParser()
ap.add_argument("--groups", type=int, default=12)
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--train", default=f"{C}/rollouts_hotpot_fresh_train.jsonl")
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()
random.seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
rows, _ = load_rows(a.train, make_token_bytes(tok), 4608, tail_ok=True)
groups = collections.defaultdict(list)
for r in rows:
    groups[r.qid].append(r)
batch = [groups[g] for g in random.sample(sorted(groups), a.groups)]
old, stop = hotpot_candidates(tok).trie, hotpot_stop_candidates(tok).trie
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()


LAYOUTS = {"inter133": (old, False), "inter133b": (old, False), "inter499": (stop, False), "last133": (old, True),
           "last499": (stop, True), "none": (None, False), "perrow": None}  # name: (trie, tries_last)


def states(layout, group):
    prefixes = [r.prefix for r in group]
    if LAYOUTS[layout] is None:
        return [model.model(input_ids=torch.tensor([p], device="cuda"), use_cache=False).last_hidden_state[0] for p in prefixes]
    trie, last = LAYOUTS[layout]
    out = packed_candidate_logps(model, prefixes, trie, hidden=True, tries_last=last)
    return out if trie is None else out[-1]
PAIRS = [("inter133", "inter133b"), ("inter133", "inter499"), ("last133", "last499"), ("last133", "none"),
         ("inter133", "none"), ("none", "perrow"), ("inter133", "perrow")]
kl = collections.defaultdict(list)
equal = collections.defaultdict(list)
with torch.no_grad():
    for group in batch:
        got = {name: states(name, group) for name in LAYOUTS}
        for b, row in enumerate(group):
            sl = slice(row.start - 1, len(row.prefix) - 1)  # positions predicting the completion tokens
            h = {name: got[name][b][sl] for name in LAYOUTS}
            lp = {name: model.lm_head(h[name]).float().log_softmax(-1) for name in LAYOUTS}
            for x, y in PAIRS:
                kl[(x, y)].append((lp[x].exp() * (lp[x] - lp[y])).sum().item())
                equal[(x, y)].append((h[x] == h[y]).all(-1).float().mean().item())
        print(json.dumps(dict(group=group[0].qid, rows=len(group))), flush=True)

report = dict(groups=len(batch), rows=sum(len(g) for g in batch), trie_nodes=dict(old=len(old.tokens), stop=len(stop.tokens)),
              pairs={f"{x} vs {y}": dict(kl_mean=sum(kl[(x, y)]) / len(kl[(x, y)]), kl_max=max(kl[(x, y)]),
                                         bitwise_equal_positions=sum(equal[(x, y)]) / len(equal[(x, y)])) for x, y in PAIRS})
print(json.dumps(report, indent=1), flush=True)
json.dump(report, open(os.path.join(a.out, "check_trie_layout.json"), "w"), indent=1)
