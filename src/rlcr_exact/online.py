"""On-policy pieces for train_frozen.py --refresh-every (vLLM backend): fresh rollouts from the current adapter,
graded as the cache was (rollout_vllm.py: RLCR's tabc format check and correctness label), and the answer-KL
reference computed on the fly instead of read from ref_topk.py's cache.

Rollouts start from the cached questions' own prompt token ids (TokensPrompt), so prompts are identical to the cache.
"""
import collections, sys

import numpy as np
import torch
from vllm import SamplingParams, TokensPrompt

from rlcr_exact.trie_score import packed_candidate_logps


def load_questions(path):
    """The cache's questions (one per line of a rollouts file): id, prompt ids, answer, source."""
    import json
    out = []
    for line in open(path):
        q = json.loads(line)
        out.append(dict(id=q["id"], prompt_ids=q["prompt_ids"], answer=q["answer"], source=q.get("source")))
    return out


class Roller:
    """Generate and grade k rollouts per question with a LoRARequest (None = base)."""

    def __init__(self, llm, rlcr_dir, k=8, temperature=0.7, max_new=1536, seed=0):
        sys.path.insert(0, rlcr_dir)
        from reward_fns import accuracy_reward, format_reward
        self.accuracy_reward, self.format_reward = accuracy_reward, format_reward
        self.llm, self.k, self.temperature, self.max_new, self.seed, self.calls = llm, k, temperature, max_new, seed, 0
        tok = llm.get_tokenizer()
        self.stop = [tok.convert_tokens_to_ids("<|im_end|>")]

    def __call__(self, questions, lora):
        self.calls += 1
        params = SamplingParams(n=self.k, temperature=self.temperature, top_p=1.0, max_tokens=self.max_new,
                                seed=self.seed * 100003 + self.calls, stop_token_ids=self.stop)
        res = self.llm.generate([TokensPrompt(prompt_token_ids=q["prompt_ids"]) for q in questions], params,
                                lora_request=lora, use_tqdm=False)
        out = []
        for q, r in zip(questions, res):
            comps = [[{"role": "assistant", "content": o.text}] for o in r.outputs]
            out.append(dict(id=q["id"], answer=q["answer"], source=q["source"], prompt_ids=q["prompt_ids"],
                            outputs=[o.text for o in r.outputs], completion_ids=[list(o.token_ids) for o in r.outputs],
                            truncated=[o.finish_reason == "length" for o in r.outputs],
                            format_ok=self.format_reward("tabc", comps),
                            correct=self.accuracy_reward("tabc", comps, [q["answer"]] * len(comps), [q["source"]] * len(comps))))
        return out


class OnlineRefTopk:
    """Same interface as train_frozen.RefTopk (get(row, device) -> top-k ids and log-probs at each completion
    position), computed from the model with its adapter disabled: the untrained policy, so the coarse-grained KL
    has the cached reference's definition. Rows are added per question group (one packed forward each)."""

    def __init__(self, k=64):
        self.k, self.store = k, {}

    @torch.no_grad()
    def add(self, model, group):
        with model.disable_adapter():
            states = packed_candidate_logps(model, [r.prefix for r in group], None)
            head = model.get_base_model().lm_head
            for row, h in zip(group, states):
                logp = head(h[row.start - 1: len(row.prefix) - 1]).float().log_softmax(-1)
                top = logp.topk(self.k, dim=-1)
                self.store[(row.qid, row.j)] = (top.indices.cpu(), top.values.half().cpu())

    def drop(self, keep):
        self.store = {key: v for key, v in self.store.items() if key[0] in keep}

    def get(self, row, device):
        ids, logp = self.store[(row.qid, row.j)]
        return ids.to(device), logp.to(device)


def rollout_stats(qs):
    """Accuracy, format and truncation of a refresh's rollouts (all outputs)."""
    c = collections.Counter()
    for q in qs:
        c["outputs"] += len(q["outputs"]); c["correct"] += sum(q["correct"]); c["format_ok"] += sum(q["format_ok"])
        c["truncated"] += sum(q["truncated"]); c["tokens"] += sum(map(len, q["completion_ids"]))
    n = max(c["outputs"], 1)
    return dict(outputs=c["outputs"], accuracy=round(c["correct"] / n, 4), format_ok=round(c["format_ok"] / n, 4),
                truncated=round(c["truncated"] / n, 4), mean_tokens=round(c["tokens"] / n, 1))
