# Stage B: online training of answers and confidence with the exact confidence reward

Status: design, 2026-10-20. Hotpot first (short rollouts), one RTX 4090, LoRA, shared-weight vLLM backend
(`src/rlcr_exact/shared_vllm.py`), on-policy rollouts (`--refresh-every 1`, `src/rlcr_exact/online.py`).

## What RLCR does (configs/Qwen-7B/hotpot/RLCR.yaml, GRPO_Trainer.py)

- GRPO, full fine-tune of Qwen2.5-7B from base, lr 1e-6 constant (5% warmup), beta 0 (no KL), one update per batch
  (num_iterations 1, so the ratio is 1: plain REINFORCE with a group baseline), `scale_rewards: false` (advantage
  = reward - group mean, no std division), loss_type bnpo (token losses summed, divided by the batch's token count).
- 32 rollouts per question at T 0.7, 512 completions per step (16 questions), max 1,536 new tokens, one epoch.
- Reward per rollout: 0.5 format + 0.5 correct + 0.5 (1 - (c - y)^2), all zero when the tabc format check fails.
  This equals our `rlcr_reward` for well-formed candidates. The confidence c is the sampled number: k = 1.

## The objective

A rollout is a = everything up to and including `<confidence>` (reasoning, answer, analysis), then the confidence
completion c (value, closing, end token). With q(c | a) the policy's distribution over the candidate list (the trie
scorer, off-list mass earning 0) and R(a, c) RLCR's reward:

    J = E_a E_{c ~ q(.|a)} R(a, c),   grad J = E_a[ grad log pi(a) * r(a) ] + E_a[ grad_q sum_c q(c|a) R(a, c) ]
    with r(a) = sum_c q(c|a) R(a, c)

- **Exact (ours):** the confidence is integrated analytically. Answer-side tokens get REINFORCE with r(a), the
  expected reward under the policy's own confidence distribution (Rao-Blackwellized: no variance from sampling c);
  the confidence tokens get Stage A's exact gradient. The analysis is part of a, so it is credited through how
  well the confidence it leads to scores.
- **Sampled k = 1 (RLCR on the same code):** REINFORCE on all tokens including c, with the sampled R(a, c).
- Rollouts that fail the format check get r = 0 (as RLCR) and stay in their group for the answer-side term; they
  have no confidence term. Tail-fault rows: RLCR gives 0; we keep RLCR's rule here (Stage A's EM relabelling was
  for confidence-only training).

Loss per step (rollouts i in groups g, answer tokens t):

    L = - alpha * sum_i A_i sum_t log pi(a_i,t) / N_tokens        (A_i = r_i - mean_g r, no std; bnpo normalization)
        - sum_i r_conf,i                                          (Stage A's exact objective, unchanged weight)
        + beta * floor_i + W * KL_i(answer tokens, vs adapter-off reference)

alpha weighs the answer-side term against the confidence term, whose scale Stage A tuned (lr 3.7e-4). Constant
floor beta 0.053 (2026-10-20 ablation). W stays positive (0.02 to start; lower values later, never 0): with a
correctness reward the answers have their own signal, but the shared LoRA drifted tens of nats without W in Stage A.

## Implementation (train_frozen.py, on top of --refresh-every 1)

1. Keep every rollout (not only well-formed rows) with its sampled token ids, format flag and label.
2. One packed forward per question (prompt once, 8 completions): the trie read at each well-formed row's
   `<confidence>` (exact term, floor) and the log-probs of the sampled completion tokens before it. The token
   log-probs reuse the KL's chunked, checkpointed lm_head pass (`_chunk_kl_topk` gets the gather of the sampled ids).
3. r_i from the same forward's q (detached) for exact; for k = 1, from the sampled confidence text.
4. Logs per step: reward parts (format, correct, calibration), advantage spread, answer tokens, KL, gen/ref/train
   seconds. Online dev generation eval every M steps: 500 dev questions greedy with the current adapter (~20 s),
   accuracy, format, stated-confidence Brier/AUROC, answer changed vs the base; the cached-dev readout as now.

## Validation ladder (each run small; pass before the next)

| step | what | pass |
|---|---|---|
| B0 | alpha = 0 | reproduces the --refresh-every 1 Stage A run (same seed): the plumbing adds nothing |
| B1 | overfit 8 questions, accuracy-only reward, alpha > 0, 100 steps | train accuracy on them rises clearly (e.g. 0.5 -> 0.8); flat with alpha = 0. Checks the sign and the advantage |
| B2 | from the Stage A 2k leader adapter, alpha in two values, 300 steps | dev generation accuracy within noise of the start, no length or format collapse, calibration no worse than Stage A |
| B3 | from RLVR + fresh LoRA, exact vs k = 1, 2 seeds, 300-500 steps | the comparison: Brier / AUROC / accuracy at matched steps and rollouts |

B2 isolates what answer training adds to a calibrated model; B3 is the paper-comparable setup (confidence learned
online too). Cost at ~12 s per step (4 questions x 8 rollouts): 300 steps ~ 1 h.

## Expectations and risks

- Accuracy gains are unlikely at this scale: RLVR was trained for correctness and is level with RLCR. The likely
  gains are in the analysis (AUROC) and in stability (exact's Rao-Blackwellized answer reward vs k = 1's).
- Batch size: 32 rollouts per step vs RLCR's 512. Noisier; compensate with alpha / lr, not by matching RLCR.
- One shared LoRA (r 4, layers 14-27) for answers and confidence: capacity may limit answer learning; rank and
  layer range are knobs for B2.
- Eval noise: 500 dev questions give SE ~2.2 points on accuracy, and greedy regeneration alone moves it ~1-2;
  claims need paired comparisons and seeds.
- Memory: the answer-side log-probs add chunked lm_head passes over ~1,600 completion positions per question; the
  KL already does this, so the term reuses it.

## Open choices (defaults in parentheses)

- alpha (start 1 with bnpo normalization, then x0.3 / x3) and whether lr changes for B.
- Start point for B2 (Stage A 2k leader, seed 0).
- LoRA rank / layers for answer learning (r 4, layers 14-27, as Stage A).
- W (0.02).
