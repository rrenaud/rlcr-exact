# rlcr-exact: work log, decisions and plans

Goal: port the exact expected-reward confidence objective from
[rrenaud/rewarding_doubt](https://github.com/rrenaud/rewarding_doubt) to RLCR (Damani et al., *Beyond Binary
Rewards*, arXiv 2507.16806; code [damanimehul/RLCR](https://github.com/damanimehul/RLCR)), and test whether
it improves calibration over RLCR's sampled GRPO signal. Summary as of 2026-10-15.

This directory is a git repo (`main`). The subdirectories `rlcr-reward-conf-weighted-tt-scaling/`
(conf-weighted-vote work stream) and `rlcr-judge/` (judge work, a worktree of that repo) are other sessions'
own repos; they are git-ignored here and not covered. `data/` is ignored too: rebuild it with
`scripts/build_math_fresh.py` and `scripts/math_unseen.py` (it derives from the gated Big-Math dataset).

## Decisions so far

### 1. Continue training the released checkpoints, not reproduce RLCR from scratch
- The paper's main runs fully fine-tune Qwen2.5-7B **base** with GRPO on 4–6 A100s (32 samples/prompt,
  640k / 480k rollouts per run). The appendix adds OLMo-2-7B-Instruct and Qwen3-8B (HotpotQA only); no RL
  run uses a model below 7B. Neither the 4090 (24 GB) nor the W7900 (48 GB) can run the full recipe.
- Small models are not a viable stand-in: untrained Qwen2.5-0.5B/1.5B/3B base follow RLCR's format in
  0–36% of rollouts. Instruct models do better on math (74–96%) but skip `<think>` on HotpotQA.
- So: start from the released RLCR checkpoints and continue training on **fresh data**, comparing exact
  vs sampled objectives. The baseline must be continued training on the same data and steps, so a gain
  is not just more data.

### 2. Math first
- Released RLCR-math: `mehuldamani/big-math-digits-v2-brier-base-tabc` (prompt `tabc`, format pattern `tabc`).
- **Fresh data.** `big-math-digits` is a uniform sample of Big-Math (`SynthLabsAI/Big-Math-RL-Verified`,
  gated) under the rule `0 < llama8b_solve_rate < 0.75` and a plain numeric answer. The paper says "0–70%";
  the data shows 0.75 exclusive and zero solve rate excluded. The source mix and mean solve rate (0.284)
  match the pool, so a uniform draw from the remainder has the training distribution.
  - 43,140 fresh deduplicated problems (not in big-math-digits train/test or any RLCR eval set).
  - Another 15,024 `big-math-digits` train rows were never reached: the run stopped at epoch 0.4992
    (208 steps x 72 prompts, from replaying `RepeatSampler`'s `torch.randperm`, seed 43). The split is in
    `data/big_math_digits_seen_split.json`; still unverified against their wandb generations.
  - `data/math_fresh_{train,dev}.jsonl`: 4,000 / 1,000 fresh problems (shuffle seed 0).
    `data/math_fresh_test.jsonl` is big-math-digits' own 1,000-problem test split, for comparison with
    the released evaluation.
- HotpotQA is second: 70,447 fresh distractor-train questions, but its "Modified" construction
  (0/1/2 gold paragraphs removed, 8 paragraphs total) must be rebuilt first.

### 3. Stage A design: frozen prefixes, train only the confidence
- Cache rollouts of the frozen checkpoint, cut each after `<confidence>`, and keep RLCR's correctness label.
  Only the distribution of the confidence completion is trained (LoRA on the 7B in bf16). This is the
  `rewarding_doubt` "cached answers" setup: cheap, apples-to-apples, and it isolates the estimator.
- **Exact objective:** `J = sum_j q_j R_j` with RLCR's reward `R = 0.5 (format + accuracy + Brier)`
  = `0.5 (1 + y + 1 - (y - v)^2)` for a well-formed string. Mass on unlisted strings earns 0, as a malformed
  completion does in RLCR, so the objective also keeps the format intact. No renormalization or hinge.
- **Sampled baseline:** GRPO restricted to the confidence tokens. Each step samples one completion per row
  from the current model (listed strings, or "other"). The advantage is the reward minus the mean over the
  problem's 8 rollouts, with no std division as in the paper. Same prefixes and labels as exact.
- **Candidate set:** "0", "1", "1.0", "1.00", "0.d", "0.dd" (114 strings), each followed by
  `</confidence>` and either end token (`<|endoftext|>`, the checkpoint's eos, or `<|im_end|>`).
  This gives 228 candidates in a 686-node trie. The released model writes `0 . 8 </ confidence >` as
  separate tokens.
- **Prefixes are the sampled token ids, never retokenized text.** The model's samples are often not the
  canonical tokenization of their own text. For example, it generates `<`,`think` where retokenizing gives
  `<th`,`ink`: 743/743 test prefixes differ from their retokenized version while decoding to the same text.
  Training cuts `prompt_ids + completion_ids` with `logprob_spans.cut_point(..., "confidence")`. (An earlier
  check that "the prefix tokenizes like the full sequence" compared retokenized text with itself and missed this.)
- **Scoring:** `src/rlcr_exact/trie_score.py` appends the trie after the prefix with a custom 4-D attention
  mask (each node sees the prefix and its ancestors; RoPE position = prefix length + depth). One forward
  pass scores all candidates exactly and also gives token-level greedy decoding along the trie, which is
  the paper's temperature-0 confidence. CPU tests (`tests/test_trie_score.py`) check logps and LoRA
  gradients against full-sequence scoring, and greedy decoding against `generate`.
- **Evaluation:** greedy confidence (paper protocol) and expected confidence on the dev/test greedy
  prefixes. Metrics are accuracy, Brier, ECE (10 bins, as in `~/rlcr_platt/recal_eval.py`), AUROC, the
  mass on listed strings, and how often greedy decoding leaves the list.

### 4. Infrastructure choices
- vLLM 0.31 in its own venv `~/envs/vllm` (torch 2.13+cu130), separate from `~/envs/pytorch`. The HF
  `generate` alternative is about 9x slower.
- `HF_HOME` in `/etc/environment` points to another user's unwritable `/scratch/hf`; all scripts use
  `HF_HOME=~/.cache/huggingface`. The HF login (robrenaud) is saved there.
- **vLLM 0.31 hangs on shutdown and keeps the GPU**, which blocked the gpuq queue twice (once for another
  session). Every vLLM script kills its engine subprocesses and then calls `os._exit(0)` after writing
  results, and wrappers add `timeout`.
- All GPU work goes through `gpuq` (experiments `rlcr-4090-fit`, `rlcr-math-cache`).
- **Rollout logging is shared with the conf-weighted-vote work stream.** Its repo
  `rlcr-reward-conf-weighted-tt-scaling/` (git `main`) owns `logprob_spans.py`, documented there as a
  stable interface since `8adabe2`. `scripts/cache_rollouts.py` imports it by path and writes the same schema:
  per-question JSONL with prompt/completion ids, sampled logprobs and policy metadata, plus top-64 raw
  logprobs over the `<answer>` and `<confidence>` spans in `.topk.jsonl.gz`. That session folded my
  additions (jsonl input, greedy mode, chunking/resume, RLCR labels, infra revision in the policy) into its
  `rollout_vllm.py` at `0a690f0`; each row now also carries `policy["infra_rev"]` (git rev, `-dirty` if
  modified). **New rollouts use it** (`scripts/cache_rollouts.py` stays only as the producer of the current
  cache). My call maps to `rollout_vllm.py --rlcr ~/rlcr_platt/RLCR --task math --data SPLIT.jsonl --k 1
  --temperature 0 --util 0.9 --out FILE`. Since `8df33a3` it never deletes OUT. It appends, skipping
  ids already there, and refuses to start if existing rows came from different settings (base, adapter,
  seed, temperature, k). `--resume` is a no-op. Its check (gpuq `run-20261015T224646Z-44b2ab4027`): same fields as
  my cache. Only 2/32 greedy math outputs are token-identical to mine, every split a near-tie under bf16
  batch-shape nondeterminism (their chunk 24/8 vs my 250), and `correct` agrees on 29/32. So compare
  aggregates across runs, not individual traces (or match `--chunk` and the question set). Neither session
  edits the other's files.

## Measurements

**4090 capacity** (`scripts/probe_4090.py`, `scripts/probe_vllm.py`; runs under
`/data/gpuq-runs/rrenaud/rlcr-4090-fit/`):

| | 1.5B | 3B | 7B bf16 | 7B nf4 |
|---|---|---|---|---|
| LoRA fwd+bwd, HotpotQA-length row | 0.32 s | 0.56 s | 0.94 s | 0.94 s |
| LoRA, math row at the 4k cap | micro 2 | micro 1 | OOM | micro 1 |
| vLLM generation (tokens/s) | ~10k | ~6k | ~2.5–3.4k | – |

Training dominates step time once vLLM samples. 7B bf16 cannot hold both a vLLM copy and a training copy
in 24 GB.

**Released RLCR-math, reproduced** (official test, temperature 0, my vLLM rollouts vs their dump):

| | Accuracy | Brier | ECE | AUROC |
|---|---|---|---|---|
| Mine, text-only run | 0.698 | 0.181 | 0.058 | 0.723 |
| Mine, logged rerun | 0.692 | 0.182 | 0.056 | 0.722 |
| Released eval | 0.687 | 0.181 | 0.050 | 0.726 |

The differences are within vLLM's batch nondeterminism at temperature 0.

The fresh dev split behaves like the official test, confirming the fresh draw is in-distribution:
accuracy 0.702, Brier 0.179, ECE 0.042, AUROC 0.720 (format ok 98.9%).

**Fresh train rollouts** (2,000 problems x 8, temperature 0.7; `rollouts_math_fresh_train.jsonl`, 16,000
rollouts, 203 MB plus a 17 MB top-k file): format ok 99.3%, accuracy 0.711, Brier 0.174, **ECE 0.036**,
AUROC 0.721. Accuracy by stated confidence is close to the diagonal: 0.1 -> 0.13, 0.2 -> 0.22, 0.3 -> 0.33,
0.4 -> 0.40, 0.5 -> 0.48, 0.6 -> 0.56, 0.7 -> 0.70, 0.8 -> 0.84, 0.9 -> 0.88. 34.7% of problems have mixed
correctness across their 8 samples. Within a problem the stated confidence barely moves (sd 0.07), and
mean confidence correlates 0.52 with per-problem accuracy.
**Implication:** in-domain, at the training temperature, the released model is already calibrated in the
large. The headroom is in discrimination. AUROC is 0.72, while RLCR's own RLVR-classifier baseline reaches
0.81 from the reasoning text (Brier 0.164 vs 0.181 on test). Its confidence also does not separate correct
from incorrect samples of the same problem. Stage A should be judged mainly on AUROC/Brier and on
within-problem resolution, with ECE as a guard. Out-of-domain sets remain where ECE has room.

59% of test rows say 0.8 (84% correct). The rows at 0.6–0.7 are overconfident (51–55% correct), and the rows
at 0.2–0.3 are slightly underconfident (32–38% correct). Headroom is modest in-domain; the bigger
miscalibrations found in `~/rlcr_platt` are out-of-domain (SimpleQA, CommonsenseQA).

## Status (2026-10-15)

- `rlcr-math-cache` run `run-20261015T212410Z-d16f220bfe` (`scripts/cache_math.sh`, with the shared logging):
  official test and fresh dev greedy, then fresh train (2,000 problems x 8, temperature 0.7). **Complete.**
  An earlier text-only run (`run-20261015T210403Z-e106aa7212`) was cancelled to add logprob logging.
- `analyze_rollouts.py` and `train_frozen.py` read the per-question schema. The trainer cuts prefixes from
  the sampled token ids. On the test cache, all 743 well-formed rows have a confidence tail that is exactly
  one of the 228 candidate token sequences.
- `scripts/train_frozen.py` is written but **not yet run on GPU**.

## Simplified digit scorer (one-decimal levels)

A simpler alternative to the trie, like the 0–10 scale in `rewarding_doubt`: insist on one decimal. The
confidence is one of 0.0 .. 1.0 or a bare "0" (12 levels, two of them worth 0). Merged into `main`; the trie
stays as the reference and is still the trainer's default until the GPU comparison below.

- `src/rlcr_exact/digit_score.py`: `level_logps(model, prefixes, DigitIds)` appends `0` `.` to each prefix
  and runs one causal forward over the batch (right padding, ordinary attention mask), reading logits at 3
  positions per row:
  log p(0.d) = log p(`0`|x) + log p(`.`|x 0) + log p(d|x 0.); log p(1.0) = log p(`1`|x);
  log p("0") = log p(`0`|x) + log p(`</`|x 0). The rest of the mass is malformed (reward 0).
  `greedy=True` also returns the greedy level: `1` -> 1.0; `0` then `.` then a digit; `0` then `</` -> bare
  "0"; anything else is off-format.
- The bare "0" is its own level because the model writes it deliberately (p(`</`|`0`) 0.56–0.92 on those
  rows) and RLCR pays it as 0. Its logits are already computed, so it costs nothing.
- Tests (`tests/test_digit_score.py`, tiny random Qwen2, fp32): log-probs equal brute-force full-sequence
  scoring of each level, a padded batch equals one row at a time, LoRA gradients equal brute force, and greedy
  agrees with `generate` (1.0, bare "0" and the three off-format exits). With the trie tests, all 7 pass.
- `scripts/train_frozen.py --scorer digit` (default `trie`): the exact objective sums q_j R_j over the levels,
  and the sampled objective draws from the levels plus "other". `--micro-batch` / `--eval-batch` batch rows
  (default 1). Eval metrics keep their meaning. Digit runs log to `train_digit_{objective}_s{seed}.jsonl`.
  The same branch fixed a bug on main: peft 0.21 rejects `target_modules="all-linear"` with
  `layers_to_transform`, which would have stopped the trainer at its first step. It now uses the explicit
  list of the 7 Qwen2 linear layers.
- Approximations: the tail after the digit (`</confidence>` + end) is not scored, so a trained model could
  shift mass there unseen. Check a trained adapter with the trie. "0.85" counts as 0.8; "1" followed by
  anything counts as 1.0.

**Coverage** (`scripts/format_coverage.py` on cache `run-20261015T212410Z-d16f220bfe`):

| | test (greedy) | train (2,000 x 8, T=0.7) |
|---|---|---|
| well-formed rows | 989 | 15,889 |
| string is a level | 989 (100%) | 15,883 (99.96%; incl. 15 bare "0") |
| not a level | 0 | 6: "0.75" x3, "0.85", "0.33", "0.367879" (second digit sampled with p ~ 1) |

Probability of leaving the levels along the sampled path is negligible on train. From the raw logprobs,
mean per row and maximum over rows:
- first token not `0`/`1`: mean 2e-6, max 1.1e-2
- after `0`, neither `.` nor `</`: mean 6e-10, max 3e-6
- after `0.`, not a digit: mean 2.6e-7, max 5e-6
- any tail other than the sampled `</confidence><end>`: mean 1.2e-5, max 3.6e-4

p(`1`) at the first token is about 1e-9: the model never says 1.0. The logged logprobs are rounded to 1e-5.

**GPU comparison on the untrained checkpoint** (`scripts/compare_scorers.py` via `scripts/smoke_scorers.sh`, gpuq
`rlcr-math-frozen` / `run-20261015T230740Z-ce6e92f8f6`; 300 greedy official-test prefixes):

| | trie | digit |
|---|---|---|
| level probabilities vs the other scorer | | mean abs diff 2e-5, p99 7e-4, max 0.027 |
| mass on listed strings | 0.99998 | 1.0 |
| greedy level = vLLM's generated confidence | 300/300 | 300/300 |
| greedy Brier / ECE / AUROC | 0.173 / 0.068 / 0.733 | identical |
| seconds per row (no grad) | 0.21 | **0.11** one row at a time; 0.18 in batches of 8 |
| peak GB (no grad) | 14.8 | 14.6 (batch 1) / 17.6 (batch 8) |

- Batching the digit scorer is slower because rows pad to the longest prefix (up to ~4.6k tokens). It needs
  length-bucketed batches to pay off; until then use batch 1.
- **Expected-confidence ECE is unstable with floor binning:** 0.071 (trie) vs 0.085 (digit) although level
  probabilities agree to 2e-5. Most rows put ~all mass on 0.8, so the expected confidence is 0.79998 for one
  scorer and 0.8 for the other, and `floor(10 c)` puts them in different bins. Expected-confidence AUROC
  also moves (0.724 vs 0.739) through near-ties. Greedy metrics, which sit exactly on the levels, are
  unaffected. For the expected readout, bin by nearest level (`round(10 c)`) instead.
- Training path: 3 steps (1 problem x 8 rollouts) for each scorer x objective all ran and saved adapters,
  with no errors or OOM. Peak device memory was 20.3 GB, from digit with `--micro-batch 4`. Wall time per process
  (model and data load, two 16-row evals, 3 steps): trie 63/71 s, digit 35/39 s (exact/sampled). The trainer
  logs step time every 10 steps, so clean per-step numbers come from the first real run.

## Stage A sweep: protocol (fixed before the runs, 2026-10-15)

Informed by `rewarding_doubt`'s sweeps (`docs/sweeps/README.md` there; reports 07, 08, 21, 23, 25):
- **Fixed settings** (`train_frozen.py` defaults): digit scorer; Adam at a constant lr; one update per 32 rows
  (4 problems x 8 rollouts). That is rewarding_doubt's cadence, whose best LoRA lr was 3e-4, with collapse at
  1e-3 in 2/3 seeds. LoRA rank 4, alpha 4, on layers 14–27 (late half; rank 4 matched rank 8 with ~8x less
  drift, and the late half matched full LoRA with a fifth of the drift). 500 steps = one pass over the 2,000
  training problems (rewarding_doubt overfit after ~4 passes). Dev metrics every 100 steps on 500 fresh-dev
  rows, answer drift on 200; official test (989 rows) at the end.
- **Sweep:** lr in {1e-4, 3e-4, 1e-3} x objective in {exact, sampled}, seed 0. Each run is its own gpuq
  submission, so other work streams can interleave.
- **Eligibility** at the final step: mass >= 0.99; greedy off-list <= 1%; not collapsed (greedy confidence
  sd >= 0.05 and greedy AUROC >= untrained - 0.02). Answer drift is reported and flagged above 1 nat per
  completion (rewarding_doubt's Qwen tolerated ~2 nats, Llama ~0.8). It is a warning, not a gate, because
  Stage A never regenerates answers.
- **Selection:** per objective, the eligible lr with the lowest final dev greedy Brier. Within 0.005 Brier,
  prefer the lower lr. AUROC is secondary. ECE is never used alone: a collapsed run can have excellent ECE.
- **Amendment (2026-10-15, before any penalized run):** the first run (exact, 3e-4, no penalty,
  `run-20261015T232731Z-fd144d67c7`) drifted its answers by 34.6 nats per completion (0.05/token, max 59) by
  step 100, with no calibration gain yet (dev Brier 0.188 -> 0.190). This repeats rewarding_doubt's sweep 11.
  The sweep now adds its answer-KL penalty: loss += W * KL(current || untrained) over the vocabulary,
  summed over the cached completion tokens before the confidence, with **W = 1** (their LoRA setting; W = 10
  cost calibration there). Our completions are ~700 tokens against their few-token answers, so the summed
  penalty is much stronger per answer; per-token KL is reported too. The no-penalty run finishes as the
  W = 0 reference. Eligibility adds: report `answer_kl` per completion and per token at the final step.
  The training penalty uses a cached reference (`scripts/ref_topk.py`, gpuq `run-20261015T235922Z-0e08242a9b`):
  the untrained model's top-64 log-probs at all 12.3M training completion positions (4.4 GiB), computed
  once. It is coarse-grained (top-64 plus a rest bucket, never more than the exact KL), as in
  rewarding_doubt's `topk_answer_kl`. This saves a reference forward per row per step in every sweep run. Dev
  `answer_kl` stays the exact full-vocabulary KL from a live pass.
  W is not tuned unless every lr at W = 1 fails to move dev Brier/AUROC beyond the seed spread.
- **Headline:** seeds 1 and 2 at each objective's chosen lr. Report official-test greedy Brier / AUROC / ECE as
  mean ± sd over 3 seeds, against the untrained checkpoint. A difference counts only if it exceeds the seed
  spread. In-domain the target is discrimination (Brier, AUROC), since ECE is already ~0.04.

## Next steps

1. ~~Analyze the cache~~ (done: see "Fresh train rollouts" above).
2. GPU smoke test of `train_frozen.py` (a few steps, small dev subset), with **both scorers on the untrained
   checkpoint**. Check that trie and digit give the same level probabilities and eval metrics, and that both
   match the vLLM greedy confidences (mass near 1, greedy rarely off the list). Measure step time and memory
   for each. If they agree, make `--scorer digit` the default.
3. Stage A runs: exact vs sampled, same data and steps, 3 seeds each, short LR sweep (LoRA, around 1e-5 to
   1e-4; the paper's full fine-tuning used 5e-6). Report dev and official-test greedy metrics.
4. If step time is too slow (about 1 s per row with gradients through the prefix), consider LoRA on late
   layers only, or a stop-gradient prefix KV (biased; measure the difference first).
5. Out-of-domain evaluation of trained adapters: cache greedy prefixes from the frozen model on RLCR's
   other eval sets and score with the trained LoRA. These are the sets where the released model's
   calibration is worst.
6. Stage B (online): full GRPO continuation where the reasoning also updates. vLLM samples with the
   current LoRA adapter and sleeps during the HF training step (weights to host RAM), and exact scoring
   replaces the sampled confidence reward for the confidence tokens. Feed the expected Brier reward into
   the group advantage for the prefix (Rao-Blackwellized). About 9 min per 512-rollout step at 7B on the 4090,
   or use Beast (48 GB).
7. HotpotQA: rebuild the Modified construction for fresh questions (check how removed gold paragraphs
   were replaced), then repeat stages A and B with `hotpot-v2-brier-7b-no-split`.
8. Open checks: confirm the math seen/unseen split against their wandb generations. Decide whether the
   expected confidence (not only greedy) is worth reporting as a cheap readout.
