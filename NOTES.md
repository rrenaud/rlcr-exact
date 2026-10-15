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
  `rollout_vllm.py` at `0a690f0`. Switching to it is pending a check that it reproduces my records on a
  small greedy split. Neither session edits the other's files.

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

59% of test rows say 0.8 (84% correct). The rows at 0.6–0.7 are overconfident (51–55% correct), and the rows
at 0.2–0.3 are slightly underconfident (32–38% correct). Headroom is modest in-domain; the bigger
miscalibrations found in `~/rlcr_platt` are out-of-domain (SimpleQA, CommonsenseQA).

## Status (2026-10-15)

- `rlcr-math-cache` run `run-20261015T212410Z-d16f220bfe` (`scripts/cache_math.sh`, with the shared logging):
  official test and fresh dev greedy, then fresh train (2,000 problems x 8, temperature 0.7). An earlier
  text-only run (`run-20261015T210403Z-e106aa7212`) was cancelled to add logprob logging. Test and dev
  are done; train is running.
- `analyze_rollouts.py` and `train_frozen.py` read the per-question schema. The trainer cuts prefixes from
  the sampled token ids. On the test cache, all 743 well-formed rows have a confidence tail that is exactly
  one of the 228 candidate token sequences.
- `scripts/train_frozen.py` is written but **not yet run on GPU**.

## Simplified 11-level scorer

A simpler alternative to the trie, like the 0–10 scale in `rewarding_doubt`: insist on one decimal, so the
confidence is one of 11 levels 0.0 .. 1.0. The trie path is kept and stays the default.

- `src/rlcr_exact/digit_score.py`: `level_logps(model, prefixes, DigitIds)` appends `0` `.` to each prefix
  (right padding, ordinary attention mask, one causal forward for the batch) and reads logits at 3 positions
  per row: log p(0.d) = log p(`0`|x) + log p(`.`|x 0) + log p(d|x 0.), log p(1.0) = log p(`1`|x). The rest
  of the mass is malformed (reward 0). `greedy=True` also returns the greedy level: argmax over the vocabulary
  must be `1` (1.0) or `0`, then `.`, then a digit; otherwise off-format.
- `tests/test_digit_score.py` (tiny random Qwen2, fp32): log-probs equal brute-force full-sequence scoring,
  a padded batch equals one row at a time, LoRA gradients equal brute force, greedy agrees with `generate`
  (including the three off-format exits). All 7 tests pass (`PYTHONPATH=src ~/envs/pytorch/bin/python -m pytest -q tests`).
- `scripts/train_frozen.py --scorer digit` (default `trie`): values are the 11 levels, exact objective
  sum_j q_j R_j over them, sampled draws from the 11 levels + "other". `--micro-batch` / `--eval-batch` batch
  rows (default 1). Eval metrics keep their meaning (greedy, expected, mass, `greedy_off_list`). Digit runs
  log to `train_digit_{objective}_s{seed}.jsonl`. A CPU smoke run with a tiny random model ran all four
  scorer x objective pairs; it also found that `target_modules="all-linear"` with `layers_to_transform`
  raises in peft 0.21, which would have stopped the trainer on GPU (fixed: explicit list of the 7 linears).
- Approximations: the tail after the digit (`</confidence>` + end) is not scored, so a trained model could
  drift there unseen; "0.85" counts as 0.8; "1", "1.0", "1.00" (and "1" + anything) count as 1.0; bare "0"
  counts as malformed although RLCR rewards it as confidence 0.

**Coverage** (`scripts/format_coverage.py`, cache `run-20261015T212410Z-d16f220bfe`; train is the 750 problems
x 8 at temperature 0.7 written so far). Among well-formed rows:

| | test (greedy) | dev (greedy) | train (T=0.7) |
|---|---|---|---|
| well-formed rows | 989 | 985 | 5951 |
| string not a level | 1 ("0") | 0 | 10 (8 "0", 2 "0.75") |
| tail not exactly `0. d </confidence> <end>` | 1 | 0 | 10 (the same rows) |

From the raw logprobs along the sampled path (top-64 over the confidence span; the span ends at the digit,
so the step after it uses the sampled logprob of `</`), the probability of leaving the format is tiny:
mean per row at the first token (not `0`/`1`) 0 / 0 / 1.4e-6 (test / dev / train; max 8.6e-3 on one train row),
after `0` (not `.`) 0 / 1e-8 / 4.9e-6 (max 2.9e-2), after `0.` (not a digit) 2.5e-7 (max 5e-6), after
`0.d` (not `</`) 1e-8 / 0 / 5e-8 (max 7e-5), and any tail other than the sampled `</confidence><end>` 1.2e-5
(max 9e-5). p(`1`) at the first token is about 1e-9. The logged logprobs are rounded to 1e-5.

Caveats: when the model writes bare "0" it means it (p(`</`|`0`) is 0.56–0.92 on those rows, so this is not
a "0." prefix cut short): the digit scorer calls that mass malformed while RLCR pays it as 0. On the test
greedy prefixes this makes 1 row of 989 greedy off-format for digit but not for trie. The two "0.75" rows
put p(`5`|`0.7`) ≈ 1.000: the digit scorer pays them as 0.7. Both are about 0.1% of rows, and on the
released model the format outside the 11 levels has essentially no probability.

## Next steps

1. When the cache finishes: run `analyze_rollouts.py` on dev/train (per-problem accuracy vs confidence,
   share of problems with mixed correctness).
2. GPU smoke test of `train_frozen.py` (a few steps, small dev subset). Measure step time and memory, and
   check that untrained trie metrics on test match the vLLM greedy confidences (mass near 1, greedy rarely
   off the list).
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
