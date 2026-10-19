# rlcr-exact: work log, decisions and plans

Goal: port the exact expected-reward confidence objective from
[rrenaud/rewarding_doubt](https://github.com/rrenaud/rewarding_doubt) to RLCR (Damani et al., *Beyond Binary
Rewards*, arXiv 2507.16806; code [damanimehul/RLCR](https://github.com/damanimehul/RLCR)), and test whether
it improves calibration over RLCR's sampled GRPO signal. Summary as of 2026-10-15.

This directory is a git repo (`main`). The subdirectories `rlcr-reward-conf-weighted-tt-scaling/`
(conf-weighted-vote work stream) and `rlcr-judge/` (judge work, a worktree of that repo) are other sessions'
own repos; they are git-ignored here and not covered. `data/` is ignored too: rebuild it with
`scripts/build_math_fresh.py` and `scripts/math_unseen.py` (it derives from the gated Big-Math dataset).

## What we have learned (as of 2026-10-17)

Short version of the sections below; each point names its evidence.

**Where the exact objective can help**
- **Math is a dead end for frozen training.** The math model writes its number in `<analysis>` and the tag copies
  it (p 0.9997), so q is one-hot at the tag and both gradients vanish. The no-penalty run drifted ~30 nats per
  answer with no calibration change. (Finding and Decision sections, 2026-10-15/16.)
- **HotpotQA keeps the confidence uncertain** at the tag (entropy ~0.66 nats), the regime where exact scoring
  helped in rewarding_doubt.
- **The released RLCR-hotpot checkpoint has nothing left to gain** from continued frozen training on its own
  reward and data: no config beat untrained by more than the noise (dev Brier SE ~0.007). Exact was ahead of
  sampled at every lr, within noise. (RLCR-start sweep.)
- **So the test bed is RLVR-hotpot** (same data, correctness-only reward, confidence ~0.95-1.0 everywhere, AUROC
  0.55). The gap to RLCR is calibration alone.

**Collapse and the support floor**
- Exact gradients on a candidate scale with its q. RLVR's q is peaked (median row has < 1% on values <= 0.7), so
  "one constant confidence" is nearly absorbing. rewarding_doubt saw the same in frozen runs at lr >= 1.6e-4.
- ReFIne's (arXiv 2510.09062) histogram remap of its SFT labels toward 0.9 empirical + 0.1 uniform reduces to:
  keep mass on every level. Our version is `--support-floor beta`: beta * CE(uniform over the 11 levels, the
  row's level marginal). Its gradient (~q_l - 1/L) does not saturate; reward cost is at most ~beta.
- beta = 0.03 raises the smallest level mass from ~0 to 0.003-0.005 and expected-confidence AUROC 0.58 -> 0.63 even
  at W=1. Late in a working run it points against the objective (cosine about -0.8 at step 500), so it acts as
  a brake. Whether it is needed at low W is being tested (floor 0.01 and 0 in the current round).

**The answer-KL penalty**
- **W=0 is out:** without the penalty the answers drift by 30-35 nats per completion within 100 steps (math run
  `run-20261015T232731Z-fd144d67c7`). Not re-run on HotpotQA.
- **W=1 stalls RLVR training.** The cached reference comes from an unpacked per-row forward; training scores
  packed. In bf16 that mismatch is ~0.1 nats per row at step 0, and its gradient is noise (cosine across batches
  ~0.02) that is 9-21x larger than the calibration gradient. Adam then moves at its noise floor (step fraction
  ~0.14-0.2). Subtracting the step-0 KL does not change the gradient; fixes are a packed reference cache or a
  dead zone relu(KL - KL0 - margin). (grad_check.py; Gradient diagnostics.)
- **Lowering W fixes it:** W = 0.1 / 0.03 / 0.01 gives dev greedy Brier 0.192 / 0.178 / 0.175 (from 0.414),
  AUROC ~0.80, and confidences that separate right from wrong (0.73 vs 0.36 at W=0.01). Dev answer KL ends
  0.23-0.38 nats, not monotone in W. Test is harder (accuracy 0.45 vs dev 0.55), so test confidence runs high.
- Overfit checks (a fixed batch) moved even at W=1: a fixed batch makes the KL gradient consistent. The stall
  on full data was gradient noise, not a broken pipeline.

**Optimization**
- After ~250 steps at 3e-4 the Adam step fraction falls under the noise floor (~0.09 at step 500): a decaying
  schedule (WSD) or a lower late lr may help.
- lr 1e-3 drifted answers 3.3-3.5 nats even at W=1 from the RLCR start.

**Measurement caveats**
- bf16: packed vs per-row scoring agree on greedy picks 87.5% of the time and give gradients with cosine 0.97;
  the scorer matches vLLM greedy on ~95% of rows (the rest are exact ties). Compare aggregates, not traces.
- Trained adapters must be checked with generation, not only frozen-prefix scoring: the LoRA is on during every
  generated token. `gen_eval.sh` (end of every hotpot run) measures the accuracy change and the answers that changed,
  against a base regeneration as the noise floor.

**Process**
- Hyperparameter rounds jitter each center by exp(N(0, 0.35)) on lr and the loss/schedule knobs (`JITTER=0.35
  sweep_chain.sh`, `scripts/jitter.py`), so repeated centers explore. Select on dev, report test.

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

## Finding: the math model's confidence is deterministic given its analysis (2026-10-15)

The no-penalty reference run (exact, 3e-4, `run-20261015T232731Z-fd144d67c7`) barely moved calibration: dev
greedy Brier 0.188 -> 0.190 -> 0.190 at steps 0/100/200, every greedy level unchanged, expected-confidence AUROC
0.712 -> 0.723. Meanwhile it drifted the answers by ~30 nats per completion. The reason is in the cached top-64
at the position that writes the decimal digit (raw logprobs, temperature 1):

| released checkpoint | rollouts | mean max digit prob | rows > 0.999 | rows < 0.9 | mean entropy |
|---|---|---|---|---|---|
| math (`big-math-digits-v2-brier-base-tabc`), fresh train, T=0.7 | 15,874 | 0.998 | 98.4% | 0.8% | 0.006 nats |
| HotpotQA (`hotpot-v2-brier-7b-no-split`), conf-vote iteration-0 rollouts | 640 | 0.70 | 0% | 96–98% | 0.69 nats |

- **Math:** the confidence is decided in `<analysis>` and only read out at the number. With q near one-hot,
  the exact gradient sum_j q_j (R_j - J) grad log q_j vanishes, and so does the sampled one. The expectation
  equals any sample, so there is also no variance to remove. Adam rescales tiny gradients to lr-sized steps,
  which fits drift without calibration gain. Expected outcome of the math frozen-prefix sweep: null for
  both objectives. **Stopped before it started.**
- **HotpotQA:** the confidence stays uncertain given the analysis (entropy ~0.7 nats). This is the regime
  where exact scoring helped in Rewarding Doubt.
- HotpotQA's surface form differs: a leading space token (`Ġ 0 . 7 Ġ</`), and ~13% two-decimal values
  (0.25, 0.75). The digit scorer would need the space variant and a second-digit read (or use the trie).
- Caveat: the HotpotQA numbers come from 640 rollouts (20 questions x 32) of the other work stream.

## Decision: Stage A moves to HotpotQA (2026-10-16)

- **Math copies a pre-written confidence.** In 98.6% of fresh-train math rollouts the `<analysis>` already
  states the number, and the tag copies it (p(sampled tag digit) 0.9997). Where the analysis first writes the
  digit, the model is uncertain: p(sampled digit) mean 0.87, 32% of rows < 0.9, 11% < 0.6. Brier also
  punishes randomness in the readout (E[(y - c)^2] = (y - E c)^2 + Var c), so RLCR's optimum is a
  deterministic tag, and training moved the uncertainty upstream into free text.
- The math model is not a constant predictor across problems: 14 distinct values, accuracy on the diagonal,
  pooled AUROC 0.72. Within a problem it nearly is: mean within-problem AUROC 0.585 over 681 mixed-correctness
  problems, and in 11% of those all 8 samples state the same value.
- **HotpotQA does not copy.** Its analysis states the number in 0% of 512 untrained rollouts (16 questions,
  conf-vote's iteration 0); p(sampled tag digit) averages 0.62.
- Confidence spread, as unselected mass (1 - sum q^2):

  | model | unselected mass |
  |---|---|
  | RLCR-hotpot, released | 0.41 (entropy 0.69 nats) |
  | rewarding_doubt base models | 0.33-0.37 |
  | rewarding_doubt exact, trained | 0.30 (300 steps), 0.22-0.27 (2,729 steps) |
  | rewarding_doubt sampled + value head | 0.13-0.18 |
  | rewarding_doubt sampled | 0.02-0.06 (0.000 with RLOO whitening) |
  | RLCR-math, released | 0.003 (entropy 0.006 nats) |

  Sampled policy gradients collapse the readout; exact keeps it spread. RLCR-math (GRPO, lr 5e-6) looks like
  rewarding_doubt's collapsed sampled runs; RLCR-hotpot (lr 1e-6) is more spread than rewarding_doubt's bases.
- **Data:** `scripts/build_hotpot_fresh.py` rebuilds mehuldamani/hotpot_qa's prompts byte for byte
  (20,000/20,000). Fresh: 4,000 train + 1,000 dev distractor-train questions (gold_removed 0/1/2 in thirds),
  plus their 500-question test (the paper's in-domain HotpotQA-Modified eval).
- **Scorer:** `--scorer hotpot`: 0.0–0.9, 0.05–0.95, the spellings 0.00–0.90 and 1.0 (31) x {`Ġ 0 . d Ġ</`,
  `0 . d </`} = 62 candidates, through a trie up to the closing token, which separates 0.7 from 0.75. It
  covers all but 0.04% of 16,000 fresh-train tails (before adding 0.d0 it missed 0.23%, mostly "0.10"). The
  trie scorer also returns hidden states for the answer-KL penalty.
- **Fresh train cache** (2,000 x 8, T=0.7): format ok 100%, accuracy 0.543, Brier 0.183, ECE 0.036, AUROC
  0.797, accuracy by stated level close to the diagonal. At the tag digit, unselected mass is mean 0.389
  (median 0.41), entropy 0.66 nats, and 0% of rows exceed 0.999. The analysis states the number in 0.1% of
  rows, so the spread holds at scale. 21.9% of problems have mixed correctness, and within-problem AUROC
  is 0.586 (436 problems): as for math, the headroom is discrimination.
- Cache: gpuq `rlcr-hotpot-cache` / `run-20261016T003439Z-bc13159a85` (`scripts/cache_hotpot.sh`).
- **Released RLCR-hotpot, reproduced** (paper's 500-question test, temperature 0): accuracy 0.435, Brier 0.196,
  ECE 0.084, AUROC 0.778, against the released eval's 0.444 / 0.188 / 0.078 / 0.798; format ok 99.8%, median
  completion 246 tokens. Fresh dev (1,000): 0.517 / 0.184 / 0.036 / 0.795. The accuracy gap is the difficulty
  mix. Their test comes from HotpotQA's validation split (all "hard"); fresh dev comes from the train split
  like RLCR's training data (easy/medium/hard). Hard-only dev accuracy is 0.386 against test's 0.434. Accuracy
  by gold paragraphs removed (0/1/2): dev 0.75 / 0.52 / 0.28, test 0.68 / 0.41 / 0.20. Select on dev; report
  both.
- The math W=1 sweep and its reference cache were cancelled before they started. The no-penalty math run was
  cancelled at step ~420 of 500. Its dev greedy Brier / AUROC were 0.188 / 0.713 at step 0 and 0.1895 /
  0.718 at step 400, and drift was ~30 nats per completion: the null result, kept in its log.

## HotpotQA Stage A: grader, smoke results, sweep (2026-10-16)

- **Grading.** RLCR grades HotpotQA by exact match after `normalize_answer`, in training (`accuracy_reward`)
  and in its HotpotQA eval (`confidence_verifier`); only TriviaQA / SimpleQA / CommonsenseQA / GPQA use the Llama
  judge. On our caches, F1 >= 0.5 relabels EM-wrong -> right on 13.2% (test), 11.1% (dev) and 9.7% (train) of
  rows, and never the reverse (e.g. "14 August" for "14 August 1947", "Herbert Ross" for "Herbert David Ross").
  The released confidence tracks F1-correctness better: dev AUROC 0.818 vs 0.795 under EM. Other evidence:
  rewarding_doubt `docs/grading.md` (EM rejects ~5% of correct TriviaQA answers, F1 > 0.5 accepts ~1% wrong
  ones; it chose F1); conf-vote `JUDGE_RESULTS.md` on branch `judge` (Haiku: 71% of EM-wrong overlapping answers
  are acceptable; Qwen2.5-7B-Instruct judge at θ=0.9 precision 97%, recall 53%).
  **Decision (user): train and select on RLCR's exact-match labels.** Every eval also reports metrics against
  F1 >= 0.5 (`*_f1`; `src/rlcr_exact/grading.py`, normalization verbatim from RLCR). Watch for EM-metric gains
  without F1 gains, which would be grader quirks being learned.
- **Smoke** (gpuq `rlcr-hotpot-frozen` / `run-20261016T011428Z-e9054eafab`, 63 min, peak 17.0 GB):
  - Reference top-64 cache of the 15,997 training rows (3.98M positions, 1.5 GiB): `ref_topk_train/` in that
    run directory, 0.19 s per row.
  - The hotpot scorer's greedy confidence equals vLLM's on 95.6% of 499 test rows. The other ~5% are exact
    ties: in vLLM's own logprobs 5.0% of rows have a top-2 gap under 0.02 nats in the confidence (bf16 logits).
    Mass on the 62 candidates is mean 0.9985 (min 0.953), off-list 0%. Scorer metrics equal vLLM's: Brier
    0.195 vs 0.196, AUROC 0.778.
  - 3 steps per objective with `--ref-topk` ran and saved adapters.
  - **The answer-KL metric has a bf16 floor of ~0.2 nats per completion** (0.198 at step 0, 0.0008/token), from
    the trie forward vs the plain reference forward. Read drift as the change from step 0.
- **Sweep:** same protocol as above (fixed before the runs). It uses `--scorer hotpot`, W = 1 with the cached
  reference, and selects on dev greedy Brier (EM labels). The lr grid is {1e-4, 3e-4, 1e-3} x {exact, sampled},
  seed 0, then seeds 1-2 at each objective's pick.

## Packed shared-prefix scoring (2026-10-16)

`trie_score.packed_candidate_logps`: a question's 8 rollouts go into one sequence (shared prompt once, then each
row's suffix and its trie nodes, with a block mask and each row's own positions). The trie scorers always use
it, and their default micro-batch is the whole group.
- **Exact in fp32:** CPU tests on a tiny model, and on GPU with Qwen2.5-0.5B on real HotpotQA groups
  (gpuq `run-20261016T034417Z-a2eac8d77a`): probabilities identical, greedy 100%, gradient cosine 1.000,
  relative difference 9e-5.
- **bf16 noise, not error:** on the 7B (`run-20261016T031122Z-6c52ffb8a7`) packed vs per-row differ by up to
  0.06 in a candidate probability, agree on 87.5% of greedy picks (32 rows) and have gradient cosine 0.971 (relative
  difference 0.24). The 0.5B in bf16 shows the same gradient noise (cosine 0.962, relative difference 0.44), so
  neither bf16 path is closer to the truth. The 20-step run's step-10 expected reward equals the per-row run's
  to 4 decimals (1.0749).
- **Speed:** 0.285 -> 0.093 s per row without grad, 5.8 -> 2.0 s per group forward+backward. **20.8 -> 7.0 s per
  32-row step** (peak 17.0 GB).
- The sweep's first run (exact 3e-4, per-row code, `run-20261016T021907Z-c918d9854a`) was cancelled at step 200
  so that all six runs use the packed code. Its dev evals at steps 0/100 are in its log. It ran under
  experiment `rlcr-math-frozen` because sweep_chain.sh hard-coded that name, now fixed.

## HotpotQA Stage A from the RLCR checkpoint: seed 0 sweep (2026-10-16)

Six packed runs (`rlcr-hotpot-frozen`), 500 steps, W=1, cached reference. Dev (500 rows, EM labels), final step;
untrained: greedy Brier 0.1848, ECE 0.039, AUROC 0.794.

| objective | lr | greedy Brier | ECE | AUROC | expected Brier | Brier (F1) | drift (nats) |
|---|---|---|---|---|---|---|---|
| exact | 1e-4 | 0.1833 | 0.028 | 0.793 | 0.1827 | 0.1751 | 0.004 |
| exact | 3e-4 | 0.1851 | 0.030 | 0.789 | 0.1829 | 0.1761 | 0.050 |
| exact | 1e-3 | 0.1824 | 0.039 | 0.795 | 0.1818 | 0.1656 | 3.34 |
| sampled | 1e-4 | 0.1869 | 0.035 | 0.784 | 0.1850 | 0.1757 | 0.002 |
| sampled | 3e-4 | 0.1892 | 0.066 | 0.791 | 0.1892 | 0.1792 | 0.039 |
| sampled | 1e-3 | 0.1944 | 0.066 | 0.776 | 0.1935 | 0.1872 | 3.53 |

- The rule selects 1e-4 for both (exact: 1e-3 is within 0.005, lower lr preferred). Test at 1e-4: exact Brier
  0.192 / AUROC 0.785 / ECE 0.076; sampled 0.194 / 0.780 / 0.082; untrained 0.196 / 0.778 / 0.084.
- Exact is ahead of sampled at every lr, but nothing beats untrained by more than the noise (Brier SE about
  0.007 at 500 rows). The released model was trained on this reward and data, so there is little left.
- At 1e-3 answer drift reaches 3.3-3.5 nats despite W=1. The eligibility rule does not check drift; treat 1e-3
  as out of bounds in practice.

## Uncalibrated start: RLVR-hotpot (2026-10-16)

RLCR-hotpot was trained from Qwen2.5-7B base (`configs/Qwen-7B/hotpot/RLCR.yaml`). Probe under tabc_long
(`probe_uncalibrated.sh`, `rlcr-hotpot-uncal`), 300 dev greedy and 100x8 train at T=0.7:

| model | split | format ok | accuracy | mean conf | Brier | ECE | AUROC |
|---|---|---|---|---|---|---|---|
| RLVR-hotpot (`hotpot-v2-correctness-7b`) | dev | 0.83 | 0.44 | 0.96 | 0.52 | 0.52 | 0.54 |
| RLVR-hotpot | train | 0.72 | 0.40 | 0.95 | 0.54 | 0.55 | 0.56 |
| Qwen2.5-7B base | dev | 0.93 | 0.30 | 0.85 | 0.52 | 0.55 | 0.63 |
| Qwen2.5-7B base | train | 0.70 | 0.20 | 0.80 | 0.54 | 0.60 | 0.63 |

- Chosen start: **RLVR-hotpot**. Same data and the same accuracy as RLCR-hotpot, trained with the GEN prompt (no
  confidence), so the gap to RLCR is calibration alone. Nearly constant 0.95-1.0 confidence. The base model is
  less accurate and states off-list forms (`" 1 "`, `"0.9"`, none) more often.
- Full cache `rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137` (same splits as the RLCR cache). Fresh
  dev: format ok 0.78, accuracy 0.43, Brier 0.53, AUROC 0.55. Paper test: Brier 0.60, AUROC 0.52.
- Smoke `rlcr-hotpot-rlvr-frozen/run-20261016T144115Z-73d8ea5b8b` (ref cache, scorer check, 3 steps each). Scorer
  agrees with vLLM greedy on 95.4% of dev rows. Mass on the list starts at 0.85, mostly `" 1 "`, and rises to 0.99
  after 3 exact steps. Greedy confidence sd starts at 0.055, just above the 0.05 eligibility floor.
- Caveat: RLVR never learned to reason about uncertainty in its analysis, and frozen training reads only the
  cached text. A low AUROC ceiling here points at Stage B.
- Queue: the same six-config sweep from RLVR (`TASK=hotpot-rlvr`), then seeds 1-2 at 1e-4 from the RLCR checkpoint.

## Gradient diagnostics (2026-10-16)

- `grad_check.py` (offline, 24 batches): from the RLVR start the calibration-only gradient is consistent
  (consistency 0.53-0.67 at step 0 and after 500 steps) but 9-21x smaller than the W=1 answer-KL gradient, which
  is noise across batches (consistency 0.04-0.13; at step 0 it is the bf16 packed-vs-unpacked reference mismatch).
  Adam's step fraction for calibration + KL is 0.17-0.23, about the noise floor (0.14), so W=1 runs random-walk.
  The strong step-0 objective gradient is the off-list (" 1 ") fix, finished by step 100. The overfit runs moved
  because a fixed batch makes even the KL gradient consistent.
- `train_frozen.py --grad-decomp-every N` logs per-term gradient norms and cosines (`event: grad`) at ~5% cost
  for N = 25. **It defaults to off: use it while tuning loss weights, and drop it once training is stable.**
  `adam_fraction` (median |m_hat|/sqrt(v_hat) from Adam's state) is logged every 10 steps at no cost.
- Queued: exact 3e-4, beta 0.03, W in {0.1, 0.03, 0.01}, with --grad-decomp-every 25.

## Lowering the answer-KL weight (RLVR start, exact 3e-4, floor 0.03, 500 steps, seed 0; 2026-10-16)

The step-0 KL against the unpacked reference is about 0.1 nats of bf16/packing mismatch, and its gradient is noise
(batch-to-batch cosine about 0.02). At W=1 it swamps the calibration gradient and training stalls. Lowering W lets
the objective through:

| W | dev Brier | dev AUROC | mean conf (correct / wrong) | test Brier | test AUROC | dev answer KL |
|---|---|---|---|---|---|---|
| untrained | 0.414 | 0.548 | 0.968 / 0.954 | | | 0.097 |
| 1 | 0.408 | 0.567 | | 0.491 | 0.538 | 0.127 |
| 0.1 | 0.192 | 0.800 | 0.811 / 0.513 | 0.241 | 0.764 | 0.380 |
| 0.03 | 0.178 | 0.799 | 0.747 / 0.393 | 0.216 | 0.765 | 0.230 |
| 0.01 | 0.175 | 0.809 | 0.725 / 0.361 | 0.205 | 0.782 | 0.287 |

Dev accuracy 0.548, test 0.452, so test confidence runs high. The answer KL is not monotone in W.
Sampled at W=0.01 is the matched comparison.

## Generation check: accuracy vs answer KL (2026-10-17)

Greedy vLLM generation of fresh dev (1000) and test (500) with the adapter on, against the base policy's cached
rollouts (`gen_eval.sh`). Accuracy counts a format failure as wrong, so it splits into format rate x accuracy
given format. The base regeneration (no adapter) is the noise floor.

| run | dev answer KL | dev format | dev acc | dev acc given format | answers changed dev/test | right->wrong dev | test acc |
|---|---|---|---|---|---|---|---|
| base regen (no adapter) | 0 | 0.778 -> 0.778 | 0.426 -> 0.429 | 0.548 -> 0.551 | 0.05 / 0.07 | 0.029 | 0.350 -> 0.338 |
| exact W=0.03 | 0.23 | -> 0.889 | -> 0.495 | -> 0.557 | 0.07 / 0.09 | 0.012 | -> 0.426 |
| sampled W=0.01 | 0.28 | -> 0.814 | -> 0.452 | -> 0.555 | 0.10 / 0.11 | 0.033 | -> 0.366 |
| exact W=0.01 | 0.29 | -> 0.842 | -> 0.471 | -> 0.559 | 0.10 / 0.13 | 0.020 | -> 0.412 |
| exact W=0.1 | 0.38 | -> 0.846 | -> 0.460 | -> 0.544 | 0.13 / 0.16 | 0.023 | -> 0.348 |
| exact lr 4.4e-4, W=0.0045 (jittered) | 0.66 | -> 0.750 | -> 0.442 | -> 0.589 | 0.15 / 0.20 | 0.044 | -> 0.358 |
| sampled lr 1.4e-3, W=0.011 (jittered) | 1.27 | -> 0.685 | -> 0.370 | -> 0.540 | 0.20 / 0.24 | 0.095 | -> 0.322 |

- Up to ~0.4 nats the adapter *raises* accuracy, all through format: RLVR often fails RLCR's format check
  (the off-list `" 1 "` confidence), and training on the confidence fixes it. Accuracy given format stays at the base
  level within noise.
- Above that, format breaks and answers change: at 0.66 nats format falls 3-4 points; at 1.27 nats dev accuracy
  drops 5.6 points, right->wrong flips triple the noise floor.
- Rule for the search: a run is eligible only if generation accuracy and format are not below base by more than
  the control's spread (~0.01-0.02), and its dev answer KL stays under ~0.5 nats. Keep W >= ~0.01 and lr <= ~5e-4.

## Overnight search, round 1 (jittered, 2026-10-17)

`hp_table.py` ranks every run. Round 1 (centers -> submitted values):
- exact, W 0.003 -> lr 4.4e-4, W 0.0045, F 0.028: dev 0.176 / 0.812, test 0.201 / 0.781, but answer KL 0.66 and
  generation format -2.8 (dev) / -4.2 (test) points. Ineligible by the generation rule.
- sampled, lr 1e-3 -> 1.4e-3, W 0.011, F 0.041: dev 0.201 / 0.716, KL 1.27, generation accuracy -5.6 points. Out.
- exact, F 0.01 -> lr 2.7e-4, W 0.011, F 0.0077: dev 0.176 / 0.807, test 0.208 / 0.777, KL 0.30, but generation
  format -1.3 / -4.6 points, where F 0.03 at the same W gained +6 / +8.
- sampled, W 0.003 -> lr 4.4e-4, W 0.0016, F 0.022: dev 0.243 / 0.630, KL 1.08, accuracy -5 points. Out.

Read: (1) a smaller floor lost format in generation at the same answer KL. (Correction: not through the
confidence list. Scored mass on the list is 1.000 by step 100 with any floor, since off-list earns 0 in the
objective. The format failures come from the LoRA acting on the generated reasoning. One seed; may be noise.) (2) Sampled drifts the answers much more than
exact at the same W (1.1-1.3 nats vs 0.3-0.7): its noisier gradient random-walks the shared LoRA. Sampled needs a
larger W or a smaller lr. (3) On dev Brier, exact sits on a plateau around 0.175-0.178 for lr 2.7e-4..4.4e-4 and
W 0.0045-0.03; the generation check, not Brier, separates them.

Round 2 centers: exact W 0.03 F 0.03 + WSD; exact W 0.02 F 0.06; exact lr 1.5e-4 W 0.02; sampled lr 1e-4 W 0.03;
sampled 3e-4 W 0.03 + WSD; sampled 2e-4 W 0.1 F 0.06.

## Overnight search, round 2 (2026-10-17)

- **New best exact:** lr 3.7e-4, W 0.02, F 0.053: dev 0.173 / 0.807, test 0.208 / 0.775, KL 0.26, generation accuracy
  +5.4 points dev and test (format +9 / +6). A larger floor again helps format.
- exact lr 1.3e-4, W 0.031, F 0.024: dev 0.178 / 0.800, the lowest KL of the eligible runs (0.18).
- exact with WSD (lr 3.7e-4, W 0.035, F 0.024, warmup 17, decay 0.23): dev 0.187, format -2.4 points. No gain from
  the schedule in this one seed.
- **Sampled is squeezed between slow and drifting.** lr 8.5e-5 (W 0.047): dev 0.268 / 0.660, still near
  collapse; lr 1.5e-4 with W 0.12, F 0.082: dev 0.347 / 0.559, barely moved. lr 4.5e-4 WSD with F 0.018: dev 0.206
  / 0.753, the best sampled Brier, but generation format -6.6 / -4.8 points (low floor). The best eligible sampled run
  is still lr 3e-4, W 0.01, F 0.03 (dev 0.228 / 0.768).
- Exact vs sampled so far (best eligible, seed 0): dev Brier 0.173 vs 0.228, AUROC 0.807 vs 0.768; test 0.208 vs
  0.274.

Round 3: seeds 1 and 2 of the best exact config (no jitter); sampled around lr 3e-4..6e-4 with W 0.02-0.05 and F
0.06 (more floor for format, more W against drift); exact with F 0.1.

## Does the floor need to stay on? (2026-10-17)

Gradient logs of the best run (F 0.053) and of F 0.0077:
- Early (steps 25-100) the floor and the objective point in different directions (cos -0.5 to -0.7), but the
  floor's job there is to put mass on empty levels (smallest level mass 0 -> 0.002-0.012 by step 100-200) so
  the objective's gradient, which scales with q, can reach them.
- Late (steps 300-500) the cosine is -0.6 to -0.94: the floor mostly opposes the objective. The exact objective
  is linear in q, so its optimum per row is a point mass; the floor holds an equilibrium spread of roughly
  (1/11) beta / (delta + beta) per level. That is a tax on expected reward (at most ~beta) and does not change the greedy
  readout much.
- Hypothesis: the floor is a startup aid plus insurance while the readout is moving; annealing it late should keep
  the gain and cut the tax. Test: `--floor-decay-from 0.2` and `0.6` (beta linear to 0 from step 100 / 300) at the
  best exact config, against its seeds 0-2 with a constant floor.

## Confidence format: rlcr11 (2026-10-18)

The candidate list is now designed from the format, not mined from rollouts: `--scorer rlcr11` (TASK=hotpot-rlvr-11)
scores 0.0 .. 0.9, 1.0 in the spaced spelling, each through `Ġ</ confidence > <|endoftext|>` (11 candidates, 60
nodes). Everything else (0.75, unspaced, `<|im_end|>`, a junk tail) earns 0, so training teaches the format;
tail-fault rows are kept. Every candidate is one support-floor level (no 0.d5 quirk).

- Stop vs value (`stop_by_value.py`, run b9f02baa42, 500 dev rows): p(ending | v) depends on v. At base the close
  `Ġ</` after 0.7-0.9 is 0.27-0.44 (the model wants a second digit); on adapters trained without the ending
  scored, `Ġ</` is ~1 but eos is ~0.5 and the stop rises from ~0.36 (after 0.0) to ~0.51 (after 0.8). One shared
  stop for all values turns the number gradient (median cosine 0.36-0.75 vs per-value), so endings are scored
  per value. `confidence` is >= 0.99997 everywhere; eos is the real decision.
- Layout check (check_trie_layout.py, run 72813b497d): every pair of layouts (none / trie interleaved / trie last /
  per row, 133 or 499 nodes) differs by the same ~0.10 nats summed answer KL (max 0.25-0.38): the bf16 floor,
  not the layout. The top-64 reference needs no packed rebuild.
- Smoke (run d986989153, live KL): base list mass 0.25 (greedy leaves the list on 57% of rows); after 10 steps
  0.997 (exact) / 0.995 (sampled). Step 10.6 s exact / 12.5 s sampled vs 17.3 / 21.9 s for hotpot-stop (499
  nodes); 18.3 GB peak. Answer KL after 10 steps as for hotpot-stop (dev 0.69 vs 0.60).
- Reference cache for the tail-ok rows (15,447): run eeac77918e (ref_topk_rlvr_stop.sh).

TODO (deferred, keep the 60-node trie for now): trim the ending. The 44 ending nodes are ~10% of a packed group's
tokens, ~6% of step time (~0.015 s/node/step). Once the format is learned the stop is ~1 for every value and
"assume it stops" gives the same gradient; early on it does not (see above). Options: drop the `confidence` node
(error < 1e-4, ~1.5%), or score endings for the first N steps then assume the stop while monitoring ending mass
at eval. A sampled ending is not cheaper if unbiased (needs the stop for every value) and puts sampling noise back
into the exact objective.

## Hotpot rlcr11: exact vs sampled k (leader config lr 3.7e-4, W 0.02, F 0.053; 2026-10-18)

| objective | dev3 Brier (seeds 0 / 1) | dev AUROC | test Brier (seeds 0 / 1) |
|---|---|---|---|
| exact | 0.181 / 0.187 | 0.80 | 0.202 / 0.202 |
| k=64 | 0.182 | 0.80 | 0.204 |
| k=16 | 0.186 | 0.80 | 0.205 |
| k=4 | 0.196 / 0.192 | 0.78 | 0.205 / 0.212 |
| k=2 | 0.197 | 0.78 | 0.207 |
| k=1 (RLCR-style) | 0.214 / 0.210 | 0.74-0.78 | 0.222 / 0.190 |

- Dev orders exact ~ k64 < k16 < k4 ~ k2 < k1 in every seed (seed spread ~0.005): exact is worth ~0.028 dev Brier
  over one confidence sample per answer; matching it takes ~16-64 samples. k is nearly free in frozen training
  (sampled scores the whole trie anyway); online it costs k short continuations after the answer.
- Test does not separate k >= 4. k=1 seed 1's test 0.190 is underconfidence (mean conf 0.50 right / 0.28 wrong)
  meeting the harder all-"hard" test set; it is among the worst on dev. Select and claim on dev.
- lr was tuned for exact; k=1 lr checks (1.5e-4, 2.5e-4) and seed 2 are queued.
- Released RLCR-hotpot on the paper's test: Brier 0.188 (eval file) / 0.196 (our reproduction); ours 0.205 from
  generation with a confidence-only LoRA on RLVR-hotpot (gen check), with answer EM unchanged and stop +22 points.

## Math from RLVR (2026-10-19)

Start: `mehuldamani/big-math-digits-v2-correctness` (same data as RLCR-math, correctness reward; public, now in
the shared hub). Probe `rlcr-math-uncal/run-20261019T050640Z-af3dc5c20a` (`probe_rlvr_math.sh`, tabc; `tag_spread.py`):

| | RLVR-math dev | RLVR-math train | Qwen2.5-7B base dev | RLCR-math dev |
|---|---|---|---|---|
| accuracy (stripped EM) | 0.627 | 0.614 | 0.236 | 0.627 |
| stops after </confidence> | ~50% (rest loops to 4096) | 49% | 96% | ~100% |
| stated confidence | 1.0 on 93% | 1.0 | mean 0.93 | 7 values, mean 0.67 |
| analysis states the value | 12% | 15% | 21% | 98% |
| Brier / AUROC stated | 0.36 / 0.55 | 0.38 / 0.53 | 0.66 / 0.58 | 0.20 / 0.71 |

- RLCR's labels are 0 when the format check fails, so the looping makes RLVR-math's raw cached accuracy 0.25.
- Correction: 28% of rows with p("1") < 0.9 at the first digit is mostly format mass (`<`), not values. Digit-level
  entropy is ~0.02 nats: the readout is nearly one-hot on 1.0 at the value level, but not copied from the analysis.
- One stated number collapses the tag (base: entropy 0.69 -> 0.15 nats with a number in the analysis; RLCR-math
  0.38 -> 0.000). Analyses that give a range without committing almost never occur naturally.

**Prompt probe** (`rlcr-math-uncal/run-20261019T072732Z-636fd3e715`, `prompt_probe_math.sh`, `analyze_prompts.py`):
analysis-only regeneration of the RLVR-math rollouts' `<think>...</answer>` under the 9 prompts of
`math_prompts.py` (merged from our drafts and two outside reviews, `docs/math_prompt_brief.md`); 297 dev greedy +
390 train at T=0.7; labels prompt-independent (accuracy 0.616 / 0.631).
Numbers below are from the rescore with both closings (`run-20261019T091622Z-ee92567dfc`; the first scoring,
636fd3e715's own, required a bare `>` close and was biased).
- **The hidden state at the tag already carries the signal, whatever the prompt.** A linear probe (5-fold grouped by
  problem) gets dev AUROC 0.72-0.75 for every prompt (control 0.731; train 0.72-0.77), above RLCR-math's stated
  0.705. Paired bootstrap vs control: no prompt moves probe AUROC beyond +-0.05 (CIs all include 0; rubric -0.044).
- Expected-confidence AUROC (frozen readout, dev): control 0.687; verify +0.030 [-0.026, +0.092], rederive +0.016,
  weakest +0.014; range -0.050, rubric -0.043. None significant. Stated confidence stays ~1.0 everywhere (mean
  0.93-0.999); rubric is the only prompt that moves it (1.0 on 66%, Brier 0.33, stated AUROC 0.61).
- **Stopping is the prompt-dependent quantity.** The `>\n` close is never followed by an end token (p ~ 0.000 for
  every prompt): it is the looping path. Only `>` + `<|endoftext|>` stops (`<|im_end|>` ~ 0). Legal mass (value +
  `>` + eos), dev: control 0.062, weakest 0.034, rubric 0.027, competing 0.099, falsify 0.114, range 0.180,
  rederive 0.239, verify 0.293, **verify_open 0.769** (it closes with `>` 84% of the time, others 17-55%). Value
  mass through `</confidence` is 0.96-1.0 for all; ~0.15-0.2 of it under most prompts goes to other closings.
  With RLCR's reward (off-list mass earns 0) the exact objective's gradient under control would be mostly "learn to
  stop" (grad size 0.058 vs 0.29 for verify_open). Value entropy over the 11 levels: 0.11-0.31 nats for the tabc
  guides, 0.56-0.65 for verify_open and rubric.
- **Range fails as predicted by one review:** 95% write the line, width 0.10 on average ("0.9 to 1.0"), and the tag
  copies the upper end 97% of the time (midpoint 6%). An interval does not keep the tag uncertain; it hands it a
  number to copy.
- No-number compliance is high everywhere (1.3-8% of analyses with a confidence-like number; the regex also
  counts math percentages). Analyses 100-220 words; <= 4% hit the 1024-token cap.
- Implication: prompt choice matters little for what a trained readout can extract; it matters for stopping and the
  starting entropy. **Pilot prompts: verify_open (stops, highest entropy) and rederive** (control dropped: 6% legal
  mass confounds it). RLVR-math cache `rlcr-math-rlvr-cache/run-20261019T072732Z-a698e7495d` (think+answer, stop at
  </answer>): accuracy 0.695 test / 0.683 dev / 0.701 train (16k rows), 98-99% reach </answer>. Prep
  (`math_pilot_prep.sh verify_open rederive`, `rlcr-math-pilot/run-20261019T104921Z-8611e96354`) regenerates the
  analysis and builds the reference; then exact vs sampled with `TASK=math-rlvr`, scorer `rlcr11-math` (its
  `>\n` + eos candidates carry ~0 mass, harmless).

**Math pilot, first run** (`rlcr-math-pilot/run-20261019T142426Z-e678e8eb65`: verify_open, exact, rlcr11-math, lr
3.7e-4, W 0.02, floor 0.053, 500 steps, `--micro-batch 2` — a whole 8-rollout group in one forward OOMs on math's
~1,500-token rows). Trainer evals (cached answers, greedy readout):

| | dev step 0 | dev step 500 | test step 500 | RLCR-math stated (probe, dev) |
|---|---|---|---|---|
| Brier | 0.293 | 0.155 | 0.152 | 0.204 |
| ECE | 0.294 | 0.042 | 0.023 | |
| AUROC | 0.58 | 0.82 | 0.81 | 0.705 |
| mean confidence (accuracy ~0.69) | 0.98 | 0.71 | 0.72 | 0.67 |
| well-formed stop mass | 0.75 | 1.00 | 1.00 | |

Stopping is learned within ~10 steps. Answer KL 0.43 at step 0 (the bf16 packed-layout floor; 0.0004/token) ->
0.57 at step 500.

**Math generation check** (`gen_math.py`, `gen_eval_math.py`, `gen_eval_math.sh`; run
`rlcr-math-pilot/run-20261019T164450Z-748a7e8142`): greedy full rollouts of math_fresh dev/test (1,000 each) under
verify_open, adapter vs base model under the same prompt (base rollouts shared in `<prep>/verify_open/gen_base`).
`run_frozen.sh TASK=math-rlvr` now ends with it.

| | dev base | dev adapter | test base | test adapter |
|---|---|---|---|---|
| accuracy (format-free) | 0.681 | 0.678 | 0.678 | 0.687 |
| accuracy delta [95% CI] | | -0.003 [-0.021, +0.015] | | +0.009 [-0.008, +0.026] |
| strict (RLCR's, format-gated) | 0.626 | 0.675 | 0.607 | 0.684 |
| stops after </confidence> | 0.886 | 0.967 | 0.875 | 0.970 |
| truncated at 4,096 | 0.087 | 0.033 | 0.097 | 0.030 |
| mean confidence (correct / wrong) | 0.989 (0.994 / 0.976) | 0.709 (0.785 / 0.534) | 0.983 | 0.718 (0.790 / 0.545) |
| Brier / ECE / AUROC | 0.284 / 0.285 / 0.56 | 0.149 / 0.041 / 0.82 | 0.289 / 0.287 / 0.55 | 0.150 / 0.031 / 0.80 |

- The adapter does not hurt answers (deltas within +-0.02; flips balanced, right->wrong 4.7% / wrong->right 4.4%
  on dev). 17% of first-<answer> strings differ, but there is no base-vs-base regeneration yet, so the share due
  to vLLM batching nondeterminism is unknown (hotpot needed that control).
- Calibration holds in free generation: the generated confidence matches the trainer's cached-answer readout
  (test Brier 0.150 vs 0.152) and beats RLCR-math's stated confidence (0.204 / 0.705 on the earlier probe) with a
  Stage A LoRA on cached rollouts. Strict accuracy rises 0.61 -> 0.68 because the adapter stops.
- Pending: verify_open sampled k=1 and rederive exact / k=1 (queued, each ends with this check).

## Next steps (2026-10-17)

In the queue (gpuq, `rlcr-hotpot-rlvr-*`):
1. Sampled, 3e-4, W=0.01, floor 0.03 (`run-20261017T003639Z-9143c6c52d`): the matched exact-vs-sampled comparison.
   Submitted before gen_eval was added to run_frozen.sh; run `gen_eval_batch.sh` on it afterwards.
2. Generation check (`rlcr-hotpot-rlvr-geneval/run-20261017T011050Z-2641c372bc`): the W=0.01/0.03/0.1 adapters
   plus a no-adapter base regeneration (noise floor for "answer changed").
3. Jittered round (s=0.35) around exact 3e-4 W=0.01 floor 0.03: W center 0.003; floor 0.01 and 0; lr 1e-3 and
   1e-4; WSD (warmup 25, decay 0.3). Each run ends with gen_eval.

Then:
- Seeds 1-2 for the best exact config and for sampled at the same config: the headline needs seed spread.
- ~~A packed-layout reference cache~~: not needed (layout check, see rlcr11 section); a dead zone on the KL for the
  ~0.1-nat bf16 floor is still an option.
- rlcr11 at the leader config (exact + sampled, 500 steps, tail-ok reference, gen check) after the cache.
- Out-of-domain evals of the best adapter; Stage B (online GRPO with exact confidence reward) if Stage A holds up.

Older plan items (math, digit-scorer default, HotpotQA rebuild) are done or dropped; see the sections above.
