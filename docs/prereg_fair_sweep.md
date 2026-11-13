# Pre-registration: exact vs sampled (k = 1) confidence training, equal tuning, time vs quality

Written and committed before any run below starts. Each run's gpuq description carries the commit
(`prereg@REV`); `scripts/prereg_submit.sh` refuses to submit from a tree with uncommitted changes to the
pre-registration (this file, `docs/prereg/`, `scripts/`, `src/`). Changes after the first launch are appended
under "Amendments" with a date and reason, never edited in place.

## Questions

1. With equal hyperparameter search, does the exact confidence objective calibrate better than RLCR's single
   sampled confidence (k = 1)? Faster, better at the end, or both?
2. What does calibration cost? The frontier of training GPU-hours (one RTX 4090) against test calibration,
   offline (Stage A: confidence-only LoRA on cached RLVR rollouts) and online (Stage B: RLCR-style GRPO with fresh
   rollouts), next to the released RLCR model and zero-training baselines, all scored by one pipeline.

## Setup common to every run

- Start: RLVR-hotpot (`mehuldamani/hotpot-v2-correctness-7b`), fresh LoRA (rank 4, layers 14-27), RLCR's reward
  (0.5 format + 0.5 correct + 0.5 Brier), tabc_long prompt, rlcr11 confidence levels.
- Training questions: the 4,000 fresh-train HotpotQA questions (cache `run-20261019T233718Z-4511bf5189`).
  Earlier work: one pass over 4k beat one pass over 2k by ~0.015-0.02 test Brier, and a second pass over 4k hurt
  test. Stage A is one pass, so its checkpoint steps are also a data-size curve (step s has seen 4s questions).
- Arms: `exact` (all 11 levels scored; Stage B: Rao-Blackwellised answer reward) and `sampled` (k = 1; Stage B:
  RLCR's estimator, `--pg-reward sampled`). Same code, same seed per config.
- **Stage A** (offline): 4 questions x 8 cached rollouts per step, 1000 steps = one pass. Knobs: lr, answer-KL
  weight W, support floor F.
- **Stage B** (online): 16 questions x 8 fresh rollouts per step (temperature 1.0), 600 steps, RLCR's
  answer:confidence balance (`--pg-balance rlcr`), no support floor. Knobs: lr, W.
- Deviations from RLCR, unchanged from the README: RLVR start, LoRA, 16 x 8 instead of 16 x 32, answer-KL
  penalty, temperature 1.0.

## Configs (`docs/prereg/configs.json`, from `scripts/prereg_configs.py`, RNG seed 20261103)

Config 0 of each stage is the default from earlier work (tuned on exact): the "no tuning" point. The others form a
Latin hypercube over log-uniform boxes, so a few configs still cover every range:

| stage | box | configs |
|---|---|---|
| A | lr 1e-4..1e-3, W 0.005..0.1, F 0.01..0.15 | A0 default (3.7e-4, 0.02, 0.053) + A1-A6 |
| B | lr 5e-5..5e-4, W 0.005..0.1 | B0 default (1e-4, 0.02) + B1-B4 |

Every config runs under both arms: 14 Stage A and 10 Stage B runs. All at gpuq `--priority low`, checkpointing
every 25 steps; a preempted run resumes in its own directory, with training compute counted only once.

## Checkpoints and evaluation

- Adapters are saved at the registered steps: Stage A 50, 100, 200, ..., 1000 (11); Stage B 50, 100, ..., 600 (12).
  Each save logs `compute_s`: training-loop GPU time (rollouts, scoring, updates), not counting evals, saves, model
  loads or time lost to preemption.
- Every checkpoint is scored on **dev** (the first 500 fresh-dev questions) by greedy vLLM generation
  (`scripts/prereg_eval.py`). **Test** (the paper's 500 HotpotQA test questions) is scored only for the
  checkpoints the selection rule below names, after selection is frozen.
- Metric conventions: the confidence is the last `<confidence>` in [0, 1]; missing or unparseable counts as 0.
  The label is RLCR's correctness reward (format-gated). **Primary metric: Brier.** Secondary: AUROC, ECE
  (10 bins), answer EM (format ignored), format rate. Brier over parsed rows only is reported as well.
- **Eligibility** of a checkpoint: dev answer EM and format rate not more than 0.02 below the RLVR start's
  (reference panel, mean over regenerations).

## Selection rule (dev only)

`scripts/prereg_analysis.py select` writes `docs/prereg/selections.json`:

- **Frontier**: per stage, per arm (and both arms together), for each training budget (Stage A 0.05, 0.1, 0.25,
  0.5, 1, 2 GPU-h; Stage B 0.25, 0.5, 1, 2, 4, 7 GPU-h): the eligible checkpoint with the lowest dev Brier among
  all runs and steps within the budget. This is early stopping on dev at every budget.
- **Per run**: its best eligible dev checkpoint, and its last checkpoint.
- Only these checkpoints are scored on test.

## Hypotheses and tests

- **H1 (speed).** At Stage A step 300 and Stage B step 200, exact's dev Brier is lower than sampled's in the
  paired configs. One-sided sign test over configs (7 pairs in A, 5 in B; the smallest attainable p is 0.008 and
  0.031). The median paired difference is reported.
- **H2 (endpoint).** At the last step, the paired test Brier difference (sampled minus exact) is under 0.01 in
  magnitude (equivalence). Reported per config and as a mean, with a sign test. Secondary: the same comparison at
  each run's best-dev checkpoint.
- **H3 (cheap recipe).** On the Stage A frontier, a dev-selected checkpoint with at most 1 GPU-hour of training
  (3.6 including the rollout cache) reaches a test Brier within 0.02 of released RLCR's in our pipeline, with
  answer EM within the eligibility margin. Non-inferiority: the upper end of the 95% paired bootstrap interval
  (over test questions, against RLCR's regeneration 0) of our Brier minus RLCR's is at most 0.02.
- **Instability (descriptive).** Per run, test Brier at the best-dev checkpoint vs at the last checkpoint.

## Replication

After `select`, for each stage and arm, the config of the full-budget frontier point is rerun with 2 new seeds
(`CIDr1`, `CIDr2`: seed + 1000, + 2000), evaluated identically, with test at their best-dev and last checkpoints.
Replicates never enter frontier selection. They measure how much of a frontier point is seed luck.

## Reference panel (same evaluator, `scripts/prereg_refs.sh`)

RLVR-hotpot (the start), released RLCR-hotpot (`mehuldamani/hotpot-v2-brier-7b-no-split`) and Qwen2.5-7B base,
3 greedy regenerations each (question order reshuffled, which reproduces vLLM's batch nondeterminism). Dev before
selection, test in the test phase. Comparisons to RLCR use our reproduction (paired by question); the published
eval file numbers (Brier 0.188, AUROC 0.798) are quoted as a secondary column with the pipeline gap. Zero-training
baselines, fitted on dev and applied to test, from the cache's RLVR completions: the dev base rate, Platt-scaled
RLVR stated confidence, and Platt-scaled answer-token probability. RLVR-Classifier is quoted only from the paper's
eval file (it needs their separately trained classifier).

Cost axis for references: 0 for zero-training baselines. RLCR's training compute is not measured here; it is
quoted from the paper's setup (full fine-tune, 16 x 32 rollouts per step) and marked as such.

## Budget (one RTX 4090, low priority)

| part | runs | GPU-hours |
|---|---|---|
| Stage A sweep | 14 x (~2 train + ~0.3 dev evals) | ~32 |
| Stage B sweep | 10 x (~10 train at ~60 s/step, the smoke's rate + ~0.3 dev evals) | ~103 |
| Replication | A 4 x 2.3, B 4 x 10.3 | ~50 |
| Reference panel, test phase | | ~4 |
| **total** | | **~190 (about 8 days of queue time without other users' jobs)** |

Launch order: reference panel (dev), Stage A configs, then Stage B, then replication after `select`, then the test
phase (`prereg_eval.py --split test` for the selected checkpoints, `prereg_refs.sh test`), then `report`.

## Exclusions and failures

- A run that crashes is rerun with the same config and seed (a new gpuq run, `--parent-run` the old one). Both
  are kept in history; the analysis uses the run that finished. Preemptions are not failures.
- No run is dropped for its results. Ineligible checkpoints remain in the report.
- Rollout cache cost (for "+cache"): the gpuq runs `04bf759137` (33 min) and `4511bf5189` (121 min), 2.6 GPU-h.

## Amendments

- 2026-11-13, order. The test phase for the original runs (48 checkpoints from `selections.json` @5d40391) and the
  reference panel's test split ran before the replication runs finished, not after them as the launch order said.
  No choice depended on test: replicates never enter selection, and their configs were fixed by `select` @5d40391.
- 2026-11-13, replicate test checkpoints. After the 8 replicates finished, `select` was rerun (@3602813). Its
  frontier, best-dev steps and replicate list for the original runs are identical; it adds the replicates' best-dev
  and last checkpoints (11) to the test list, which `prereg_test.sh` then scored. The first submission of all 8
  replicates was cancelled before any started and resubmitted behind the test phase; those cancelled runs never ran.
- 2026-11-13, descriptive addition. The report adds, per replicated config, the 3-seed mean test Brier at each
  seed's best-dev checkpoint with a bootstrap interval over questions against RLCR, and the seed-averaged
  sampled-minus-exact gap per stage. These intervals exclude seed variance.
