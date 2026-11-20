# Pre-registration: exact vs sampled confidence training started from released RLCR

Written and committed before any run below starts. It reuses the protocol of `docs/prereg_fair_sweep.md` (the
"fair sweep") and states only what differs. Each run's gpuq description carries the commit (`prereg@REV`);
`scripts/prereg_submit.sh` refuses to submit from a tree with uncommitted changes to the pre-registration (this file,
`docs/prereg_rlcr_init/`, `scripts/`, `src/`). Changes after the first launch are appended under "Amendments" with a
date and reason, never edited in place.

## Questions

1. Can the online recipe that brought RLVR-hotpot to RLCR-level calibration (fair sweep, Stage B: 3-seed test Brier
   0.185 vs released RLCR 0.192) improve **released RLCR itself**, when its learning rate and answer-KL weight are
   tuned for this start?
2. Does the exact estimator still beat RLCR's sampled one (k = 1) when the start is already calibrated?

Earlier evidence (NOTES.md, 2026-10-16): Stage A (offline, frozen cached completions) from RLCR, one seed, lr
{1e-4, 3e-4, 1e-3} at W = 1: exact was ahead of sampled at every lr, but nothing beat untrained RLCR by more than
noise (test at 1e-4: exact 0.192, sampled 0.194, untrained 0.196). That run trained only the confidence on old
text, used a different scorer and W, and its lr grid was not tuned for this start. The online recipe was never tried.

## Setup (Stage C)

Stage C is the fair sweep's Stage B with one change, the start:

- Start: released RLCR-hotpot (`mehuldamani/hotpot-v2-brier-7b-no-split`), fresh LoRA (rank 4, layers 14-27). The
  answer-KL penalty is against this start (adapter off), so it holds RLCR's answers in place.
- Same as Stage B: the 4,000 fresh-train questions, 16 questions x 8 fresh rollouts per step (temperature 1.0), 600
  steps, RLCR's reward (0.5 format + 0.5 correct + 0.5 Brier), tabc_long prompt, `--pg-balance rlcr`, no support
  floor, rlcr11 confidence levels. Arms `exact` and `sampled` (`--pg-reward sampled`); same code, same seed per config.
- `TASK=hotpot-rlcr-11` in `scripts/run_frozen.sh`; `scripts/prereg_run.sh` reads Stage C configs from
  `docs/prereg_rlcr_init/configs.json` and evaluates checkpoints with RLCR as the base model.

Deviations from RLCR, as knobs: LoRA, 16 x 8 instead of 16 x 32, the answer-KL penalty, temperature 1.0 (all as in the
fair sweep), and **rlcr11**: released RLCR writes about 9% of its dev confidences off the 11 levels (0.25, 0.75,
0.15; reference panel, regeneration 0). Under rlcr11 those strings earn 0 in both arms, so training moves that mass onto
the 11 levels. Both arms share the rule; the comparison against untrained RLCR includes this format change.

## Pilot (before freezing; dev only, not registered runs)

Purpose: decide whether rlcr11's 0 reward for off-list confidences hurts this start (rule fixed beforehand: a dip in
dev Brier, EM or format at step 50 would switch Stage C to the 41-value `hotpot-stop` scorer), and sanity-check the lr
box. Exact arm only, 150 steps, otherwise the Stage C recipe; gpuq `rlcr-init-pilot` (`scripts/pilot_rlcr_init.sh`),
seeds 900-903, 2026-11-19. Off-list rate from `scripts/offlist_rate.py` (greedy dev text).

| lr | W | dev Brier 50 / 100 / 150 | AUROC 150 | EM 150 | format 150 | off-list 50-150 |
|---|---|---|---|---|---|---|
| released RLCR (untrained; 3 regens) | | 0.183 (range 0.182-0.185) | 0.799 | 0.505 | 1.000 | 0.090-0.096 |
| 2.2e-5 | 0.0067 | 0.185 / 0.183 / 0.185 | 0.791 | 0.506 | 1.000 | 0.000 |
| 9e-5 | 0.02 | 0.181 / 0.180 / 0.183 | 0.797 | 0.516 | 1.000 | 0.000 |
| 1.6e-4 | 0.0043 | 0.187 / 0.188 / 0.179 | 0.800 | 0.510 | 0.998 | 0.000 |
| 3.3e-4 | 0.023 | 0.197 / 0.186 / 0.177 | 0.807 | 0.506 | 1.000 | 0.000 |

Readings (one seed each; regeneration spread of RLCR's dev Brier is about 0.003):

- **Scorer kept (rlcr11).** Every run writes 0% off-list confidences by step 50, with EM and format unchanged, so the
  0 reward costs nothing visible. The one step-50 Brier dip (lr 3.3e-4: 0.197) comes with format 0.998 and EM 0.508; it
  is a high-lr transient that recovers to the best value of the pilot by step 150, not the scorer.
- **Box kept.** No region clearly hurts. The lowest lr is a near no-op over 150 steps (it may still move over 600);
  the two higher lrs end lowest and are still falling at step 150, which the full 600-step runs cover. Configs C0-C4
  are unchanged.

## Configs (`docs/prereg_rlcr_init/configs.json`, from `scripts/prereg_rlcr_configs.py`, RNG seed 20261119)

| config | lr | W | seed | role |
|---|---|---|---|---|
| C0 | 2.4e-4 | 0.0054 | 300 | the RLVR-start winner (B2), unchanged: does the tuned recipe transfer? |
| C1 | 2.3e-4 | 0.038 | 301 | Latin hypercube |
| C2 | 1.9e-4 | 0.048 | 302 | Latin hypercube |
| C3 | 2.8e-5 | 0.0072 | 303 | Latin hypercube |
| C4 | 8.1e-5 | 0.014 | 304 | Latin hypercube |

C1-C4 are a Latin hypercube over log-uniform boxes, lr 2e-5..4e-4 and W 0.005..0.1. The lr box reaches 5x lower than
Stage B's (5e-5) because the start is already trained on this reward. The first draw (seed 20261118) put C2 within 10%
of C0 on both knobs, a wasted run, so the seed was moved to 20261119 before anything ran. Every config runs under both
arms: 10 runs, gpuq experiment `rlcr-prereg-C`, `--priority low`, checkpointing every 25 steps, resumable.

## Checkpoints, evaluation, eligibility

As in the fair sweep: adapters saved at steps 50, 100, ..., 600; every checkpoint scored on dev (first 500 fresh-dev
questions, greedy vLLM, `scripts/prereg_eval.py`); test only at checkpoints the selection rule names, after selection
is frozen. Primary metric Brier (missing confidence = 0); secondary AUROC, ECE, answer EM, format rate.

**Eligibility** of a checkpoint: dev answer EM and format rate not more than 0.02 below **released RLCR's** (the start),
from the fair sweep's reference panel (`rlcr-prereg-ref`, mean over 3 regenerations; same evaluator, already run).

## Selection rule (dev only)

Per arm: the config whose best eligible dev checkpoint has the lowest dev Brier. Per run: its best eligible dev
checkpoint and its last checkpoint go to test. The selected config of each arm is rerun with 2 new seeds (`C<k>r1`,
`C<k>r2`: seed + 1000, + 2000), evaluated identically; test at their best-dev and last checkpoints. Replicates never
enter selection. The selection is written before any Stage C test scoring, to `docs/prereg_rlcr_init/selections.json`,
by an extension of `scripts/prereg_analysis.py` committed before it runs.

## Hypotheses and tests

All on test, at each seed's best eligible dev checkpoint, for the selected config of the arm (3 seeds).

- **H1 (improves RLCR).** The 3-seed mean test Brier of exact-from-RLCR is below released RLCR's (reference panel test,
  mean of 3 regenerations). Supported if the upper end of the 95% paired bootstrap interval over the 500 test questions
  (3-seed mean minus RLCR, per question) is below 0. Reported with the point difference either way. Seed variance is
  shown separately (each seed's own difference), not folded into the interval.
- **H2 (exact vs sampled).** Seed-averaged paired test Brier, sampled minus exact, with its 95% bootstrap interval
  over questions; exact better if the lower end is above 0. Also the per-config paired comparison at the last step
  (5 pairs, sign test), as in the fair sweep's H2.
- **Descriptive.** Exact-from-RLCR vs exact-from-RLVR (fair sweep B2, 3 seeds, 0.185), paired by question; answer EM
  and format vs RLCR; per-run best-dev vs last-step test Brier; the share of confidences off the 11 levels (`scripts/offlist_rate.py`).

A null H1 is a result, not a failure: it would say RLCR already sits near what this reward and data allow with this
adapter.

## Budget (one RTX 4090, low priority)

| part | runs | GPU-hours |
|---|---|---|
| Stage C sweep | 10 x (~6.5 train + ~0.5 dev evals) | ~70 |
| Replication | 4 x ~7 | ~28 |
| Test phase | ~16 checkpoints | ~2 |
| **total** | | **~100 (about 4-5 days of queue time without other users' jobs)** |

The reference panel (RLCR dev and test, 3 regenerations) is reused from the fair sweep: same evaluator, same splits.

Launch order: a plumbing smoke (`PREREG_SMOKE=1`, 2 steps each arm, not a registered run), then the 10 runs, then
`select`, then replicates, then the test phase, then the report.

## Exclusions and failures

As in the fair sweep: a crashed run is rerun with the same config and seed (`--parent-run`); preemptions are not
failures; no run is dropped for its results; ineligible checkpoints stay in the report.

## Amendments
