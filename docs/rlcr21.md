# rlcr21: exact confidence on a 0.05 grid

## Conclusion (2026-11-26)

**No gain; closed.** A 0.05 grid does not improve exact confidence training over rlcr11, and the reason is structural.
- **Offline (Stage A, A2, 3 seeds paired with rlcr11):** rlcr21 is worse in every seed, +0.009 / +0.003 / +0.005 dev
  Brier at step 1000 (progressive +0.002 to +0.003), with no AUROC gain. rlcr11's hyperparameters transfer (200-step
  paired gaps ≤ 0.004 at lr 1e-4 to 3.7e-4), so this is not mistuning. Step time is +13-15%.
- **Online (Stage B, B2, seed 202):** equal to rlcr11 (0.160 vs 0.158 @600) only because training drops the half
  steps. Their share of q goes from 0.56 to 0 between steps 6 and 11, in the overconfident start, and there is no
  support floor to bring them back. B2r1 / B2r2 were cancelled: they would only re-confirm the collapse.
- **Why the grid cannot matter much:** the Brier decomposition below. Dev Brier ≈ 0.16 is UNC (uncertainty) 0.249 minus
  RES (resolution, i.e. discrimination) 0.08-0.10 plus REL (reliability, i.e. miscalibration), which is only
  0.001-0.006 above its 500-row noise floor.
  Cross-validated isotonic recalibration, the best any reshaping of the stated confidences can do without changing
  their ranking, moves Brier by -0.006 to +0.003. The seed-to-seed spread (~0.015) is discrimination, not calibration.
  A finer grid, smoother or regression-like gradients for q (a log-concavity penalty was considered), or an ordinal
  loss (RPS, which tied exact earlier) all compete for at most ~0.005 Brier.
- **Where the headroom is:** discrimination, meaning what the confidence knows about correctness, not the shape of
  q. Examples: answer agreement across rollouts (loo-agreement worktree, offline AUROC 0.83), or more data per step.
  Smoothing q would still be worth trying for checkpoint stability (greedy flips between adjacent levels, ECE
  0.03-0.16 across checkpoints), but not for Brier.

| run (dev, 500 q) | Brier | REL (noise floor) | RES | isotonic CV |
|---|---|---|---|---|
| released RLCR, regen 0 / 1 | 0.185 / 0.182 | 0.006 / 0.007 (0.004) | 0.071 / 0.075 | 0.184 / 0.182 |
| A2 rlcr11 @1000, s102 / 1102 / 2102 | 0.153 / 0.168 / 0.172 | 0.004 / 0.009 / 0.008 (0.003) | 0.100 / 0.091 / 0.085 | 0.153 / 0.163 / 0.167 |
| A2 rlcr21 @1000, same seeds | 0.163 / 0.171 / 0.177 | 0.008 / 0.014 / 0.008 (0.007) | 0.094 / 0.092 / 0.081 | 0.164 / 0.165 / 0.174 |
| B2 rlcr11 @600, s202 / 1202 / 2202 | 0.158 / 0.172 / 0.156 | 0.004 / 0.008 / 0.002 (0.003) | 0.095 / 0.086 / 0.093 | 0.161 / 0.167 / 0.159 |
| B2 rlcr21 @600, s202 | 0.160 | 0.011 (0.003) | 0.101 | 0.154 |

`scripts/brier_decomp.py` (REL noise floor = Σ v(1-v)/N, larger for rlcr21 because it fills more levels). The B2
rlcr21 run's high RES and recalibrated 0.154 come from one seed at one checkpoint and are treated as noise.

### Reading the decomposition

Worked example with charts, free-throw analogy and every Stage A/B/C pick:
**https://rrenaud.github.io/rlcr-exact/brier/** (source `docs/brier_site/`, data from `scripts/brier_site_data.py`).

The three terms are Murphy's (1973) partition of the Brier score, with his names:
- **UNC = uncertainty.** How hard the task is to forecast with no information at all.
- **RES = resolution.** How much the forecast separates rows that turn out right from rows that turn out wrong. Higher
  is better. In ML terms, discrimination.
- **REL = reliability.** How far the stated numbers are from the hit rates they should equal. Lower is better: the
  name is historical (a "reliable" forecaster has REL = 0), and in ML terms it is miscalibration. It is the squared
  version of the reliability diagram's distance from the diagonal; ECE is its absolute-value cousin.

Group dev rows by stated confidence v (n_v rows, hit rate ȳ_v; overall hit rate ȳ). Then, exactly:

    Brier = UNC − RES + REL
    UNC = ȳ(1 − ȳ)                   variance of the outcome; set by the task (≈ 0.249 here: trained models are right ~half the time)
    RES = Σ n_v/N (ȳ_v − ȳ)²         between-group variance of the outcome = the part of UNC the confidence explains
    REL = Σ n_v/N (v − ȳ_v)²         squared gap between each stated value and its group's hit rate

**Toy examples.** 100 questions, 50 answered correctly, so UNC = 0.5 × 0.5 = 0.25 for every forecaster below.

| forecaster | what it says | groups (stated v: right / rows) | RES | REL | Brier = UNC − RES + REL |
|---|---|---|---|---|---|
| coin flip | 0.5 on everything | 0.5: 50/100 | 0 | 0 | 0.25 |
| overconfident, knows nothing | 0.95 on everything | 0.95: 50/100 | 0 | (0.95−0.5)² = 0.2025 | 0.4525 |
| knows something, calibrated | 0.8 or 0.2 | 0.8: 40/50; 0.2: 10/50 | ½(0.3²) + ½(0.3²) = 0.09 | 0 | 0.16 |
| knows the same, overstates it | 1.0 or 0.0 (same split) | 1.0: 40/50; 0.0: 10/50 | 0.09 | ½(0.2²) + ½(0.2²) = 0.04 | 0.20 |
| knows more, calibrated | 0.9 or 0.1 | 0.9: 45/50; 0.1: 5/50 | ½(0.4²) + ½(0.4²) = 0.16 | 0 | 0.09 |

Rows 3 → 4 are a calibration change: same groups, so same RES; only the labels moved. Mapping 1.0 → 0.8 and
0.0 → 0.2 (what isotonic recalibration learns) recovers 0.16. Rows 3 → 5 are a discrimination change: the groups
themselves are cleaner, which no relabelling can produce. Row 2 is RLVR, and row 3 is about where our trained runs
are (Brier ≈ 0.16, RES ≈ 0.09).

**The same breakdown on two real dev runs** (500 questions). Each level contributes n_v/N (v − ȳ_v)² to REL and
n_v/N (ȳ_v − ȳ)² to RES.

A2 exact rlcr11 s102 @1000 (ȳ = 0.528): Brier 0.1534 = UNC 0.2492 − RES 0.1002 + REL 0.0044 (noise floor 0.0033).

| stated v | rows | right | hit rate ȳ_v | REL part | RES part |
|---|---|---|---|---|---|
| 0.0 | 13 | 1 | 0.08 | 0.0002 | 0.0053 |
| 0.1 | 53 | 4 | 0.08 | 0.0001 | 0.0217 |
| 0.2 | 33 | 3 | 0.09 | 0.0008 | 0.0126 |
| 0.3 | 61 | 15 | 0.25 | 0.0004 | 0.0097 |
| 0.4 | 26 | 5 | 0.19 | 0.0022 | 0.0059 |
| 0.5 | 25 | 10 | 0.40 | 0.0005 | 0.0008 |
| 0.6 | 30 | 16 | 0.53 | 0.0003 | 0.0000 |
| 0.7 | 44 | 30 | 0.68 | 0.0000 | 0.0021 |
| 0.8 | 141 | 113 | 0.80 | 0.0000 | 0.0211 |
| 0.9 | 74 | 67 | 0.91 | 0.0000 | 0.0211 |

REL is 0.0044 against a 0.0033 floor. The largest single part is 0.4 (5 of 26 right), a small group. The big groups
at 0.8 and 0.9 match their stated values almost exactly. RES comes from the
ends, where many rows say 0.1 (8% right) or 0.8-0.9 (80-91% right).

RLVR, before any confidence training (ȳ = 0.438): Brier 0.5127 = UNC 0.2462 − RES 0.0043 + REL 0.2708. It says 0.95
or 1.0 on 435 of 500 rows, and those rows are right 44% and 46% of the time. Those two levels alone contribute
0.109 + 0.129 = 0.238 of REL. Their hit rates are both ≈ ȳ, so they add no RES. This is the toy's
"overconfident, knows nothing" row.

In general, it is an analysis of variance of y: UNC is the total, RES is the between-groups share, and UNC − RES is the residual
that nothing about the stated confidence can remove. REL is separate: the penalty for writing the wrong number on
a group. Calibration work (a finer grid, smoothing q, an ordinal loss, post-hoc recalibration) can only shrink REL.
Discrimination work (new information about whether this answer is right) is the only way to grow RES. Two
corrections make REL readable at N = 500:
- **Noise floor.** A perfectly calibrated forecaster still shows E[REL] ≈ Σ_v v(1−v)/N, because each ȳ_v is a sample.
  That is ~0.003 for 10 used levels and ~0.007 for 20. Only REL above the floor is evidence of miscalibration.
- **Isotonic CV.** A 5-fold cross-validated monotone recalibration is about the best Brier any relabelling of the
  stated values can reach without new information. It agrees with the floor: it gains ≤ 0.006 on every exact run.

**Analogy: NBA free throws** (2023–24 season, 53,432 attempts, forecast from 2022–23 only; `scripts/nba_ft_extract.py`,
`docs/brier_site/nba_ft.csv`). Here UNC = 0.169. The league rate for everyone has REL = RES = 0, Brier = UNC. Each shooter's
raw last-season rate buys RES 0.0047 but pays REL 0.0015 (8× its floor). Shrinking toward the league mean (beta
prior fitted on 2022–23) keeps that RES and cuts REL to 0.0004. Even the in-sample oracle, each shooter's own
2023–24 rate, only reaches RES 0.008: 5% of UNC. The rest is shot-to-shot luck that shooter identity cannot see.
That is the same shape as our problem: calibration is cheap and nearly done, and Brier is bounded by what the
forecaster knows. The difference is scale. Our trained runs explain 30–40% of UNC.

Across all experiments (the page's table):
- **Base Qwen and RLVR are REL-dominated** (REL 0.32 / 0.27, larger than UNC). Confidence RL fixes that almost
  entirely. Every trained run except two sampled-arm checkpoints (B0 s200, C0 sampled, +0.015) is within 0.008
  of its floor.
- **Trained runs differ mainly in RES.** Released RLCR has 0.071, ours 0.08–0.10. The online exact picks out-resolve the
  sampled picks (B2 0.086–0.095 vs B0 0.069–0.076; single dev checkpoints, each arm at its own pick).
- **The free-throw grid contrast.** There, forecasts crowd into 0.6–0.9, and a 0.1 grid costs ~0.001 Brier vs ~0.0004
  for 0.05. A grid only matters when the information sits in a narrow range. HotpotQA confidences span 0–1.

---

`--scorer rlcr21` (`TASK=hotpot-rlvr-21` / `hotpot-rlcr-21`, `LEVELS=21 scripts/prereg_run.sh CID exact`): the rlcr11
format at 0.05 steps, 0.0, 0.05, 0.1, ..., 0.95, 1.0, spelled as released RLCR writes them (`Ġ0.75Ġ</confidence><|endoftext|>`,
one decimal unless the second is 5). After `Ġ0.d` one softmax picks the close (0.d) or `5` (0.d5); each value's stop
is scored on its own path. 21 candidates, 110 trie nodes (rlcr11: 60). Exact arm only; no sampled arm.

Compared with rlcr11, as knobs:
- **Grid 0.05** (the change). RLCR's prompt and reward accept any number in [0, 1]. rlcr21 is closer to that than rlcr11,
  and puts on-list the 0.25 / 0.75 / 0.15 that released RLCR writes on ~9% of dev rows.
- **Spelling**: minimal (0.7, 0.75), not uniform two-digit (0.70). This is RLCR's own spelling. 0.70 earns 0.
- **Support floor** (Stage A only): the same F. `support_floor` is a mean over levels, so the floor's total mass at
  equilibrium is unchanged and its per-level mass halves.

## What to expect (prior evidence, no GPU)

The direct gain from a finer grid is small. Rounding a calibrated forecast to a 0.1 grid costs at most ~0.1²/12 = 0.0008
Brier; on a 0.05 grid the cost is 0.0002. Proxy from the three rlcr11 exact A2 runs (q log, steps 700-1000, ~9.3k rows
each): round E_q[c] to 0.05 instead of 0.1 and Brier drops by 0.0008 / 0.0009 / 0.0001, AUROC rises by 0.007 / 0.007 /
0.003 (fewer ties). Seed-to-seed spread of A2's step-1000 dev Brier is 0.153 / 0.168 / 0.172. **A Brier effect from
the grid alone is ~10x below seed noise.** Plausible measurable outcomes are a small AUROC/ECE gain and a cost (slower
learning while q spreads over 21 levels; +~6% step time from 50 more nodes).

So the questions these runs can answer are:
1. Does rlcr21 learn as fast as rlcr11 at rlcr11's hyperparameters, without loss of format or EM?
2. Does it use the half steps (share of greedy confidences on 0.d5)?
3. Paired AUROC/ECE across seeds.
4. From released RLCR (Stage C), does rlcr21 beat rlcr11 there, now that RLCR's native 0.x5 values are no longer
   penalised?

## Hyperparameter transfer

- lr, W: taken unchanged from rlcr11's prereg picks. Same LoRA and reward scale. The confidence gradient
  Σ_j q_j (R_j - r) ∇log q_j is spread over more candidates but has the same scale.
- F: unchanged (see above). The floor matters more for rlcr21. The exact gradient on a level is proportional to its
  probability, so levels the start never uses (0.d5 on RLVR) are reached through neighbouring mass and the floor.
- Retune only if run 1 trails its rlcr11 pair early (steps 100-300) by more than the seed spread. If so, try lr x1.5.
- Warm start from an rlcr11-trained adapter: not recommended. Its greedy text has 0% half steps, so it starts with
  no gradient on them. It also needs a second pass over the 4k questions, which hurt test before. Released RLCR is
  the natural start that already uses 0.d5 (Stage C).

## Short transfer study (20% of Stage A), 2026-11-25

**Not a tuning tool.** On rlcr11's nine Stage A exact runs, step-200 numbers do not predict where a run ends:
Spearman of dev Brier @200 vs dev @1000 is -0.03 (vs best dev, -0.45). Progressive argmax Brier over steps 100-200
vs dev @1000 is 0.42. The eventual winner (A2 s102, 0.153) sits mid-pack at step 200 (0.228). Ranking configs by
their first 200 steps would select noise.

**What it can test**: paired transfer. Stage A uses a constant lr with no schedule, so a 200-step run is an exact
prefix of the full run (`MAX_STEPS=200`), and the same seed gives the same LoRA init and data order as the rlcr11 run
it pairs with. The per-config gap, rlcr21 minus rlcr11 at the same seed, removes most config and seed variance. If the
best lr moved, the gap should trend with lr.

| run | lr / W / F | rlcr11 pair | role |
|---|---|---|---|
| A1 x 200 steps | 1.0e-4 / 0.012 / 0.019 | `80e8675ea0` | low lr, low floor |
| A0 x 200 steps | 3.7e-4 / 0.02 / 0.053 | `200eb082f1` | ~A2 at 2.3x the lr |
| A2 full (first 200 = its probe) | 1.6e-4 / 0.016 / 0.063 | `9d2314faca` | the main run |

Hypotheses, stated before the runs:
- **T1 (lr transfers).** The confidence gradient has the same scale (more candidates, each with less q). Predict
  |gap| < 0.02 on dev Brier @200 and on progressive argmax Brier, steps 100-200, at all three lrs, with no trend in lr.
- **T2 (start).** rlcr21's list covers more of what the start writes (0.95, 0.85, ...), so list mass at step 0 and 50
  is at least rlcr11's. The smoke's scorer check gives the step-0 mass.
- **T3 (floor).** The half steps get gradient mostly from neighbouring mass and the floor. Predict that the half-step
  share of greedy dev confidences @200 is lowest for A1 (F 0.019).

Rules for new guesses, fixed in advance (noise floor for the paired gap unknown; across A2 seeds dev @200 spans 0.011):
- Gap grows as lr falls (A1 > A2 > A0, A1 - A0 > 0.02) → the optimum moved up: next guess A2 at lr 2.4e-4 (x1.5).
- Gap grows as lr rises → next guess A2 at lr 1.1e-4 (x0.67).
- Half-step share @200 < 5% in every run → next guess A2 with F 0.13 (x2).
- Otherwise: keep A2, run its replicates (ladder row 2).

Cost ~0.9 GPU-h for the two probes. All runs at normal priority, gated on the smoke (`scripts/after_ok.sh`).

### Results (2026-11-25; smoke b2ca238df3, probes dba390183b / 00c01294ee, A2 b650f0b60e; all exit 0)

Smoke, untrained, on cached dev rows (300): **list mass** with rlcr21 vs rlcr11 is 0.45 vs 0.25 on RLVR, 0.95 vs 0.78 on
released RLCR. Greedy-scorer agreement with vLLM's text on RLCR is 0.95 vs 0.85 (its 0.x5 writes are now on the list).

Paired with rlcr11 at the same config and seed (dev, greedy generation; "prog" = pre-update calibration of training
rows, mean over a step window):

| config | dev Brier @200, 21 / 11 | prog argmax Brier 100-200, 21 / 11 | half-step share @200 | step time |
|---|---|---|---|---|
| A1 (lr 1e-4) | 0.220 / 0.220 | 0.218 / 0.222 | 0.86 | +15% |
| A2 (lr 1.6e-4) | 0.228 / 0.228 | 0.228 / 0.224 | 0.62 | +14% |
| A0 (lr 3.7e-4) | 0.185 / 0.187 | 0.218 / 0.221 | 0.81 | +13% |

- **T1 holds.** Every gap is within 0.004, with no trend in lr. By the pre-stated rules, there are no new guesses: keep A2.
- **T2 holds** (mass above).
- **T3 fails.** Half steps are used heavily without help from the floor; A1, with the smallest floor, uses them most.
- EM and format equal rlcr11's everywhere. Step time is +13-15%, above the +6% guessed from node counts.

A2 full run, 1000 steps (one seed): dev Brier 0.163 vs 0.153 @1000. The last four checkpoints differ by
+0.007 / +0.009 / -0.009 / +0.009. Dev AUROC is 0.837 vs 0.852 (late checkpoints -0.005 to -0.020). The
progressive numbers are level: argmax Brier over steps 800-1000 is 0.190 vs 0.188, AUROC 0.801 vs 0.798. Both runs
spike at step 900 (0.191 / 0.200): a shared data-order transient, which shows the pairing works.
- The half-step share swings between checkpoints (0.09 @500, 0.99 @600, 0.29 @800), and ECE swings with it (0.03-0.16).
  Greedy sits near ties between adjacent levels, so a small shift moves many rows at once. rlcr11 swings too.
- So far rlcr21 is not better, and possibly worse by ~0.01 dev Brier at the end. That is inside the A2 seed spread
  (0.153-0.172). The hoped-for AUROC gain is not seen. Next: the A2r1 / A2r2 replicates (queued 2026-11-25, 80b1a0c088 /
  3da62f73c4) for a 3-seed paired gap (done, below).

### Results, 2026-11-26: A2 three seeds, B2 seed 202

**Stage A, A2, three seeds** (rlcr21 minus rlcr11, paired by seed):

| seed | dev Brier @1000 | best dev, 21 / 11 | dev AUROC @1000 | prog argmax Brier 800-1000 |
|---|---|---|---|---|
| 102 | +0.009 | 0.163 / 0.153 | -0.015 | +0.002 |
| 1102 | +0.003 | 0.166 / 0.163 | -0.000 | +0.003 |
| 2102 | +0.005 | 0.171 / 0.170 | -0.009 | +0.002 |

rlcr21 is slightly worse in 3 of 3 seeds: about +0.006 dev Brier at the end, +0.002 progressive. That is small, but
it is consistent and in the wrong direction for the hoped-for gain (≤0.001). No AUROC gain.

**Stage B, B2 seed 202** (online from RLVR, no support floor): dev Brier 0.160 vs 0.158 @600, progressive 400-600
0.163 vs 0.165, best dev 0.159 vs 0.158. Equal, **because online training abandons the half steps.** The share of
half-step mass in the training rows' q is 0.45 at step 1, 0.56 at step 6, 0.02 at step 11 and 0.00 from step 16 on
(0.07 by step 600). Greedy dev text has 0-3% half steps at every checkpoint. The collapse coincides with the
overconfident start (E_q[c] 0.94 → 0.79 over steps 1-11, accuracy ~0.5). Hypothesis: while the model is
overconfident, the `5` after `0.d` (worth +0.05) is the worse choice on nearly every row. So one shared token decision
is pushed down hard (lr 2.4e-4, 128 rows per step), and the exact gradient is proportional to q, so once q is ~0 it
never comes back. Stage A keeps the half steps because of the support floor (even F 0.019 suffices) and its smaller
steps (32 rows). In effect, online rlcr21 without a floor is rlcr11 at +5% step time.

## Run ladder (exact only; dev only; one RTX 4090 via gpuq)

| # | run | pair (rlcr11, existing) | GPU-h | decision |
|---|---|---|---|---|
| 0 | `smoke_rlcr21.sh`: scorer check on RLCR / RLVR (rlcr21 vs rlcr11 greedy agreement), A2 + C0 at 2-4 steps | | ~0.5 | plumbing, exit codes |
| 1 | Stage A `A2` exact, seed 102 | `9d2314faca` (dev 0.153 / AUROC 0.852 @1000) | ~1.8 | curve tracks or trails? half-step use? |
| 2 | Stage A `A2r1`, `A2r2` (1102, 2102) | `3e9453e9de` 0.168, `7656f17798` 0.172 | ~3.6 | 3-seed paired Brier/AUROC/ECE; equal within ±0.01 Brier = no cost |
| 3 | Stage C `C0` exact, seed 300 (from released RLCR) | rlcr-prereg-C C0 exact (+ r1, r2) | ~7 | the RLCR-aligned comparison |

Run 1 alone answers question 1, and the step-1000 checkpoint is ready ~2 h after it starts. Run 2 only if run 1 is
not clearly worse. Run 3 is the expensive one; queue it only if 1-2 show no cost.
Test is not scored in this ladder. If rlcr21 is kept, a test comparison against rlcr11 at a dev-selected step
gets its own written rule first, as in the preregs.
