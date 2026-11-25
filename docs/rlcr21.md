# rlcr21: exact confidence on a 0.05 grid

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
