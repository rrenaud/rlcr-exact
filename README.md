# rlcr-exact

Does rewarding a model's stated confidence by its **exact expected reward** make RL calibration training work
better than the single sampled confidence RLCR uses?

- RLCR: Damani et al., *Beyond Binary Rewards*, arXiv 2507.16806, code [damanimehul/RLCR](https://github.com/damanimehul/RLCR).
- The exact objective comes from [rrenaud/rewarding_doubt](https://github.com/rrenaud/rewarding_doubt).

The full work log is [NOTES.md](NOTES.md), and the online-training design is
[docs/stage_b_design.md](docs/stage_b_design.md). This page summarises them. Findings are as of 2026-10-25.

## The idea

RLCR's completions end in `<confidence>c</confidence>`, and each rollout is scored with

    R(a, c) = 0.5 format + 0.5 correct + 0.5 (1 - (c - y)^2)

GRPO samples one confidence c (k = 1) and uses that one reward as the advantage for every token.

The model's distribution q(c | a) over the stated confidence is cheap to read exactly. We restrict confidences to the
11 strings `0.0 ... 1.0` RLCR's prompt asks for, and score all 11 from the prefix with one packed forward (a
token trie, `src/rlcr_exact/trie_score.py`). Off-list mass earns 0. Then

    J = E_a E_{c ~ q} R(a, c)
    grad J = E_a[ grad log pi(a) * r(a) ] + E_a[ grad_q sum_c q(c|a) R(a, c) ],   r(a) = sum_c q(c|a) R(a, c)

The two arms differ as follows:
- **Exact:**
  - the answer tokens get REINFORCE with r(a), Rao-Blackwellised over the confidence;
  - the confidence tokens get the analytic gradient over all 11 levels.
- **Sampled (k = 1):** the same code with RLCR's estimator. One c is drawn from q, and R(a, c) drives both the answer
  and the confidence tokens.

Everything else matches RLCR unless named as a knob: the reward weights, the group-mean baseline without std
scaling, the tabc_long prompt and the HotpotQA data. See [Deviations from RLCR](#deviations-from-rlcr).

## Findings

### Stage A: frozen answers, train only the confidence

- **Math is a dead end.** The math model writes its number in `<analysis>` and the tag copies it (p 0.9997). q is
  one-hot, so neither estimator has a gradient.
- **HotpotQA keeps q uncertain** at the tag (entropy about 0.66 nats), so we moved there.
- **The released RLCR-hotpot checkpoint has nothing left to gain.** No config beat the untrained model beyond noise.
  The test bed became the **RLVR-hotpot checkpoint** (`mehuldamani/hotpot-v2-correctness-7b`), which has the same data
  and a correctness-only reward. It is confident almost everywhere, with AUROC about 0.55, so its gap to RLCR is
  calibration alone.
- **An answer-KL penalty is needed, but small.** Without one (W=0) the shared LoRA drifts the answers 30+ nats.
  W=1 stalls, because a bf16 packed-vs-unpacked reference mismatch makes its gradient mostly noise. W = 0.01–0.1
  works: dev Brier goes 0.41 to about 0.18 and AUROC to about 0.80.

### Stage B: answers and confidence trained online (exact vs k = 1)

The setup:
- start from the RLVR checkpoint with a fresh LoRA (rank 4 on layers 14–27);
- 16 questions × 8 rollouts per step, lr 1e-4, W 0.02, RLCR's reward balance, on-policy vLLM rollouts.

Two seeds per arm. Dev means greedy generation on 500 held-out dev questions, with the confidence parsed from the
text. Test is the paper's 500 HotpotQA test questions, bootstrapped against the released RLCR model (Brier 0.188,
AUROC 0.798).

| step 300 | dev EM | dev Brier | dev AUROC | test Brier | test AUROC |
|---|---|---|---|---|---|
| exact, seed 0 | 0.528 | **0.174** | **0.806** | **0.204** | **0.756** |
| exact, seed 1 | 0.536 | 0.187 | 0.796 | — | — |
| sampled, seed 0 | 0.532 | 0.235 | 0.656 | 0.257 | 0.663 |
| sampled, seed 1 | 0.534 | 0.248 | 0.573 | 0.238 | 0.565 |
| exact, step 600, seed 0 / 1 | | | | 0.220 / 0.199 | 0.773 / 0.774 |

- **Exact calibrates and k = 1 mostly doesn't.** Against RLCR, the sampled arm's test AUROC is lower by 0.13 [0.09,
  0.18] for seed 0 and 0.23 [0.19, 0.27] for seed 1. Exact continued to step 600 gets within 0.024 of RLCR's AUROC,
  with about 1/16 of its rollouts per step and a LoRA instead of a full fine-tune.
- **Accuracy is flat in every run.** Neither arm trades answers for calibration.
- **The arms split late, not early.** They are indistinguishable up to step 100, and at step 200 all four runs are
  at 0.58–0.71 dev AUROC. The split happens between steps 200 and 300.
- **Exact's advantage is not variance reduction in the answer reward.** The noise a sampled confidence adds to the
  answer reward is Var_q[R]. Comparing it with the spread of the exact reward across a question's rollouts
  (300-step seed-0 runs; the exact one is the rank-8 run `6f14d21b79`):

  | steps | sampled: noise / signal | exact: noise / signal | listed mass |
  |---|---|---|---|
  | 1–50 | 1.38 | 1.11 | 0.90–0.94 |
  | 51–100 | 0.12 | 0.08 | 0.999 |
  | 101–300 | 0.07–0.10 | 0.04–0.08 | 1.000 |

  The draw noise is large while the format is being learned. That is mostly off-list draws scoring 0, and the
  arms don't differ then. By the time the arms split, q is nearly one-hot and the draw adds under 10% noise. So
  "var(q) goes to 0, so exact can't help much" is right about the answer reward but does not match the outcome.
- **Where the difference likely comes from, untested:** the confidence-token gradient. With q peaked, a k = 1 draw
  is almost always the mode, so the sampled loss pushes only on the level it already prefers and mass rarely moves
  between levels. Exact moves every level by q_l (R_l − E_q R) on every rollout. Consistent with this, sampled seed 1
  ranks well in q (dev expected-confidence AUROC 0.75) but not in its greedy text (0.57); for exact the two agree.

### Caveats

- There are two seeds per arm, and exact seed 1 has no test eval at step 300.
- The sampled arm has no step-600 continuation.
- In the step-600 continuation the exact run's mean confidence swings between evals, so its Brier worsens even as
  AUROC improves.
- bf16 packed scoring matches per-row scoring on 87.5% of greedy picks, with gradient cosine 0.97. Compare
  aggregates, not traces.

## In progress

- **RLCR-matching pieces from the RLVR start:**
  - linear lr warmup (`--warmup-steps`);
  - the whole policy at temperature 0.7, as RLCR's trainer does (`--policy-temperature`);
  - 32 rollouts per question, split into memory-sized parts (`--micro-batch`, `--part-tokens`).

  At 16 × 32 a step takes about 190 s on one RTX 4090.
- **k = 1 at low lr:** with warmup at lr 5e-5, k = 1 learned the confidence format slowly (listed mass 0.19 at
  step 10, against 0.96 at lr 1e-4), and its accuracy lagged. Next step: start both arms after a short
  format-learning phase, so the comparison is about calibration.
- **Long-term:** an RLCR-comparable run from `Qwen/Qwen2.5-7B` base on RLCR's 20k training questions
  (`scripts/build_rlcr_train.py`, `--train-questions`, `TASK=hotpot-base-11`). That is 1250 steps × 16 × 32, about
  2 GPU-days per arm.

## Deviations from RLCR

| knob | RLCR | here |
|---|---|---|
| start | Qwen2.5-7B base | RLVR-hotpot checkpoint (the base model is prepared as `hotpot-base-11`) |
| update | full fine-tune, lr 1e-6, ZeRO-2 | LoRA rank 4 (layers 14–27), lr 1e-4 |
| batch | 16 questions × 32 rollouts | 16 × 8 (16 × 32 in progress) |
| KL | none (beta 0) | answer-token KL to the adapter-off model, W 0.02 |
| temperature | 0.7, logits tempered in training | 1.0 in the results above; `--policy-temperature 0.7` in progress |
| confidence | sampled number, k = 1 | exact over 11 listed levels; k = 1 as the comparison arm |

## Layout

- `src/rlcr_exact/`:
  - `trie_score.py`: packed scoring of the listed confidences;
  - `confidence.py`: RLCR's confidence strings and rewards, and the exact objective (`sampled_loss` is in `train_frozen.py`);
  - `grading.py`: answer grading (RLCR's exact match, plus token F1);
  - `online.py` and `shared_vllm.py`: on-policy rollouts with vLLM sharing the trainer's weights.
- `scripts/`:
  - `train_frozen.py`: the trainer for Stage A and B;
  - `run_frozen.sh`: the task presets;
  - `gen_eval*.py`: greedy generation evals;
  - `build_*`: data;
  - `check_*` and `smoke_*`: correctness checks.
- `tests/`: run with `PYTHONPATH=src python -m pytest -q tests/`.

`data/` is git-ignored; rebuild it with the `scripts/build_*` scripts. Every run in the notes was submitted to a
local GPU queue (`gpuq`), and run IDs refer to its run directories.
