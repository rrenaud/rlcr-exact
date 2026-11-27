# Poker equity task: zero-shot results (2026-11-27)

Dev set: `data/poker_v1_*` (500 uniform random postflop spots, villain known or random 50/50; exact oracle p_win;
gold = one sampled showdown). Stated equity e = the model's P(win). Scored against the oracle with
`scripts/poker_analyze.py`. Baselines on the same rows: a constant (dev mean) and the oracle.

## Qwen2.5-7B starting points: no signal

Local, via gpuq: `claim` run-20261127T091259Z-bc5d9cbbe0 (base and RLVR are copies from the preempted
run-20261127T084839Z-797deefb9f); `rlcr` run-20261127T091259Z-2749cd5943. 500 spots x k = 8 at T 0.7, RLCR's
tabc_long prompt; all exit 0.

| model | format | parsed | Spearman e vs p | AUROC vs outcome | MAE | Brier vs oracle |
|---|---|---|---|---|---|---|
| Qwen2.5-7B base | claim | 0.94 | -0.00 | 0.50 | 0.42 | 0.40 |
| RLVR | claim | 0.99 | -0.01 | 0.49 | 0.43 | 0.42 |
| released RLCR | claim | 0.99 | -0.02 | 0.50 | 0.35 | 0.31 |
| Qwen2.5-7B base | rlcr | 0.81 | 0.04 | 0.53 | 0.38 | 0.37 |
| RLVR | rlcr | 0.99 | 0.09 | 0.53 | 0.39 | 0.38 |
| released RLCR | rlcr | 0.99 | 0.12 | 0.55 | 0.31 | 0.28 |
| constant | | | 0 | 0.50 | 0.31 | 0.25 |
| oracle | | | 1 | 0.90 | 0 | 0.13 |

- **Every model scores worse than the constant guess.** Their confidences are overconfident noise.
- **RLCR's Brier training lowers overconfidence** (0.31 vs 0.40), but it adds no discrimination.
- **The only signal is in the rlcr format, on face-up spots.** Released RLCR reaches Spearman 0.16 there, from the
  yes/no choice on easy face-up rivers. On random-villain spots it is 0.02.
- **The exact-q confidence** (E_q over the rlcr11 grid) is within 0.01 of the text confidence.
- **The headroom is all discrimination, starting from zero.** A confidence-only reward has nothing to sharpen
  here, so equity skill has to come from somewhere first: supervised training against the oracle, or RL on the
  answer.

## Inkling-2-Mini (Tinker, free): enumerates in its head

200 spots x k = 2, `claim` format (`scripts/tk_poker_eval.py`), outputs in /data/gpuq-runs/rrenaud/poker-zeroshot/tinker.

| prompt / budget | parsed | mean tokens | MAE | Spearman | Brier vs oracle | random-villain Spearman |
|---|---|---|---|---|---|---|
| no guide, 16k cap | 0.35 | 13.5k | 0.085 | 0.86 | 0.120 | 0.82 (12 rows) |
| guide v1 (`estimate`), 16k | 0.44 | 12.6k | 0.247 | 0.45 | 0.270 | |
| guide v2 (`estimate2`), 16k | 0.34 | 13.6k | 0.061 | 0.88 | 0.099 | 0.83 (12 rows) |
| v2 + budget forcing 1k | 0.34 | 2.5k | 0.284 | 0.32 | 0.267 | 0.21 |
| v2 + budget forcing 2k | 0.50 | 3.2k | 0.250 | 0.43 | 0.246 | 0.13 |
| v2 + budget forcing 4k | 0.72 | 4.7k | 0.198 | 0.60 | 0.207 | 0.34 |

- **It computes rather than estimates.** Left alone, it enumerates runouts and only finishes the cheap spots
  (face-up rivers about 100%, face-up turns 76%, random-villain flops 5%). Where it finishes it is good.
- **Prompt text does not shorten its native thinking.** Guide v1's word budget was ignored, and v1 hurt accuracy:
  half its "no" answers switched to meaning P(answer right). Guide v2 fixes that meaning (best Brier, 0.099) but
  not the length.
- **Budget forcing controls length** (`--think-budget`: cut the thinking, append a commit line, close the
  thinking channel). The forced estimates are weak, though. On random-villain spots they are close to the
  constant (Brier ~0.25 at every budget, Spearman 0.1-0.3). The model has no usable at-the-table intuition for
  equity against a random hand; its quality comes from computing.
- **The forced answer often keeps reasoning** in the text channel and hits the 2,048-token answer cap. That is
  why only 34-72% of forced completions are parsed.

## By difficulty: no easy stratum exists for the 7B models

Answer accuracy (does the stated equity land on the right side of 0.5) by stratum, rlcr format, against the
majority-answer rate:

| model | river, face up | turn, face up | flop, face up | river, random | turn, random | flop, random |
|---|---|---|---|---|---|---|
| base | 0.49 (maj 0.55) | 0.54 (0.53) | 0.52 (0.54) | 0.51 (0.52) | 0.53 (0.54) | 0.53 (0.51) |
| RLVR | 0.53 | 0.54 | 0.58 | 0.54 | 0.51 | 0.51 |
| released RLCR | 0.57 | 0.57 | 0.59 | 0.52 | 0.52 | 0.50 |
| Inkling, 16k | 0.93 | 0.95 | (1 row) | 0.90 | – | (2 rows) |

**Face-up rivers by how far apart the two hands are.** "Who wins" is deterministic there. Even with a gap of 3+
hand categories (e.g. a flush against a pair), the 7B models are at chance (0.46 / 0.55 / 0.59). Inkling is at
0.80-0.90. So the 7B starting points cannot read poker hands. Filtering spots by model pass rate is impossible
(every stratum is at chance), so an easier set has to come from the task's structure:

- **E0** river with the villain face up: who wins, no uncertainty in the cards. Confidence measures the
  model's own uncertainty, as in HotpotQA.
- **E1** turn with the villain face up: equity = outs / 44.
- **E2** (recommended) the fixed matchup A♠K♦ vs 8♥8♣, flop only. A short prompt and a function of three
  cards with trimodal exact equity, i.e. notebooks/poker_akv88.py at LLM scale. The toy predicts the
  exact-vs-sampled outcome, which makes it the sharpest short train/test: ~4k train flops, 1k held-out test
  flops.

Tinker so far: zero-shot only (Inkling-2-Mini free tier). No poker training on Tinker or locally yet.

## Next

- **Supervised against the exact oracle** on Qwen2.5-7B: SFT to the grid, exact expected Brier vs p, and the
  same loss vs the sampled outcome. This is the ladder from notebooks/poker_akv88.py at LLM scale; the toy
  predicts that exact targets beat sampled ones by a wide margin once the model can memorize.
- **Toy first:** the AK vs 88 notebook already shows sampled-label noise reversing the ranking of model sizes.
