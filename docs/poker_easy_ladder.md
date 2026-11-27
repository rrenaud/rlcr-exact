# Easy-game ladder: datasets and fastest models (2026-11-27)

Generator: `src/rlcr_exact/easy_games.py`. Data: `scripts/easy_data.py data/easy_v1` (20k train / 1k dev / 2k test
per game, with exact p_win, one sampled outcome, claim-format prompt text, and Qwen2.5 prompt_ids). Fits:
`scripts/easy_fit.py data/easy_v1` (CPU, 8 threads, ~5 min for everything; results in data/easy_v1/fit_results.json).

| game | what varies | distinct p | needs |
|---|---|---|---|
| g1_high | one card each, high card wins | 13 | rank order, counting |
| g1b_high2 | two cards each, compare high then low | 91 | lexicographic compare, counting |
| g2_leduc | 6-card deck, one hole + one board card | 6 | the idea of pairing the board |
| g3_three | three-card poker vs a random hand | ~1,500 | 3-card hand categories, counting |
| g4_river | hold'em river, both hands face up | 2 (win/lose) | best 5 of 7, compare |
| g5_turn | hold'em turn, both face up | 45 (k/44) | best 5 of 7 per river, count outs |
| e2_akv88 | A♠K♦ vs 8♥8♣, flop | ~180 | board texture (A/K, 8, suits) |

## Fastest models (test, against exact p_win)

Models: logistic regression and MLP 256x256 on role-tagged one-hot cards (156 dims), and a set transformer (card and
role embeddings, 2 layers, d 64). 1,500 steps each, sigmoid output on squared error. Targets: exact p_win, or the
one sampled outcome.

| game | const MSE | logreg exact | MLP exact | MLP sampled | set-TF exact | set-TF sampled | floor (Brier vs oracle) |
|---|---|---|---|---|---|---|---|
| g1_high | 0.086 | 0.003 | **0.000** | 0.001 | 0.000 | 0.001 | 0.163 |
| g1b_high2 | 0.089 | 0.006 | **0.000** | 0.007 | 0.000 | 0.006 | 0.161 |
| g2_leduc | 0.079 | 0.051 | **0.000** | 0.000 | 0.000 | 0.001 | 0.159 |
| e2_akv88 | 0.097 | 0.012 | **0.000** | 0.021 | 0.000 | 0.011 | 0.150 |
| g3_three | 0.085 | 0.060 | **0.001** | **0.047** | 0.004 | 0.051 | 0.166 |
| g5_turn | 0.147 | 0.135 | 0.107 | 0.203 | 0.113 | 0.176 | 0.102 |
| g4_river | 0.250 | 0.239 | 0.267 | (= exact) | 0.232 | (= exact) | 0 |

Values are test MSE vs exact. Correlation and decision accuracy are in fit_results.json. On g4 the sampled and
exact targets are identical, because the river is deterministic.

- **Easy for tiny models:** g1, g1b, g2, e2 (MSE ~0 with exact targets). Caveat: these have few states (52 to
  17k), so test states also appear in train. The models are learning a table, which is exactly what makes the
  exact-vs-sampled gap visible.
- **Exact vs sampled is starkest on g3** (22,100 states, 20k train rows): MLP MSE 0.001 exact vs 0.047 sampled
  (correlation 0.99 vs 0.79). With about one noisy label per state, sampled targets cannot be averaged away.
  The same holds on g5 (0.107 vs 0.203; sampled is worse than the constant).
- **Hand reading is the hard part, not uncertainty.** g4 (deterministic river) is the hardest game for
  every model: accuracy 0.70 at best, and the MLP is worse than the constant. g5 inherits that.
  Best-of-7 selection plus comparison needs more data, steps or depth than these defaults. That is the
  natural test bed for looped or scratchpad models.
- **Difficulty order for small models:** g1 < g2 ≈ g1b < e2 < g3 < g5 < g4. For the oracle, uncertainty
  makes g5 "harder" than g4. For a learner, g4's combinatorics dominate.

## Capacity on the hard rungs (scripts/easy_capacity.py, CPU 4 threads, half steps: 2-3k steps of batch 512)

Test sets are the same as above (data/easy_big test == data/easy_v1 test). Results are in data/easy_big/capacity.jsonl.

| config | MSE vs exact | corr | accuracy | time |
|---|---|---|---|---|
| g4 river, set-TF 64x2, 20k | 0.271 | 0.42 | 0.70 | 3 min |
| g4 river, set-TF 64x2, 200k | 0.117 | 0.73 | 0.85 | 6 min |
| g4 river, set-TF 128x4, 200k | 0.063 | 0.87 | 0.92 | 9 min |
| **g4 river, set-TF 128x4, 200k + aux hand categories** | **0.037** | **0.92** | **0.96** | 6 min |
| g4 river, looped 128 (1 shared layer), T 1-8, 200k | 0.087 | 0.81 | 0.89 | 5 min |
| g4 river, MLP 1024, 200k | 0.129 | 0.72 | 0.84 | 0.5 min |
| g5 turn, set-TF 128x4, 200k, exact | **0.017** | 0.94 | 0.97 | 5 min |
| g5 turn, set-TF 128x4, 200k, sampled | 0.032 | 0.89 | 0.94 | 5 min |
| g5 turn, looped 128, T 1-8, 200k, exact | 0.025 | 0.91 | 0.95 | 5 min |
| g3 three-card, MLP 256, 200k, exact | **0.0006** | 1.00 | 0.98 | 4 s |
| g3 three-card, MLP 256, 200k, sampled | 0.0052 | 0.97 | 0.95 | 4 s |

The looped model at test loop counts T (accuracy / MSE):

| | T=1 | T=2 | T=4 | T=8 (train max) | T=16 |
|---|---|---|---|---|---|
| g4 river | 0.855 / 0.122 | 0.885 / 0.096 | **0.893 / 0.087** | 0.892 / 0.087 | 0.885 / 0.093 |
| g5 turn | 0.921 / 0.042 | 0.954 / 0.029 | **0.958 / 0.024** | 0.950 / 0.025 | 0.945 / 0.028 |

- **Hand reading is learnable.** The 20k defaults were simply too small. 10x data takes river accuracy from
  0.70 to 0.85, depth to 0.92, and auxiliary hand-category targets to 0.96. Telling the model what each player
  holds is the biggest single lever, i.e. a scratchpad-like intermediate label.
- **Looping helps up to about 4 loops, then plateaus**, and running past the trained range (16) slightly hurts.
  One shared layer looped 4x approaches the 4-layer model with a quarter of the block parameters, but does not
  extrapolate to "think longer".
- **Exact beats sampled at every scale tried.** On the turn: 0.017 vs 0.032 (2x) at 200k. On three-card: the
  gap shrinks with 10x data (0.001 vs 0.047 at 20k, 0.0006 vs 0.0052 at 200k), but sampled is still ~9x worse
  with ~9 noisy labels per state.

## 7B zero-shot on the ladder (gpuq run-20261127T111811Z-0b26a0ff91, 300 dev spots per game x k = 4, claim format)

Spearman of stated equity with exact p / Brier vs oracle. The constant guess scores Brier ~0.25 on every game.

| game | Qwen2.5-7B base | RLVR | released RLCR |
|---|---|---|---|
| g1 high card | **0.26** / 0.333 | 0.11 / 0.407 | 0.22 / 0.301 |
| g1b two-card high | 0.07 / 0.347 | 0.09 / 0.355 | 0.09 / 0.288 |
| g2 Leduc | 0.06 / 0.356 | -0.04 / 0.405 | 0.13 / 0.270 |
| g3 three-card | 0.00 / 0.379 | -0.01 / 0.352 | 0.06 / 0.320 |
| g4 river | 0.01 / 0.440 | 0.03 / 0.446 | 0.05 / 0.342 |
| g5 turn | 0.04 / 0.376 | -0.04 / 0.404 | 0.04 / 0.318 |
| e2 AK vs 88 | 0.07 / 0.367 | 0.09 / 0.379 | 0.10 / 0.279 |

- **Every model is worse than the constant on every game, even high card**, where the truth is 4r/51 for rank
  r. The models get the direction: stated equity rises from ~0.57 for a deuce to ~0.8 for an ace. But they
  compress the whole range into 0.5-0.85 and miscount the deck. One example: "10 cards for each of the 4 lower
  ranks: 2, 3, 4, 5" for a 9.
- **Released RLCR is the least overconfident** (lowest Brier everywhere) but no better at discrimination.
- **So the 7B lacks card arithmetic, not just poker.** A tiny MLP solves g1 exactly from 20k labels in about 2 s.
  The cheapest real test of exact vs sampled confidence training at 7B scale is g1 / g1b / g2: the targets are
  learnable, the starting point has weak signal, and there are few states.

## Next

- **7B zero-shot on the ladder** (prompt_ids are already in the data): one gpuq run. Can the 7B at least do
  g1 and g2?
- **g4 / g5 with more capacity:** more steps and data, a deeper set transformer, a looped transformer
  (accuracy vs loops), and auxiliary hand-category targets.
