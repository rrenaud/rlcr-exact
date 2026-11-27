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

## Next

- **7B zero-shot on the ladder** (prompt_ids are already in the data): one gpuq run. Can the 7B at least do
  g1 and g2?
- **g4 / g5 with more capacity:** more steps and data, a deeper set transformer, a looped transformer
  (accuracy vs loops), and auxiliary hand-category targets.
