# Poker prior work (survey, 2026-11-27)

Done by a research subagent; [V] = page read, [S] = search snippet only. Not independently re-checked.

**Bottom line.** No prior work trains or scores an LLM's stated confidence that it wins at showdown against exact
enumerated equity with a proper score. The closest is Poker Arena (arXiv 2606.13815, [V]): evaluation only, MAE of
equity stated in free-form reasoning against engine Monte Carlo. Released poker checkpoints learned action labels,
not equity, and none is Qwen2.5-7B. So the contamination risk is low, apart from memorized preflop matchup tables
(AA vs KK). v1 is therefore postflop only.

## Trained poker LLMs (all action/decision tasks)
- PokerBench (arXiv 2501.08328, AAAI-25) [V]: 11k solver-labelled decisions; SFT on Llama-3-8B, Llama-2-7B and
  Gemma-2B; no equity labels. Data: github.com/pokerllm/pokerbench, HF RZ412/PokerBench.
- YiPz/qwen3-4b-pokerbench-grpo [V]: Qwen3-4B, SFT then GRPO with an LLM-judge action reward (Apache-2.0).
- ToolPoker (arXiv 2602.00528, ICLR-26) [V]: behavior cloning plus RL fell short; solver tools added.
- SPIRAL (arXiv 2506.24119) [V/S]: self-play RL on Kuhn poker transfers to math (MIT code).
- Also [S]: PokerGPT (2401.06781), MemoPilot (2606.08656), ECHO-2 (2602.02192).
- Emergent World Beliefs (2512.23722) [V]: a model pretrained on hand histories encodes equity in probes.

## Benchmarks
- GTO Wizard Benchmark (2603.23660) [V]: full HUNL against GTO Wizard AI, scored by win rate.
- PokerSkill (2605.30094) [V]: prompt and skill library, no training.
- chrystians/poker-holdembench [V]: has a self-stated `equity_estimate` field, null in the rows checked.

## Calibration against exact chance
- Dice, coin and sampling-fidelity papers (2511.00620, 2601.05414, 2506.09998) [S]: about sampling, not stated
  confidence.
- Probabilistic Calibration Is a Trainable Capability (2605.11845) [V]: trie soft targets for sampling
  distributions; method is close to our trie work.
- None found on stated confidence against exact card probabilities.

## Gyms and libraries
- Multi-step poker environments: TextArena (MIT; wrapped by verifiers), RLCard, PettingZoo, OpenSpiel, PokerKit
  (MIT, best game engine).
- None is single-step equity. reasoning-gym and prime-environments have no poker task.
- Evaluators: eval7 (MIT, Cython, exact hand-vs-range), phevaluator (Apache-2.0, fast C), treys (pure Python).
- We use our own numpy evaluator (src/rlcr_exact/poker.py, tested against a reference). eval7 or phevaluator is
  the upgrade path if speed matters.

# Pro and poker-system evals, human baselines (survey 2, 2026-11-27)

**AI vs pros.** These are scored by win rate, not equity:
- Claudico (2015): the humans' edge was not significant.
- DeepStack (Science 2017): about 486 mbb/g scored with AIVAT; histories are in the supplement [S].
- Libratus (2017): 14.7 bb/100 over 120k hands [S].
- Pluribus (Science 2019): about 5 bb/100 with AIVAT, p = 0.021; its 10k hands are Data File S1 [V].
- AIVAT and duplicate deals are the variance-reduction tools.

**Hand histories.** The PHH dataset (github.com/uoftcprg/phh-dataset, MIT; full set on Zenodo) [V] holds:
- all 10k Pluribus hands, with every hole card visible;
- 21.6M HandHQ online no-limit hands (2009, 25NL-1000NL), with villain cards mostly shown only at showdown;
- the ACPC logs (Zenodo only);
- 83 televised WSOP hands.

PokerKit reads them. These are realistic spot distributions to replace uniform random deals. The IRC poker
database is play money, with no license stated [S].

**Human equity estimation, the human baseline.** None of these studies uses a proper scoring rule:
- Liley & Rakow 2010, J. Behav. Decis. Making 23:496, doi:10.1002/bdm.670 [S]. 69 moderately experienced players
  estimated P(best hand) preflop and on the flop against 1-9 opponents, compared with simulation. They were
  generally accurate and slightly overconfident when P < 0.7, and paid too little attention to the number of
  opponents. This is the closest match to our task.
- Zhu et al. 2022, Cognition 223:105022, PMC8987733 [V]. Experts (186, from twoplustwo) estimated flop-event
  frequencies much better than novices or amateurs. Scored by summed absolute error. OSF prereg osf.io/8scg4.
- Linnet et al. 2010, J. Gambling Issues 24 [S]: experienced players showed less estimation bias.

**LLMs against bots.** GTO Wizard Benchmark (arXiv 2603.23660) [V]: every LLM loses (GPT-5.3 xhigh -16 bb/100
was best). Husky Hold'em Bench and Kaggle Game Arena poker [S]. No study pits LLMs against human pros.

**Equity-trainer apps** publish no accuracy numbers. No benchmark measures equity estimation by solvers or bots.

**Use.** (a) Human baseline: compare MAE and bias with Liley & Rakow on preflop and flop spots. For a Brier
baseline, ask the authors for item-level data, or run our dev spots on human players. (b) Realistic spots: PHH via
PokerKit (Pluribus for fully visible hands, HandHQ for realistic boards and action lines).
