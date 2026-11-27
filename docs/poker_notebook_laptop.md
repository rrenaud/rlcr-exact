# Running the AK vs 88 notebook on a laptop, driven by a browser agent

`notebooks/poker_akv88.py` is a marimo notebook for the toy version of the poker equity task: one fixed matchup,
flop only, exact equity for all 17,296 flops. It runs on CPU only and never touches a GPU (it sets
`CUDA_VISIBLE_DEVICES=""` before torch loads). A laptop is the intended host. The browser and the server then sit
on the same machine, which avoids an SSH tunnel and competition with gpuq jobs on Prune.

## 1. Setup (once)

```sh
git clone -b explain-to-expert-poker-player git@github.com:rrenaud/rlcr-exact.git
cd rlcr-exact
python3 -m venv .venv && . .venv/bin/activate
pip install marimo numpy matplotlib torch   # the CPU build of torch is enough
```

Only numpy and torch are used, through `src/rlcr_exact/{poker,poker_toy,betting}.py`. Exact enumeration is cached
in `~/.cache/poker_toy/` (one ~0.3 MB file per matchup). The first load of a matchup takes a few seconds, later
loads are instant.

## 2. Start the server

Choose app mode or edit mode.

- **App mode** shows outputs and controls with the code hidden. Use it for a browser agent, which then cannot
  edit or save the notebook by accident:
  ```sh
  marimo run notebooks/poker_akv88.py --headless --port 2718
  ```
- **Edit mode** shows the code, which you can change. marimo auto-saves edits to the file:
  ```sh
  marimo edit notebooks/poker_akv88.py --headless --port 2718
  ```

marimo prints a URL like `http://localhost:2718?access_token=...`. Give the agent that exact URL, token included.
On a laptop you may add `--no-token` for a plain `http://localhost:2718`. Do this only on a machine where nobody
else can reach the port: the server binds to localhost by default, so keep it that way and don't pass
`--host 0.0.0.0`. Stop the server with Ctrl-C.

To use Prune instead of the laptop, run the same command there from
`~/rlcr-exact/explain-to-expert-poker-player` with `~/envs/pytorch/bin/marimo`, and forward the port:
`ssh -L 2718:localhost:2718 prune`.

## 3. Brief for the browser agent

Paste this to the agent, together with the URL.

> Open the URL. It is a marimo notebook about poker equity. Every control re-runs the cells that depend on it
> automatically; wait until the "running" indicators and spinners stop before reading outputs. Read numbers from
> the text and tables (they are real text). Take screenshots for the plots. Do not edit code.
>
> Controls, top to bottom:
> - **Matchup** (dropdown): which two hands play. The default is "A♠K♦ vs 8♥8♣ ...".
> - **sample seed** (slider 0-99): redraws the table of 8 random flops with 12 sampled showdowns each.
> - **model** (dropdown: linear, MLP 8, MLP 32, MLP 128, MLP 512), **features** (onehot or onehot+counts),
>   **k runouts per target (990 = exact)** (1, 2, 4, 8, 16, 64, 256, 990), **train flops** (500-14000),
>   **epochs** (50-1000), **seed** (0-9). These retrain the model, which takes under 2 s on a laptop.
> - **Run exact-vs-sampled sweep (~1 min)** (button): 32 fits. Wait for the two-panel plot.
> - **Evaluate the trained model on** (dropdown): a transfer test on another matchup.
>
> Sliders: click the slider, then press the arrow keys to move one step, or Home/End for the ends. Check the
> value shown next to it.

## 4. Checkpoints (default matchup A♠K♦ vs 8♥8♣)

The agent should confirm each one. Fit numbers can differ in the third decimal across machines (CPU torch
threading), and that is fine.

1. **Header:** "Before the flop you win **44.3%** outright, ties 0.3%", "17,296 flops", "Only **188** distinct equity
   values occur".
2. **Histogram:** the legend reads "opponent makes three eights" (2,116), "no ace, king or eight on the flop"
   (6,940), "no ace, king or eight, but two cards of one of your suits (flush draw)" (2,940) and "you pair your
   ace or king" (5,300). There are three modes, near 0.05, 0.25 and 0.9, with no flops between about 0.65 and 0.8.
3. **Default fit** (MLP 128, onehot+counts, k = 1, 12000 flops, 300 epochs, seed 0): test MSE vs exact ≈
   **0.010**, correlation ≈ 0.953, REL ≈ 0.004. The "oracle" row has Brier vs oracle 0.1497.
4. **Set k to 990:** MSE ≈ **0.0013**, correlation ≈ 0.993, REL ≈ 0.0002. The scatter collapses onto the
   diagonal. This is the exact-vs-sampled effect.
5. **Sweep button:** in the left panel, with k = 1 the bigger models are worse (MLP 512 highest). At k = 990 the
   order flips (MLP 512 lowest, about 0.001). In the right panel, the exact curve sits far below the k = 1 curve
   at every training size.
6. **Transfer** (default target A♠K♦ vs 7♥7♣, model from checkpoint 3): MSE ≈ 0.07, correlation ≈ 0.68.

If a checkpoint fails, the agent should report the text it saw and stop, not retry with other settings.

## 5. Things to try

- Model "linear" with k = 990. It plateaus at a correlation of about 0.92, because it cannot represent
  flush-draw interactions.
- Train flops at 500. Compare k = 1 with k = 990: exact targets on 500 flops beat k = 1 on 14,000.
- Matchup "J♠T♠ vs A♥A♣" or "A♥K♥ vs 8♠8♣". Flush draws matter more there, and the histogram changes shape.
