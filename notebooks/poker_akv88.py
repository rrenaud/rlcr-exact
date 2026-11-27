import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="AK vs 88: a toy equity task")


@app.cell
def _():
    import os, sys, time
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never touch a GPU outside gpuq (shared hosts)
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from rlcr_exact import poker
    from rlcr_exact import poker_toy as T
    from rlcr_exact.betting import Score

    CALLER = Score("caller")
    COLORS = {"villain sets": "#8a5cc2", "miss": "#7a838c", "miss + flush draw": "#2a78d6", "hero pairs": "#eb6834"}
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return CALLER, COLORS, T, mo, np, plt, poker, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AK vs 88: the equity task, small enough to see whole

    The full poker dataset deals random hands to both players. Here both hands are **fixed** and only the flop
    varies. Every one of the 48C3 = 17,296 flops is enumerated with all 990 turn-and-river runouts, so the
    **exact equity of every state is known**. A model only sees the three flop cards and must output P(you win
    outright at showdown).

    *Terms.* **You** hold the first hand and the **opponent** the second. Both are all-in, so all five board
    cards are dealt and nobody folds. **Equity** here is the probability that you win the pot outright; a split
    counts as not winning. The **flop** is the first three board cards, and the **runout** is the last two (turn
    and river). Poker writing calls you the *hero*, the opponent the *villain*, and three of a kind made with a
    pocket pair a *set*. This notebook avoids those words.

    Questions this toy answers in seconds:

    - **Capacity:** what does a linear model miss that a small MLP gets?
    - **Exact vs sampled targets:** train on one sampled showdown per flop (what RLCR sees: a 0/1 outcome) or on
      the exact equity. The dial *k* averages *k* distinct runouts per flop: *k* = 1 is one showdown, *k* = 990 is
      exact. Exact targets are the conditional expectation of the sampled label (Rao-Blackwell), the same idea as
      the project's exact confidence objective.
    - **Calibration vs discrimination:** with the truth known, the Murphy split is measured against the real
      equity, not estimated from labels.
    """)
    return


@app.cell
def _(mo):
    MATCHUPS = {
        "A♠K♦ vs 8♥8♣ (ace-king of two suits vs a pair of eights)": ("As Kd", "8h 8c"),
        "A♥K♥ vs 8♠8♣ (ace-king of one suit vs a pair of eights)": ("Ah Kh", "8s 8c"),
        "A♠Q♦ vs 9♥9♣ (ace-queen vs a pair of nines)": ("As Qd", "9h 9c"),
        "A♠K♦ vs 7♥7♣ (ace-king vs a pair of sevens)": ("As Kd", "7h 7c"),
        "K♠Q♦ vs 2♥2♣ (king-queen vs a pair of twos)": ("Ks Qd", "2h 2c"),
        "J♠T♠ vs A♥A♣ (jack-ten of one suit vs a pair of aces)": ("Js Ts", "Ah Ac"),
    }
    matchup = mo.ui.dropdown(options=list(MATCHUPS), value=list(MATCHUPS)[0], label="Matchup")
    matchup
    return MATCHUPS, matchup


@app.cell
def _(MATCHUPS, T, matchup, mo, poker):
    _h, _v = MATCHUPS[matchup.value]
    HERO, VIL = poker.cards(_h), poker.cards(_v)
    D = T.enumerate_flops(HERO, VIL)
    CAT = T.categories(D["flops"], HERO, VIL)
    LABELS = T.category_labels(HERO, VIL)
    _pre = D["eq"].mean()
    mo.md(f"""
    You hold **{poker.names(HERO, True)}** and the opponent holds **{poker.names(VIL, True)}**. Before the flop you win **{_pre:.1%}**
    outright, ties {D['ties'].mean() / T.N_RUNOUTS:.1%}. {len(D['flops']):,} flops, exact equity of each from 990
    runouts ({'cached' if D['cached'] else f"enumerated in {D['seconds']:.1f} s"}). Only
    **{len(set(D['wins'].tolist()))}** distinct equity values occur.
    """)
    return CAT, D, HERO, LABELS, VIL


@app.cell
def _(CAT, COLORS, D, LABELS, T, np, plt):
    _fig, _ax = plt.subplots(figsize=(7.5, 2.8))
    _bins = np.linspace(0, 1, 51)
    _ax.hist([D["eq"][CAT == c] for c in T.CATEGORY_ORDER], bins=_bins, stacked=True,
             color=[COLORS[c] for c in T.CATEGORY_ORDER], label=[f"{LABELS[c]} ({(CAT == c).sum():,})" for c in T.CATEGORY_ORDER])
    _ax.axvspan(1 / 6, 0.4, color="#eb6834", alpha=0.08, lw=0)
    _ax.text(0.283, _ax.get_ylim()[1] * 0.92, "calling band", ha="center", color="#b0502a", fontsize=8)
    _ax.set_xlabel("exact flop equity (P you win outright)"); _ax.set_ylabel("flops")
    _ax.legend(frameon=False, fontsize=8); _ax.set_title("Every flop's exact equity", loc="left", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    sample_seed = mo.ui.slider(0, 99, value=0, label="sample seed", show_value=True)
    sample_seed
    return (sample_seed,)


@app.cell
def _(CAT, D, HERO, LABELS, T, VIL, mo, np, poker, sample_seed):
    _rng = np.random.default_rng(sample_seed.value)
    _rows = []
    for _i in _rng.choice(len(D["flops"]), 8, replace=False):
        _f = list(D["flops"][_i])
        _dead = set(HERO + VIL + _f)
        _deck = [c for c in range(52) if c not in _dead]
        _res = []
        for _ in range(12):  # sampled showdowns: turn + river
            _ro = list(_rng.choice(_deck, 2, replace=False))
            _a = poker.eval7(np.array([HERO + _f + _ro]))[0]; _b = poker.eval7(np.array([VIL + _f + _ro]))[0]
            _res.append("W" if _a > _b else "T" if _a == _b else "·")
        _rows.append({"flop": poker.names(_f, True), "what happened": LABELS[CAT[_i]], "exact equity": f"{D['eq'][_i]:.3f}",
                      "12 sampled showdowns": "".join(_res), "sampled mean": f"{_res.count('W') / 12:.2f}"})
    mo.vstack([mo.md("**Random flops.** One sampled showdown is the 0/1 label RLCR's reward sees; the exact "
                     "equity is what it is noisy about (W = you win, T = split, · = you lose)."),
               mo.ui.table(_rows, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit a small model

    Input: the three flop cards (one-hot over 52; optionally plus rank and suit counts, which is what a player
    looks at). Output: sigmoid(·) as P(win), trained full-batch with Adam on squared error to the target, which is
    Brier against a 0/1 label and a proper score either way. Train / test split by flop.
    """)
    return


@app.cell
def _(mo):
    hidden = mo.ui.dropdown(options={"linear": 0, "MLP 8": 8, "MLP 32": 32, "MLP 128": 128, "MLP 512": 512},
                            value="MLP 128", label="model")
    feats = mo.ui.dropdown(options=["onehot", "onehot+counts"], value="onehot+counts", label="features")
    kdial = mo.ui.dropdown(options={str(k): k for k in [1, 2, 4, 8, 16, 64, 256, 990]}, value="1",
                           label="k runouts per target (990 = exact)")
    ntrain = mo.ui.slider(500, 14000, step=500, value=12000, label="train flops", show_value=True)
    epochs = mo.ui.slider(50, 1000, step=50, value=300, label="epochs", show_value=True)
    fit_seed = mo.ui.slider(0, 9, value=0, label="seed", show_value=True)
    mo.hstack([mo.vstack([hidden, feats, kdial]), mo.vstack([ntrain, epochs, fit_seed])], justify="start", gap=2)
    return epochs, feats, fit_seed, hidden, kdial, ntrain


@app.cell
def _(CALLER, D, T, epochs, feats, fit_seed, hidden, kdial, mo, np, ntrain, time):
    _rng = np.random.default_rng(1000 + fit_seed.value)
    X = T.features(D["flops"], feats.value)
    PERM = np.random.default_rng(0).permutation(len(X))  # fixed test set: the last 3,296 flops
    TE = PERM[14000:]; TR = PERM[: ntrain.value]
    _y = T.sampled_targets(D["wins"][TR], kdial.value, _rng)
    _t = time.time()
    PREDICT = T.fit(X[TR], _y, hidden=hidden.value, epochs=epochs.value, seed=fit_seed.value)
    Q = PREDICT(X[TE]); P = D["eq"][TE]
    S = T.scores(Q, P)
    _const = T.scores(np.full(len(P), D["eq"][TR].mean()), P)
    _caller = float((CALLER.H(P) + CALLER.D(P, Q)).mean()); _caller_o = float(CALLER.H(P).mean())
    mo.md(f"""
    Trained in **{time.time() - _t:.2f} s** on {len(TR):,} flops, targets = mean of k = {kdial.value} runouts
    (target noise sd ≈ {np.sqrt((D['eq'][TR] * (1 - D['eq'][TR]) / kdial.value * (990 - kdial.value) / 989).mean()):.3f}).
    Test, {len(TE):,} flops:

    | | MSE vs exact | correlation | REL (miscalibration) | RES (discrimination) | Brier vs oracle | caller loss |
    |---|---|---|---|---|---|---|
    | **model** | **{S['mse']:.4f}** | **{S['corr']:.3f}** | {S['REL']:.4f} | {S['RES']:.4f} | {S['brier_oracle']:.4f} | {_caller:.4f} |
    | constant (train mean) | {_const['mse']:.4f} | – | {_const['REL']:.4f} | 0 | {_const['brier_oracle']:.4f} | {float((CALLER.H(P) + CALLER.D(P, np.full(len(P), D['eq'][TR].mean()))).mean()):.4f} |
    | oracle (exact equity) | 0 | 1 | 0 | {S['UNC'] - S['floor']:.4f} | {S['floor']:.4f} | {_caller_o:.4f} |

    Brier vs oracle = MSE + {S['floor']:.4f} (the luck no model removes). Caller loss: final pots lost per call
    decision, bets ¼–2 × pot (see the [pot-odds page](https://rrenaud.github.io/rlcr-exact/poker/)).
    """)
    return P, PERM, PREDICT, Q, TE, X


@app.cell
def _(CAT, COLORS, LABELS, P, Q, T, TE, np, plt):
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    _c = CAT[TE]
    for _k in T.CATEGORY_ORDER:
        _m = _c == _k
        _a.scatter(P[_m], Q[_m], s=4, alpha=0.35, color=COLORS[_k], label=LABELS[_k], rasterized=True)
    _a.plot([0, 1], [0, 1], color="#7a838c", lw=1)
    _a.set_xlabel("exact equity"); _a.set_ylabel("model"); _a.set_title("Test flops", loc="left", fontsize=10)
    _a.legend(frameon=False, fontsize=7, markerscale=3)
    _bins = np.minimum((Q * 20).astype(int), 19)
    _xs, _ys, _ns = [], [], []
    for _i in np.unique(_bins):
        _m = _bins == _i; _xs.append(Q[_m].mean()); _ys.append(P[_m].mean()); _ns.append(_m.sum())
    _b.axvspan(1 / 6, 0.4, color="#eb6834", alpha=0.08, lw=0)
    _b.plot([0, 1], [0, 1], color="#7a838c", lw=1)
    _b.scatter(_xs, _ys, s=np.array(_ns) / max(_ns) * 200 + 6, color="#2a78d6", alpha=0.8)
    _b.set_xlabel("model (20 bins)"); _b.set_ylabel("mean exact equity in bin")
    _b.set_title("Reliability against the truth", loc="left", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    sweep_btn = mo.ui.run_button(label="Run exact-vs-sampled sweep (~1 min)")
    mo.vstack([mo.md(r"""
    ## The exact-vs-sampled dial

    Test MSE against the exact equity as the targets go from one sampled showdown (*k* = 1) to exact (*k* = 990),
    for several model sizes, 2 seeds each, same features and epochs as above. Right: learning curves in the number
    of training flops for *k* = 1 vs exact, at the model size chosen above.
    """), sweep_btn])
    return (sweep_btn,)


@app.cell
def _(D, PERM, T, X, epochs, hidden, mo, np, plt, sweep_btn):
    mo.stop(not sweep_btn.value, mo.md("*Press the button to run the sweep.*"))
    _te = PERM[14000:]; _P = D["eq"][_te]
    _ks = [1, 4, 16, 990]; _hs = [0, 32, 128, 512]; _seeds = [0, 1]
    _res = {}
    for _h in _hs:
        for _k in _ks:
            _m = []
            for _s in _seeds:
                _tr = PERM[:12000]
                _y = T.sampled_targets(D["wins"][_tr], _k, np.random.default_rng(100 + _s))
                _m.append(T.scores(T.fit(X[_tr], _y, hidden=_h, epochs=epochs.value, seed=_s)(X[_te]), _P)["mse"])
            _res[(_h, _k)] = (np.mean(_m), np.std(_m))
    _ns = [500, 1000, 2000, 4000, 8000, 14000]; _lc = {}
    for _k in [1, 990]:
        for _n in _ns:
            _m = []
            for _s in _seeds:
                _tr = PERM[:_n]
                _y = T.sampled_targets(D["wins"][_tr], _k, np.random.default_rng(200 + _s))
                _m.append(T.scores(T.fit(X[_tr], _y, hidden=hidden.value, epochs=epochs.value, seed=_s)(X[_te]), _P)["mse"])
            _lc[(_k, _n)] = (np.mean(_m), np.std(_m))
    SWEEP = dict(res=_res, lc=_lc)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    _cols = {0: "#7a838c", 32: "#2a78d6", 128: "#eb6834", 512: "#8a5cc2"}
    for _h in _hs:
        _mu = [_res[(_h, _k)][0] for _k in _ks]; _sd = [_res[(_h, _k)][1] for _k in _ks]
        _a.errorbar(_ks, _mu, yerr=_sd, marker="o", ms=4, capsize=2, color=_cols[_h], label="linear" if not _h else f"MLP {_h}")
    _a.set_xscale("log"); _a.set_yscale("log"); _a.set_xlabel("k runouts per target (990 = exact)")
    _a.set_ylabel("test MSE vs exact equity"); _a.legend(frameon=False, fontsize=8)
    _a.set_title("Label noise × capacity", loc="left", fontsize=10)
    for _k, _c, _lab in [(1, "#eb6834", "k = 1 (one showdown)"), (990, "#2a78d6", "exact")]:
        _mu = [_lc[(_k, _n)][0] for _n in _ns]; _sd = [_lc[(_k, _n)][1] for _n in _ns]
        _b.errorbar(_ns, _mu, yerr=_sd, marker="o", ms=4, capsize=2, color=_c, label=_lab)
    _b.set_xscale("log"); _b.set_yscale("log"); _b.set_xlabel("training flops"); _b.set_ylabel("test MSE vs exact equity")
    _b.legend(frameon=False, fontsize=8); _b.set_title(f"Learning curves, {'linear' if not hidden.value else f'MLP {hidden.value}'}", loc="left", fontsize=10)
    _fig.tight_layout()
    _fig
    return (SWEEP,)


@app.cell(hide_code=True)
def _(MATCHUPS, mo):
    transfer = mo.ui.dropdown(options=list(MATCHUPS), value=list(MATCHUPS)[3], label="Evaluate the trained model on")
    mo.vstack([mo.md(r"""
    ## Did it learn poker, or a lookup table?

    Apply the model trained above, unchanged, to every flop of a different matchup. The inputs are only the board
    cards, so a model that learned "an ace or king pairs you" transfers to matchups where that is still true,
    and one that memorized this matchup does not. One-hot weights for the original matchup's four hole cards were
    never trained (those cards cannot appear on its boards), so boards containing them are out of distribution.
    """), transfer])
    return (transfer,)


@app.cell
def _(MATCHUPS, PREDICT, T, feats, mo, np, plt, poker, transfer):
    _h, _v = MATCHUPS[transfer.value]
    _H, _V = poker.cards(_h), poker.cards(_v)
    _D2 = T.enumerate_flops(_H, _V)
    _Q2 = PREDICT(T.features(_D2["flops"], feats.value))
    _s = T.scores(_Q2, _D2["eq"])
    _c2 = T.categories(_D2["flops"], _H, _V); _l2 = T.category_labels(_H, _V)
    _fig, _ax = plt.subplots(figsize=(4.6, 3.6))
    for _k, _col in [("villain sets", "#8a5cc2"), ("miss", "#7a838c"), ("miss + flush draw", "#2a78d6"), ("hero pairs", "#eb6834")]:
        _m = _c2 == _k
        _ax.scatter(_D2["eq"][_m], _Q2[_m], s=3, alpha=0.3, color=_col, label=_l2[_k], rasterized=True)
    _ax.plot([0, 1], [0, 1], color="#7a838c", lw=1); _ax.legend(frameon=False, fontsize=7, markerscale=3)
    _ax.set_xlabel("exact equity (new matchup)"); _ax.set_ylabel("model trained on the first matchup")
    _fig.tight_layout()
    mo.hstack([_fig, mo.md(f"""
    **{poker.names(_H, True)} vs {poker.names(_V, True)}**, all {len(_Q2):,} flops

    - MSE vs exact: **{_s['mse']:.4f}**
    - correlation: **{_s['corr']:.3f}**
    - REL {_s['REL']:.4f} · RES {_s['RES']:.4f}
    - constant at this matchup's preflop equity: MSE {np.var(_D2['eq']):.4f}
    """)], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to look for

    - **Linear vs MLP.** A linear model on one-hot cards gets the big modes (does an A or K or an 8 hit) but
      cannot represent interactions such as two-to-a-suit flush draws or which straight a board allows, so it
      plateaus at correlation ~0.88–0.92. A small MLP reaches ~0.99.
    - **Sampled targets reverse the ranking of model sizes.** With one showdown per flop, bigger is worse
      (A♠K♦ vs 8♥8♣, 12k flops, 300 epochs: MSE 0.0055 / 0.011 / 0.019 for MLP 32 / 128 / 512). With exact
      targets bigger is better (0.0032 / 0.0015 / 0.0009). The noise costs both calibration (REL) and
      discrimination (RES). That is the case for exact objectives in one plot: they matter once a model can
      memorize its labels.
    - **Exact targets are worth a lot of data.** For an MLP 128, exact targets on 500 flops (MSE 0.003) beat one
      showdown per flop on all 14,000 (0.009).
    - **Transfer is partial.** Trained on AK vs 88 and applied to AK vs 77, correlation is ~0.70: it learned
      "an ace or king helps you", and it memorized "eights are bad".
    - **Discrimination is the ceiling.** Brier against the oracle can never drop below the luck term (~0.15
      here). Everything a model does happens in the MSE above it.
    """)
    return


if __name__ == "__main__":
    app.run()
