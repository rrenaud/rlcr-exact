"""Live view of the frozen-confidence training runs: the gpuq queue, a per-run summary, and dev/train curves.

Reads every train_*.jsonl under /data/gpuq-runs/rrenaud/rlcr-*/run-*/ (train_frozen.py's log) as it grows.
usage: ~/envs/dash/bin/streamlit run scripts/dash.py --server.address 127.0.0.1 --server.port 8501
(from a laptop: ssh -L 8501:127.0.0.1:8501 HOST, then open http://localhost:8501)
"""
import glob, json, os, subprocess

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = "/data/gpuq-runs/rrenaud"
DEV_METRICS = ["greedy_brier", "greedy_auroc", "greedy_ece", "expected_brier", "greedy_mean", "expected_mean", "greedy_sd", "mass",
               "level_min_median", "answer_kl", "greedy_brier_f1", "greedy_auroc_f1", "expected_auroc"]
TRAIN_METRICS = ["expected_reward", "answer_kl", "support_floor", "adam_fraction", "mass", "lr"]
# support_floor: cross-entropy (nats) from a uniform target over the 11 levels 0.0..1.0 to each row's level
# marginal; ln 11 = 2.40 when a row spreads evenly, 6-10 when peaked as RLVR starts. Weighted by beta in the loss.

st.set_page_config(page_title="RLCR exact runs", layout="wide")


@st.cache_data(show_spinner=False)
def load(path, mtime):  # mtime keys the cache, so a growing log is re-read
    rows = [json.loads(l) for l in open(path) if l.strip()]
    args = next((r["args"] for r in rows if r["event"] == "data"), {})
    return (args, [r for r in rows if r["event"] == "eval"], [r for r in rows if r["event"] == "train"],
            [r for r in rows if r["event"] == "grad"])


def config(args):  # the settings that tell runs apart, as short keys (table columns and plot names)
    c = dict(obj=args.get("objective", "?"), lr=f"{args.get('lr', 0):g}")
    for key, tag in (("support_floor", "F"), ("answer_kl", "W"), ("train_problems", "overfit")):
        c[tag] = f"{args[key]:g}" if args.get(key) else "-"
    c["wsd"] = f"{args.get('warmup_steps', 0)}/{args.get('decay_frac', 0):g}" if args.get("warmup_steps") or args.get("decay_frac") else "-"
    c["seed"] = str(args.get("seed", 0))
    return c


def label(args, path):  # full name, for the text filter
    return " ".join(f"{k}={v}" for k, v in config(args).items() if v != "-")


def name_runs(sel):
    """Plot name per run: only the config keys that vary among the selected runs; '#2' etc. for exact repeats."""
    keys = [k for k in config({}) if len({r["config"][k] for r in sel}) > 1] or ["obj"]
    seen = {}
    for r in sel:
        base = " ".join(f"{k}={r['config'][k]}" for k in keys)
        seen[base] = seen.get(base, 0) + 1
        r["name"] = base if seen[base] == 1 else f"{base} #{seen[base]}"
    return keys


def runs():
    out = []
    for path in sorted(glob.glob(f"{ROOT}/rlcr-*/run-*/train_*.jsonl")):
        args, evals, train, grad = load(path, os.path.getmtime(path))
        out.append(dict(path=path, experiment=path.split("/")[-3], args=args, evals=evals, train=train, grad=grad,
                        label=label(args, path), config=config(args), started=os.path.basename(os.path.dirname(path))[4:19]))
    return out


def queue():
    try:
        return subprocess.run(["gpuq", "status"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:  # gpuq missing or hung: show the error instead of failing the page
        return f"gpuq status failed: {e}"


@st.fragment(run_every=60)
def body(chosen, show_small, text, hide, split, smooth):
    all_runs = runs()
    sel = [r for r in all_runs if r["experiment"] in chosen and text in r["label"]
           and not hide & set(r["label"].split())
           and (show_small or (r["args"].get("steps", 0) >= 100 and not r["args"].get("train_problems")))]

    keys = name_runs(sel)

    st.subheader("Queue")
    st.code(queue() or "(empty)")

    st.subheader(f"Runs ({len(sel)})")
    table = []
    for r in sel:
        dev = [e for e in r["evals"] if e["split"] == "dev"]
        last_train = r["train"][-1] if r["train"] else {}
        steps = r["args"].get("steps", 0)
        done = last_train.get("step", 0)
        rate = last_train.get("seconds", 0) / done if done else float("nan")
        outcome = os.path.join(os.path.dirname(r["path"]), "outcome.json")
        if os.path.exists(outcome):  # written by gpuq when the run leaves the queue
            o = json.load(open(outcome))
            state = "cancelled" if o.get("cancelled") else "done" if o.get("command_exit_code") == 0 else f"exit {o.get('command_exit_code')}"
        else:
            state = "running"
        row = dict(name=r["name"], **r["config"], experiment=r["experiment"], started=r["started"], state=state, step=f"{done}/{steps}",
                   eta_min=round(rate * (steps - done) / 60) if state == "running" and done else None)
        if dev:
            for m in ("greedy_brier", "greedy_auroc", "greedy_ece", "greedy_mean", "answer_kl"):
                if m in dev[0]:
                    row[f"{m} start"] = dev[0][m]
                    row[f"{m} now"] = dev[-1][m]
        table.append(row)
    if table:
        st.caption(f"Plots name each run by the settings that differ among the selected runs: {', '.join(keys)}. "
                   "F = support floor beta, W = answer-KL weight, wsd = warmup steps/decay fraction.")
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

    evals = pd.DataFrame([dict(run=r["name"], **e) for r in sel for e in r["evals"] if e["split"] == split])
    if evals.empty:
        st.subheader(f"Eval curves ({split})")
        st.info(f"No {split}-split evals in the selected runs. Train-split evals exist only for overfit runs "
                "(tick \"Include smoke/overfit runs\"); test is evaluated once, at the end of a run. "
                "The training log itself is plotted below.")
    else:
        st.subheader(f"Eval curves ({split})")
        metrics = st.multiselect("Eval metrics", [m for m in DEV_METRICS if m in evals.columns],
                                 default=[m for m in DEV_METRICS[:6] if m in evals.columns], key="evm")
        st.markdown("**Mean confidence, overall and by correctness** (solid: all rows; dashed: correct; dotted: wrong; "
                    "derived from mean, sd, Brier and accuracy when a log does not record them)")
        conf = st.radio("Readout", ["greedy", "expected"], horizontal=True, key="conf")
        # Older logs have no per-correctness means; recover them from the aggregates (exact: y is 0/1, so
        # Brier = E[c^2] - 2 a m1 + a with E[c^2] = sd^2 + mean^2; greedy stats skip off-list rows, off by greedy_off_list).
        if all(c in evals.columns for c in (f"{conf}_mean", f"{conf}_sd", f"{conf}_brier", "accuracy")):
            a, mu, sd, b = (evals[c] for c in ("accuracy", f"{conf}_mean", f"{conf}_sd", f"{conf}_brier"))
            m1 = (sd ** 2 + mu ** 2 + a - b) / (2 * a)
            m0 = (mu - a * m1) / (1 - a)
            for col, derived in ((f"{conf}_mean_correct", m1), (f"{conf}_mean_wrong", m0)):
                evals[col] = evals[col].fillna(derived) if col in evals.columns else derived
        parts = [(f"{conf}_mean", "all"), (f"{conf}_mean_correct", "correct"), (f"{conf}_mean_wrong", "wrong")]
        long = pd.concat([evals[["run", "step", c]].rename(columns={c: "confidence"}).assign(rows=name)
                          for c, name in parts if c in evals.columns], ignore_index=True).dropna()
        if not long.empty:
            acc = evals.groupby("run")["accuracy"].first() if "accuracy" in evals.columns else None
            fig = px.line(long, x="step", y="confidence", color="run", line_dash="rows", markers=True, height=420,
                          line_dash_map={"all": "solid", "correct": "dash", "wrong": "dot"})
            if acc is not None and acc.nunique() == 1:  # one shared accuracy: draw it as the calibration target
                fig.add_hline(y=float(acc.iloc[0]), line_color="gray", annotation_text=f"accuracy {acc.iloc[0]:.3f}")
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=9)), yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True, key="conf-by-y")
        cols = st.columns(2)
        for i, m in enumerate(metrics):
            fig = px.line(evals, x="step", y=m, color="run", markers=True, height=330)
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=9)), title=m)
            cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"ev-{m}")

    train = pd.DataFrame([dict(run=r["name"], **t) for r in sel for t in r["train"]])
    if not train.empty:
        st.subheader("Training curves (rolling mean)")
        tm = st.multiselect("Train metrics", [m for m in TRAIN_METRICS if m in train.columns],
                            default=[m for m in TRAIN_METRICS[:4] if m in train.columns], key="trm")
        cols = st.columns(2)
        for i, m in enumerate(tm):
            df = train[["run", "step", m]].dropna().sort_values("step")
            df[m] = df.groupby("run")[m].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
            fig = px.line(df, x="step", y=m, color="run", height=330)
            if m == "adam_fraction":
                fig.add_hline(y=0.14, line_color="gray", annotation_text="noise floor (~0.14)")
            if m == "support_floor":
                fig.add_hline(y=2.398, line_color="gray", annotation_text="ln 11 (uniform over levels)")
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=9)), title=m)
            cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"tr-{m}")

    grad = pd.DataFrame([dict(run=r["name"], **g) for r in sel for g in r["grad"]])
    if not grad.empty:
        st.subheader("Gradient decomposition (separate backward per loss term)")
        st.caption("norm_*: gradient norm of each weighted term of the step's mean loss (objective, floor = beta*CE, "
                   "kl = W*KL). cos_prev_*: cosine with that term's previous measurement (consistency over time). "
                   "cos_a_b: cosine between terms.")
        norm_cols = [c for c in grad.columns if c.startswith("norm_")]
        long = grad.melt(id_vars=["run", "step"], value_vars=norm_cols, var_name="term", value_name="norm").dropna()
        # Full page width each, legend underneath: runs x terms make long legends that squeeze a side legend's plot.
        below = dict(orientation="h", yanchor="top", y=-0.15, x=0, font=dict(size=10))
        fig = px.line(long, x="step", y="norm", color="run", line_dash="term", markers=True, height=520, log_y=True)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=below, title="gradient norm by term")
        cos_cols = [c for c in grad.columns if c.startswith("cos_")]
        long_c = grad.melt(id_vars=["run", "step"], value_vars=cos_cols, var_name="pair", value_name="cos").dropna()
        fig2 = px.line(long_c, x="step", y="cos", color="run", line_dash="pair", markers=True, height=520)
        fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=below, title="cosines", yaxis_range=[-1, 1])
        st.plotly_chart(fig, use_container_width=True, key="grad-norm")
        st.plotly_chart(fig2, use_container_width=True, key="grad-cos")


st.title("RLCR exact: frozen-confidence runs")
experiments = sorted({p.split("/")[-3] for p in glob.glob(f"{ROOT}/rlcr-*/run-*/train_*.jsonl")})
with st.sidebar:
    chosen = st.multiselect("Experiments", experiments, default=[e for e in experiments if "rlvr-frozen" in e] or experiments)
    show_small = st.checkbox("Include smoke/overfit runs", value=False)
    text = st.text_input("Label filter (substring)", "")
    hide = set(st.text_input("Hide runs with any of these settings (space-separated, e.g. W=1 obj=sampled)", "W=1").split())
    split = st.radio("Eval split (dev every 100 steps; test at the end)", ["dev", "train", "test"], horizontal=True)
    smooth = st.slider("Train-curve smoothing (log points)", 1, 20, 5)
    st.caption("Refreshes every 60 s.")
body(chosen, show_small, text, hide, split, smooth)
