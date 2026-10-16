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
TRAIN_METRICS = ["expected_reward", "answer_kl", "support_floor", "mass", "lr"]
# support_floor: cross-entropy (nats) from a uniform target over the 11 levels 0.0..1.0 to each row's level
# marginal; ln 11 = 2.40 when a row spreads evenly, 6-10 when peaked as RLVR starts. Weighted by beta in the loss.

st.set_page_config(page_title="RLCR exact runs", layout="wide")


@st.cache_data(show_spinner=False)
def load(path, mtime):  # mtime keys the cache, so a growing log is re-read
    rows = [json.loads(l) for l in open(path) if l.strip()]
    args = next((r["args"] for r in rows if r["event"] == "data"), {})
    return args, [r for r in rows if r["event"] == "eval"], [r for r in rows if r["event"] == "train"]


def label(args, path):
    parts = [args.get("objective", "?"), f"lr{args.get('lr', 0):g}"]
    for key, tag in (("support_floor", "F"), ("answer_kl", "W"), ("train_problems", "overfit")):
        if args.get(key):
            parts.append(f"{tag}{args[key]:g}")
    if args.get("warmup_steps") or args.get("decay_frac"):
        parts.append(f"wsd{args.get('warmup_steps', 0)}-{args.get('decay_frac', 0):g}")
    parts.append(f"s{args.get('seed', 0)}")
    run = os.path.basename(os.path.dirname(path))
    return f"{' '.join(parts)} [{run[4:19]}]"


def runs():
    out = []
    for path in sorted(glob.glob(f"{ROOT}/rlcr-*/run-*/train_*.jsonl")):
        args, evals, train = load(path, os.path.getmtime(path))
        out.append(dict(path=path, experiment=path.split("/")[-3], args=args, evals=evals, train=train,
                        label=label(args, path)))
    return out


def queue():
    try:
        return subprocess.run(["gpuq", "status"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:  # gpuq missing or hung: show the error instead of failing the page
        return f"gpuq status failed: {e}"


@st.fragment(run_every=60)
def body(chosen, show_small, text, split, smooth):
    all_runs = runs()
    sel = [r for r in all_runs if r["experiment"] in chosen and text in r["label"]
           and (show_small or (r["args"].get("steps", 0) >= 100 and not r["args"].get("train_problems")))]

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
        row = dict(run=r["label"], experiment=r["experiment"], state=state, step=f"{done}/{steps}",
                   eta_min=round(rate * (steps - done) / 60) if state == "running" and done else None)
        if dev:
            for m in ("greedy_brier", "greedy_auroc", "greedy_ece", "greedy_mean", "answer_kl"):
                if m in dev[0]:
                    row[f"{m} start"] = dev[0][m]
                    row[f"{m} now"] = dev[-1][m]
        table.append(row)
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

    evals = pd.DataFrame([dict(run=r["label"], **e) for r in sel for e in r["evals"] if e["split"] == split])
    st.subheader(f"Eval curves ({split})")
    metrics = st.multiselect("Eval metrics", [m for m in DEV_METRICS if m in evals.columns],
                             default=[m for m in DEV_METRICS[:6] if m in evals.columns], key="evm")
    st.markdown("**Mean confidence, overall and by correctness** (solid: all rows; dashed: correct; dotted: wrong; "
                "per-correctness lines exist only for runs started after they were added to the log)")
    conf = st.radio("Readout", ["greedy", "expected"], horizontal=True, key="conf")
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

    train = pd.DataFrame([dict(run=r["label"], **t) for r in sel for t in r["train"]])
    if not train.empty:
        st.subheader("Training curves (rolling mean)")
        tm = st.multiselect("Train metrics", [m for m in TRAIN_METRICS if m in train.columns],
                            default=[m for m in TRAIN_METRICS[:3] if m in train.columns], key="trm")
        cols = st.columns(2)
        for i, m in enumerate(tm):
            df = train[["run", "step", m]].dropna().sort_values("step")
            df[m] = df.groupby("run")[m].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
            fig = px.line(df, x="step", y=m, color="run", height=330)
            if m == "support_floor":
                fig.add_hline(y=2.398, line_color="gray", annotation_text="ln 11 (uniform over levels)")
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=9)), title=m)
            cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"tr-{m}")


st.title("RLCR exact: frozen-confidence runs")
experiments = sorted({p.split("/")[-3] for p in glob.glob(f"{ROOT}/rlcr-*/run-*/train_*.jsonl")})
with st.sidebar:
    chosen = st.multiselect("Experiments", experiments, default=[e for e in experiments if "rlvr-frozen" in e] or experiments)
    show_small = st.checkbox("Include smoke/overfit runs", value=False)
    text = st.text_input("Label filter (substring)", "")
    split = st.radio("Eval split", ["dev", "train", "test"], horizontal=True)
    smooth = st.slider("Train-curve smoothing (log points)", 1, 20, 5)
    st.caption("Refreshes every 60 s.")
body(chosen, show_small, text, split, smooth)
