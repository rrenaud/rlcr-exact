"""On-policy runs (train_frozen.py --backend vllm, experiment rlcr-shared-vllm): fresh rollouts from the current
adapter every N steps (Stage A on-policy), and Stage B's answer-side term (--answer-pg).

Two sections: full-data runs (dev curves on the cached rollouts, and each refresh's fresh-rollout accuracy, format
and length), and B1 overfit runs (8 questions, every refresh's rollouts are the training questions themselves; plus
the per-term gradient norms). Refresh curves are smoothed with a rolling mean (sidebar).
Served by dash_data.py at ?page=online (port 8502), or alone. URL params for the initial selection:
  show=full|b1        one section only (default both)
  full=a,b / b1=a,b   preselect runs whose name contains any of the comma-separated substrings (default all),
                      e.g. full=N=1 s0,Stage B   b1=band
  smooth=N            rolling mean over refreshes (default 10)
usage: ~/envs/dash/bin/streamlit run scripts/dash_data.py --server.address 127.0.0.1 --server.port 8502
"""
import glob, json, os

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = "/data/gpuq-runs/rrenaud/rlcr-shared-vllm"
st.set_page_config(page_title="On-policy runs", layout="wide")


@st.cache_data(show_spinner=False)
def load(path, mtime):  # mtime keys the cache, so a growing log is re-read
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {ev: [r for r in rows if r.get("event") == ev] for ev in ("data", "eval", "train", "grad", "refresh")}


def label(a):
    if a.get("train_problems"):
        conf = "conf off" if a.get("conf_weight", 1) == 0 else "conf on" + (" +floor" if a.get("support_floor") else "")
        band = " band" if a.get("train_band") else ""
        return f"B1{band}: α={a.get('answer_pg', 0):g} {conf}"
    n = a.get("refresh_every") or "∞"
    pg = f" α={a['answer_pg']:g} {a.get('pg_reward')}" if a.get("answer_pg") else ""
    return f"{'Stage B' if pg else 'Stage A'} N={n}{pg} s{a.get('seed', 0)} {a['steps']}st"


def runs():
    out = []
    for path in sorted(glob.glob(f"{ROOT}/run-*/train_*.jsonl")):
        d = load(path, os.path.getmtime(path))
        if not d["data"]:
            continue
        a = d["data"][0]["args"]
        if a.get("backend") != "vllm" or a.get("steps", 0) < 50:
            continue
        run_dir = os.path.dirname(path)
        oc = os.path.join(run_dir, "outcome.json")
        if os.path.exists(oc) and json.load(open(oc)).get("cancelled"):
            continue
        run = os.path.basename(run_dir)
        out.append(dict(run=run, name=f"{label(a)} [{run[-6:]}]", overfit=bool(a.get("train_problems")),
                        done=os.path.exists(oc), **d))
    return out


R = runs()
Q = st.query_params
smooth = st.sidebar.slider("rolling mean over refreshes", 1, 50, int(Q.get("smooth", 10)))
st.sidebar.markdown("[← Exact vs k=1, more data](?)")
show = Q.get("show", "full,b1").split(",")


def preselect(names, key):
    subs = [x for x in Q.get(key, "").split(",") if x]
    return [n for n in names if any(x in n for x in subs)] if subs else names
st.title("On-policy runs (shared-weight vLLM)")
st.caption(f"{ROOT}; times in run ids are UTC. Running runs are marked *.")


def refresh_df(rs):
    rows = []
    for r in rs:
        df = pd.DataFrame(r["refresh"])
        if df.empty:
            continue
        df = df.sort_values("step")
        for m in ("em", "accuracy", "format_ok", "mean_tokens", "truncated"):
            if m in df:
                df[m] = df[m].rolling(smooth, min_periods=1).mean()
        if "mixed" in df:
            df["mixed_frac"] = (df["mixed"] / df["questions"]).rolling(smooth, min_periods=1).mean()
        df["run"] = r["name"] + ("" if r["done"] else " *")
        rows.append(df)
    return pd.concat(rows) if rows else pd.DataFrame()


def line(df, y, title, **kw):
    if df.empty or y not in df:
        return
    st.plotly_chart(px.line(df, x="step", y=y, color="run", title=title, **kw), use_container_width=True)


def grid(df, specs, cols=2):
    for i in range(0, len(specs), cols):
        cs = st.columns(cols)
        for c, (y, t) in zip(cs, specs[i: i + cols]):
            with c:
                line(df, y, t)


def full_section():
    st.header("Full data (2k RLVR train questions, 4 per step x 8 rollouts)")
    full = [r for r in R if not r["overfit"]]
    names = [r["name"] for r in full]
    pick = st.multiselect("runs", names, default=preselect(names, "full"), key="full")
    full = [r for r in full if r["name"] in pick]
    ev = pd.DataFrame([dict(e, run=r["name"] + ("" if r["done"] else " *")) for r in full for e in r["eval"] if e["split"] == "dev"])
    st.subheader("Dev, teacher-forced on cached rollouts (does not see answer changes)")
    grid(ev, [("greedy_brier", "dev greedy Brier"), ("greedy_auroc", "dev greedy AUROC"),
              ("expected_brier", "dev expected Brier"), ("answer_kl", "dev answer KL (nats / row)")])
    st.subheader("Fresh rollouts at each refresh (training questions, current adapter, T 0.7)")
    rf = refresh_df(full)
    grid(rf, [("em", "exact match (format ignored)"), ("accuracy", "RLCR accuracy (format-gated)"),
              ("format_ok", "format ok"), ("mean_tokens", "mean completion tokens"),
              ("mixed_frac", "groups with mixed correctness"), ("truncated", "truncated")])
    st.subheader("Training stats (every 10 steps)")
    tr = pd.DataFrame([dict(t, run=r["name"]) for r in full for t in r["train"]])
    grid(tr, [("expected_reward", "expected reward (train rows)"), ("answer_kl", "answer KL (train rows)"),
              ("pg_adv_abs", "mean |advantage| (Stage B)"), ("adam_fraction", "Adam fraction")])
    fin = []
    for r in full:
        row = dict(run=r["name"], done=r["done"])
        for e in r["eval"]:
            row[f"{e['split']}@{e['step']}"] = e.get("greedy_brier")
        if r["refresh"] and "em" in r["refresh"][0]:
            row["rollout em (all)"] = round(pd.DataFrame(r["refresh"])["em"].mean(), 4)
        fin.append(row)
    st.subheader("Greedy Brier at each eval, and mean fresh-rollout EM")
    st.dataframe(pd.DataFrame(fin).set_index("run"), use_container_width=True)


def b1_section():
    st.header("B1: overfit 8 questions (all 8 every step, accuracy reward, 60 steps)")
    b1 = [r for r in R if r["overfit"]]
    names = [r["name"] for r in b1]
    pick = st.multiselect("runs", names, default=preselect(names, "b1"), key="b1")
    b1 = [r for r in b1 if r["name"] in pick]
    rb = refresh_df(b1)
    grid(rb, [("em", "exact match on the 8 (format ignored)"), ("accuracy", "RLCR accuracy (format-gated)"),
              ("format_ok", "format ok"), ("mean_tokens", "mean completion tokens")])
    tb = pd.DataFrame([dict(t, run=r["name"]) for r in b1 for t in r["train"]])
    grid(tb, [("answer_kl", "answer KL (train rows)"), ("pg_adv_abs", "mean |advantage|")])
    gr = pd.DataFrame([dict(g, run=r["name"]) for r in b1 for g in r["grad"]])
    if not gr.empty:
        st.subheader("Gradient norms per term (every 5 steps)")
        norms = [c for c in gr if c.startswith("norm_")]
        long = gr.melt(id_vars=["run", "step"], value_vars=norms, var_name="term", value_name="norm").dropna()
        st.plotly_chart(px.line(long, x="step", y="norm", color="term", facet_col="run", facet_col_wrap=3, log_y=True,
                                height=300 * ((len(b1) + 2) // 3)), use_container_width=True)
    summ = []
    for r in b1:
        df = pd.DataFrame(r["refresh"])
        if df.empty or "em" not in df:
            continue
        summ.append(dict(run=r["name"], em_first10=round(df.em[:10].mean(), 3), em_last10=round(df.em[-10:].mean(), 3),
                         em_mean=round(df.em.mean(), 3), fmt_last10=round(df.format_ok[-10:].mean(), 3)))
    if summ:
        st.dataframe(pd.DataFrame(summ).set_index("run"), use_container_width=True)


if "full" in show:
    full_section()
if "b1" in show:
    b1_section()
