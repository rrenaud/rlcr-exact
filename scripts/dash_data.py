"""Exact vs sampled k=1 on 2k vs 4k hotpot training questions (rlcr11, lr 3.7e-4, W 0.02, F 0.053; the leader
config). Dev curves per run (x = optimizer steps or passes over the training questions), then the final numbers:
trainer readout on cached answers (dev averaged over the last 3 evals, test) and the generation check
(gen_eval/gen_eval.json), with the 2k seeds as points.

Caveat built into the comparison: the 4k runs take 1,000 steps (one pass) and the 2k runs 500 (one pass), so "more
data" is also "more steps"; at equal steps (x = step, <= 500) both have seen each question at most once.
usage: ~/envs/dash/bin/streamlit run scripts/dash_data.py --server.address 127.0.0.1 --server.port 8502
"""
import glob, json, os

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen"
LEADER = dict(scorer="rlcr11", lr=0.00037, answer_kl=0.02, support_floor=0.053)
METRICS = ["greedy_brier", "greedy_auroc", "greedy_ece", "expected_brier", "answer_kl", "greedy_mean"]

st.set_page_config(page_title="Exact vs k=1, more data", layout="wide")


@st.cache_data(show_spinner=False)
def load(path, mtime):  # mtime keys the cache, so a growing log is re-read
    rows = [json.loads(l) for l in open(path) if l.strip()]
    data = next((r for r in rows if r["event"] == "data"), {})
    return data, [r for r in rows if r["event"] == "eval"]


def runs():
    out = []
    for path in sorted(glob.glob(f"{ROOT}/run-*/train_*.jsonl")):
        data, evals = load(path, os.path.getmtime(path))
        a = data.get("args", {})
        if any(a.get(k) != v for k, v in LEADER.items()) or a.get("samples_per_row", 1) != 1 or a.get("train_problems"):
            continue
        if a.get("floor_decay_from") is not None or a.get("warmup_steps") or a.get("decay_frac"):
            continue
        n_q = data.get("train_problems") or 0
        obj = "exact" if a["objective"] == "exact" else "k=1"
        size = f"{round(n_q / 1000)}k" if n_q else "?"
        run_dir = os.path.dirname(path)
        outcome = os.path.join(run_dir, "outcome.json")
        if os.path.exists(outcome) and json.load(open(outcome)).get("cancelled"):  # a cancelled run's partial log
            continue
        g = os.path.join(run_dir, "gen_eval", "gen_eval.json")
        out.append(dict(obj=obj, data=size, seed=a.get("seed", 0), steps=a["steps"], n_q=n_q,
                        ppstep=a.get("problems_per_step", 4), evals=evals, run=os.path.basename(run_dir),
                        gen=json.load(open(g)) if os.path.exists(g) else None,
                        done=os.path.exists(os.path.join(run_dir, "outcome.json"))))
    return out


st.title("Exact vs sampled k=1: 2k vs 4k training questions (hotpot, from RLVR)")
sel = runs()
with st.sidebar:
    xaxis = st.radio("x axis", ["step", "passes over the training questions"])
    split_by_seed = st.checkbox("One line per seed (else 2k: mean over seeds with min-max band)", value=False)
st.caption("Leader config: rlcr11, lr 3.7e-4, W 0.02, F 0.053, 4 questions x 8 rollouts per step. "
           "4k runs: 1,000 steps (one pass); 2k runs: 500 steps (one pass). More data here is also more steps: "
           "compare at equal step for the data effect alone.")

rows = []
for r in sel:
    for e in r["evals"]:
        if e["split"] != "dev":
            continue
        rows.append(dict(obj=r["obj"], data=r["data"], seed=r["seed"], step=e["step"],
                         passes=e["step"] * r["ppstep"] / r["n_q"] if r["n_q"] else None, **{m: e.get(m) for m in METRICS}))
dev = pd.DataFrame(rows)
st.dataframe(pd.DataFrame([dict(objective=r["obj"], data=r["data"], seed=r["seed"], steps=r["steps"], run=r["run"],
                                state="done" if r["done"] else "running", gen_check="yes" if r["gen"] else "no") for r in sel]),
             hide_index=True, use_container_width=True)

if not dev.empty:
    x = "step" if xaxis == "step" else "passes"
    dev["group"] = dev["obj"] + " " + dev["data"]
    st.subheader("Dev curves (trainer readout on cached answers, 500 dev rows)")
    cols = st.columns(2)
    for i, m in enumerate(METRICS):
        if split_by_seed:
            d = dev.assign(line=dev["group"] + " s" + dev["seed"].astype(str))
            fig = px.line(d, x=x, y=m, color="group", line_group="line", markers=True, height=340)
        else:
            agg = dev.groupby(["group", x], as_index=False)[m].agg(["mean", "min", "max", "count"]).reset_index(drop=True)
            fig = px.line(agg, x=x, y="mean", color="group", markers=True, height=340,
                          error_y=agg["max"] - agg["mean"], error_y_minus=agg["mean"] - agg["min"],
                          hover_data=["count"])
            fig.update_yaxes(title=m)
        if x == "step":
            fig.add_vline(x=500, line_color="gray", line_dash="dot", annotation_text="2k runs end")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=10)), title=m)
        cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"dev-{m}")

final = []
for r in sel:
    dv = [e for e in r["evals"] if e["split"] == "dev"]
    te = [e for e in r["evals"] if e["split"] == "test"]
    base = dict(group=f"{r['obj']} {r['data']}", obj=r["obj"], data=r["data"], seed=r["seed"])
    if len(dv) >= 3 and te:
        final += [dict(base, measure="trainer dev (last 3)", brier=sum(e["greedy_brier"] for e in dv[-3:]) / 3,
                       auroc=sum(e["greedy_auroc"] for e in dv[-3:]) / 3, ece=dv[-1].get("greedy_ece")),
                  dict(base, measure="trainer test", brier=te[-1]["greedy_brier"], auroc=te[-1]["greedy_auroc"],
                       ece=te[-1].get("greedy_ece"))]
    for s, g in (r["gen"] or {}).items():
        final.append(dict(base, measure=f"generation {s}", brier=g["lora"]["brier"], auroc=g["lora"]["auroc"],
                          ece=g["lora"].get("ece"), mean_conf=g["lora"].get("mean_conf"),
                          em_delta=g.get("em_any_delta"), answer_changed=g.get("answer_changed")))
fin = pd.DataFrame(final)
if not fin.empty:
    st.subheader("Final numbers (points = seeds; 4k has one seed so far)")
    order = ["trainer dev (last 3)", "trainer test", "generation dev", "generation test"]
    cols = st.columns(3)
    for i, m in enumerate(["brier", "auroc", "ece"]):
        fig = px.strip(fin, x="measure", y=m, color="group", category_orders={"measure": order}, height=380,
                       hover_data=["seed"])
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=10)), title=m,
                          xaxis_title=None)
        cols[i].plotly_chart(fig, use_container_width=True, key=f"fin-{m}")
    st.markdown("**Reference, hotpot test (generation):** RLCR-hotpot 0.188 (paper eval file) / 0.196 (our "
                "reproduction).")
    st.dataframe(fin.sort_values(["measure", "group", "seed"]).round(3), hide_index=True, use_container_width=True)
