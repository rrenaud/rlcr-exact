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
LEADER = dict(scorer="rlcr11", lr=0.00037, answer_kl=0.02)  # any support floor: the default 0.053 is unlabeled in group names
METRICS = ["greedy_brier", "greedy_auroc", "greedy_ece", "expected_brier", "answer_kl", "greedy_mean"]

if st.query_params.get("page") == "online":  # the on-policy page (dash_online.py) on the same server
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dash_online.py"))
    st.stop()
st.set_page_config(page_title="Exact vs k=1, more data", layout="wide")
st.sidebar.markdown("[On-policy runs →](?page=online)")


@st.cache_data(show_spinner=False)
def load(path, mtime):  # mtime keys the cache, so a growing log is re-read
    rows = [json.loads(l) for l in open(path) if l.strip()]
    data = next((r for r in rows if r["event"] == "data"), {})
    return data, [r for r in rows if r["event"] == "eval"], [r for r in rows if r["event"] == "train"], \
        [r for r in rows if r["event"] == "grad"]


def runs():
    out = []
    for path in sorted(glob.glob(f"{ROOT}/run-*/train_*.jsonl")):
        data, evals, train, grad = load(path, os.path.getmtime(path))
        a = data.get("args", {})
        if any(a.get(k) != v for k, v in LEADER.items()) or a.get("samples_per_row", 1) != 1 or a.get("train_problems"):
            continue
        if a.get("warmup_steps") or a.get("decay_frac"):
            continue
        F, decay = a.get("support_floor") or 0, a.get("floor_decay_from")
        floor = "" if (F, decay) == (0.053, None) else f" F{F:g}" + (f"->0 from {decay:g}" if decay is not None else "")
        offset = 0
        if a.get("init_adapter"):  # a continuation: label it, and plot its steps after its parent's
            parent = glob.glob(os.path.join(os.path.dirname(os.path.dirname(a["init_adapter"].rstrip("/"))), "train_*.jsonl"))
            offset = load(parent[0], os.path.getmtime(parent[0]))[0].get("args", {}).get("steps", 0) if parent else 0
            floor += " cont"
            evals, train, grad = ([dict(e, step=e["step"] + offset) for e in xs] for xs in (evals, train, grad))
        n_q = data.get("train_problems") or 0
        obj = "exact" if a["objective"] == "exact" else "k=1"
        size = f"{round(n_q / 1000)}k" if n_q else "?"
        run_dir = os.path.dirname(path)
        outcome = os.path.join(run_dir, "outcome.json")
        if os.path.exists(outcome) and json.load(open(outcome)).get("cancelled"):  # a cancelled run's partial log
            continue
        g = os.path.join(run_dir, "gen_eval", "gen_eval.json")
        out.append(dict(obj=obj, data=size + floor, seed=a.get("seed", 0), train=train, grad=grad, offset=offset, steps=a["steps"], n_q=n_q,
                        ppstep=a.get("problems_per_step", 4), evals=evals, run=os.path.basename(run_dir),
                        gen=json.load(open(g)) if os.path.exists(g) else None,
                        done=os.path.exists(os.path.join(run_dir, "outcome.json"))))
    return out


st.title("Exact vs sampled k=1: 2k vs 4k training questions (hotpot, from RLVR)")
st.caption("Groups: objective, training questions, and the support floor when it is not the leader's constant 0.053. "
           "\"cont\": continued from another run's final adapter and Adam state; its steps are drawn after the parent's.")
sel = runs()
groups = sorted({f"{r['obj']} {r['data']}" for r in sel})
conts = [g for g in groups if g.endswith(" cont")]
with st.sidebar:
    # ?view=cont opens on the continuation runs only (their parent can be added back here)
    view = st.radio("Runs", ["all", "continuations only", "continuations + parent", "choose"],
                    index=1 if st.query_params.get("view") == "cont" else 0)
    if view == "choose":
        keep = st.multiselect("Groups", groups, default=groups)
    else:
        keep = groups if view == "all" else conts + (["exact 4k"] if view == "continuations + parent" else [])
    xaxis = st.radio("x axis", ["step", "passes over the training questions"])
    split_by_seed = st.checkbox("One line per seed (else 2k: mean over seeds with min-max band)", value=False)
sel = [r for r in sel if f"{r['obj']} {r['data']}" in keep]
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


# ---- Support floor ----------------------------------------------------------------------------------------------
st.header("Support floor: what it does over training, and when it could go")
st.markdown(
    "The floor is beta * CE(uniform over the 11 levels, the row's level marginal). Early it seeds mass on levels the "
    "exact gradient (proportional to q) cannot reach; late it holds a spread the objective would remove. Signs that it "
    "could decay: the smallest level mass has stopped rising, the floor's gradient has turned against the objective "
    "(cosine near -1) and is a large share of the update, and the expected readout lags the greedy one (the spread "
    "costs calibration). Panels marked (new) need the eval diagnostics added on 2026-10-20; older runs lack them.")
G = lambda r: f"{r['obj']} {r['data']}"
train = pd.DataFrame([dict(group=G(r), seed=r["seed"], **t) for r in sel for t in r["train"]])
grad = pd.DataFrame([dict(group=G(r), seed=r["seed"], **g) for r in sel for g in r["grad"]])
if not dev.empty:
    full = pd.DataFrame([dict(group=G(r), seed=r["seed"], **{k: v for k, v in e.items() if not isinstance(v, list)})
                         for r in sel for e in r["evals"] if e["split"] == "dev"])
    full["line"] = full["group"] + " s" + full["seed"].astype(str)
    full["expected_minus_greedy_brier"] = full["expected_brier"] - full["greedy_brier"]
    full["expected_minus_greedy_ece"] = full["expected_ece"] - full["greedy_ece"]


def lines(df, y, title, log_y=False, hline=None, key=None, col=None, dash=None):
    if df.empty or y not in df.columns or df[y].dropna().empty:
        (col or st).info(f"{title}: not logged by these runs.")
        return
    d = df.dropna(subset=[y]).assign(line=lambda x: x["group"] + " s" + x["seed"].astype(str))
    fig = px.line(d, x="step", y=y, color="group", line_group="line", line_dash=dash, markers=len(d) < 400,
                  height=340, log_y=log_y)
    if hline is not None:
        fig.add_hline(y=hline[0], line_color="gray", annotation_text=hline[1])
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=9)), title=title)
    (col or st).plotly_chart(fig, use_container_width=True, key=key or f"fl-{y}")


st.subheader("1. Gradients in LoRA space (grad events, every 50 steps; weighted terms)")
c = st.columns(2)
if not grad.empty:
    grad["floor_over_objective"] = grad["norm_floor"] / grad["norm_objective"]
    long = grad.melt(id_vars=["group", "seed", "step"], value_vars=["norm_objective", "norm_floor", "norm_kl"],
                     var_name="term", value_name="norm")
    long["line"] = long["group"] + " s" + long["seed"].astype(str) + " " + long["term"]
    fig = px.line(long, x="step", y="norm", color="group", line_dash="term", line_group="line", log_y=True, height=340)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(font=dict(size=9)),
                      title="gradient norm by term (objective, beta*floor, W*KL)")
    c[0].plotly_chart(fig, use_container_width=True, key="fl-norms")
lines(grad, "floor_over_objective", "|g_floor| / |g_objective| (floor's share of the pull)", log_y=True, hline=(1, "equal"), col=c[1])
c = st.columns(2)
lines(grad, "cos_floor_objective", "cos(g_floor, g_objective): < 0 = floor opposes the objective", hline=(0, ""), col=c[0])
lines(grad, "cos_prev_objective", "cos(g_objective, its previous measurement): consistency of the signal", hline=(0, ""), col=c[1])

st.subheader("2. Gradients in level-logit space (dev evals, every 100 steps) (new)")
st.caption("Per dev row, with p the row's distribution over the 11 levels as a softmax: objective ascent p_l (R_l - E_p R), "
           "floor ascent at beta = 1: 1/L - p_l. Medians over rows. Multiply the ratio by beta (panel 4) for the actual "
           "balance. Rows split by correctness. The floor always pulls toward uniform, away from wherever a row's mass sits, "
           "so it opposes the objective once the objective has picked a level.")
if not dev.empty:
    c = st.columns(2)
    long = full.melt(id_vars=["group", "seed", "step"], value_vars=[v for v in ("grad_ratio_level_correct", "grad_ratio_level_wrong") if v in full],
                     var_name="rows", value_name="ratio").dropna() if "grad_ratio_level_correct" in full else pd.DataFrame()
    lines(long.assign(rows=long["rows"].str.replace("grad_ratio_level_", "")) if not long.empty else long, "ratio",
          "|g_floor| / |g_objective| at beta = 1 (solid correct, dash wrong)", log_y=True, col=c[0], key="fl-ratio-lv", dash="rows")
    long = full.melt(id_vars=["group", "seed", "step"], value_vars=[v for v in ("grad_cos_level_correct", "grad_cos_level_wrong") if v in full],
                     var_name="rows", value_name="cos").dropna() if "grad_cos_level_correct" in full else pd.DataFrame()
    lines(long.assign(rows=long["rows"].str.replace("grad_cos_level_", "")) if not long.empty else long, "cos",
          "cos(floor, objective) in level-logit space", hline=(0, ""), col=c[1], key="fl-cos-lv", dash="rows")

st.subheader("3. How spread the stated number is")
c = st.columns(2)
if not dev.empty:
    long = full.melt(id_vars=["group", "seed", "step"], value_vars=[v for v in ("level_entropy", "level_entropy_correct", "level_entropy_wrong") if v in full],
                     var_name="rows", value_name="entropy").dropna() if "level_entropy" in full else pd.DataFrame()
    lines(long.assign(rows=long["rows"].str.replace("level_entropy_", "").str.replace("level_entropy", "all")) if not long.empty else long,
          "entropy", "entropy of the level distribution, nats (new; ln 11 = 2.40 uniform)", hline=(2.398, "uniform"),
          col=c[0], key="fl-ent", dash="rows")
lines(train.assign(**{"floor CE": train["support_floor"]}) if not train.empty and "support_floor" in train else train, "floor CE",
      "floor term CE(uniform, q) per train row (all runs; ln 11 = uniform, larger = more peaked)", hline=(2.398, "ln 11"), col=c[1])
c = st.columns(2)
if not dev.empty:
    lines(full, "level_min_median", "smallest level mass, median over dev rows (what the floor lifts)", log_y=True, col=c[0])
    lines(full, "greedy_sd", "sd of the greedy confidence across rows (resolution used)", col=c[1])

st.subheader("4. What the spread costs, and the beta schedule")
c = st.columns(2)
if not dev.empty:
    lines(full, "expected_minus_greedy_brier", "expected minus greedy readout, Brier (> 0: the spread costs)", hline=(0, ""), col=c[0])
    lines(full, "expected_minus_greedy_ece", "expected minus greedy readout, ECE", hline=(0, ""), col=c[1])
c = st.columns(2)
if not dev.empty:
    lines(full, "spread_cost", "spread cost: reward of the top level minus expected reward (new)", hline=(0, ""), col=c[0])
lines(train, "floor_beta", "beta at each step (constant unless --floor-decay-from)", col=c[1])

st.subheader("5. Mean level distribution over training, one run (new)")
with_p = [r for r in sel if any("level_p" in e for e in r["evals"])]
if not with_p:
    st.info("No selected run logs level_p yet (the floor ablation runs will).")
else:
    names = [f"{G(r)} s{r['seed']} ({r['run'][4:19]})" for r in with_p]
    r = with_p[names.index(st.selectbox("Run", names))]
    c = st.columns(3)
    for i, key in enumerate(("level_p", "level_p_correct", "level_p_wrong")):
        ev = [e for e in r["evals"] if e["split"] == "dev" and key in e]
        if not ev:
            continue
        z = [[e[key][l] for e in ev] for l in range(11)]
        fig = px.imshow(z, x=[e["step"] for e in ev], y=[f"{l / 10:.1f}" for l in range(11)], origin="lower",
                        aspect="auto", color_continuous_scale="Viridis", labels=dict(x="step", y="level", color="mean p"),
                        height=380, zmin=0, zmax=max(max(row) for row in z))
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), title=key.replace("level_p", "all rows").replace("all rows_", ""))
        c[i].plotly_chart(fig, use_container_width=True, key=f"heat-{key}")
