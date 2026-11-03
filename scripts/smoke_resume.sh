#!/bin/bash
# Preempt/resume smoke (gpuq payload). Stage A (cached, HF backend): an uninterrupted 4-step run vs one preempted
# after step 1 by $GPUQ_PREEMPT_FILE (must exit 75) and resumed in the same directory; the train records of steps
# 2-4 should match up to bf16 noise. Stage B (online, vLLM): preempted after step 1, resumed, finishes step 3.
D=${GPUQ_RUN_DIR:?run under gpuq}
A="--skip-initial-eval --dev-limit 50 --drift-rows 20 --eval-every 100 --no-log-q --resume-every 0"
phase() {  # phase DIR TASK [PY] -- args: run_frozen.sh in DIR with the preempt flag present / absent
  local dir=$1 task=$2 py=$3; shift 3
  mkdir -p $dir
  env GPUQ_RUN_DIR=$dir GPUQ_PREEMPT_FILE=$dir/preempt.json TASK=$task GEN_EVAL=0 ${py:+PY=$py} \
    bash scripts/run_frozen.sh "$@" > $dir/out.log 2>&1
  echo $?
}
SA="exact 0.00037 0 --answer-kl 0.02 --support-floor 0.053 --steps 4 $A"
echo "A uninterrupted: exit $(phase $D/a_u hotpot-rlvr-11 '' $SA)"
mkdir -p $D/a_p && echo '{}' > $D/a_p/preempt.json
echo "A preempted: exit $(phase $D/a_p hotpot-rlvr-11 '' $SA) (want 75)"
rm $D/a_p/preempt.json
echo "A resumed: exit $(phase $D/a_p hotpot-rlvr-11 '' $SA)"
SB="sampled 0.0001 0 --steps 3 $A --backend vllm --refresh-every 1 --answer-pg 1 --pg-balance rlcr --pg-reward sampled --answer-kl 0.02 --problems-per-step 2"
mkdir -p $D/b_p && echo '{}' > $D/b_p/preempt.json
echo "B preempted: exit $(phase $D/b_p hotpot-rlvr-11 /home/rrenaud/envs/vllm/bin/python $SB) (want 75)"
rm $D/b_p/preempt.json
echo "B resumed: exit $(phase $D/b_p hotpot-rlvr-11 /home/rrenaud/envs/vllm/bin/python $SB)"
/home/rrenaud/envs/pytorch/bin/python - $D <<'PY'
import glob, json, sys
D = sys.argv[1]
rec = lambda d: [json.loads(l) for f in glob.glob(f"{D}/{d}/train_*.jsonl") for l in open(f)]
for d in ("a_u", "a_p", "b_p"):
    r = rec(d)
    print(d, [(e["event"], e.get("step")) for e in r if e["event"] in ("train", "resumed", "preempted", "eval")])
u = {e["step"]: e for e in rec("a_u") if e["event"] == "train"}
p = {e["step"]: e for e in rec("a_p") if e["event"] == "train"}
for s in sorted(u):
    keys = [k for k in ("objective", "expected_reward", "answer_kl", "support_floor", "mass") if k in u[s]]
    print("step", s, {k: (u[s][k], p.get(s, {}).get(k)) for k in keys})
PY
