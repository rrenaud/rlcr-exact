#!/bin/bash
# Runs on a RunPod pod at boot (launch.sh puts it on the Modal volume and starts the pod with it). Rebuilds the gpuq
# box's layout, so run_frozen.sh and screen_frozen.sh work unchanged: the repo at /home/rrenaud/rlcr-exact, the data
# under /data/gpuq-runs/rrenaud (from the volume rlcr-exact-data, which mirrors the box), the Hub cache in
# /scratch/hf/hub. Then runs the job in /code/jobs/$RUN_NAME.env on the volume. The run directory (with boot.log)
# syncs to /runpod-runs/$RUN_NAME on the volume after setup, every 10 minutes, and at the end. Any exit, setup
# failures included, ends in a final sync and then removes the pod, so a failed pod does not sit there billing; only
# a failed final sync leaves the pod up, so nothing is lost.
# Env from launch.sh: RUN_NAME, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET. RunPod sets RUNPOD_POD_ID and RUNPOD_API_KEY.
set -uo pipefail
V=rlcr-exact-data G=/data/gpuq-runs/rrenaud R=/home/rrenaud/rlcr-exact OUT=/workspace/runs/$RUN_NAME
mkdir -p $OUT; exec > >(tee -a $OUT/boot.log) 2>&1
sync_run() { modal volume put --force $V $OUT /runpod-runs/$RUN_NAME > /dev/null; }
finish() {  # $1: exit status of the job, or a setup step's description
  echo "exit $1 $(date -Is)" | tee $OUT/exit_status
  [ -n "${SYNC:-}" ] && kill $SYNC
  if sync_run; then runpodctl remove pod $RUNPOD_POD_ID; else echo "final sync failed: pod left running"; fi
  exit 0
}
echo "boot $(date -Is) pod $RUNPOD_POD_ID"
pip install -q transformers peft scikit-learn scipy hf_transfer modal || finish "setup: pip"

get() { mkdir -p "$(dirname "$2")"; modal volume get --force $V "$1" "$2"; }
get /code/jobs/$RUN_NAME.env /root/job.env && . /root/job.env || finish "setup: job file"  # CODE_TGZ, SCRIPT, ARGS
mkdir -p $R && get /code/$CODE_TGZ /root/code.tgz && tar xzf /root/code.tgz -C $R || finish "setup: code"
mkdir -p $R/rlcr-reward-conf-weighted-tt-scaling
get /code/logprob_spans.py $R/rlcr-reward-conf-weighted-tt-scaling/logprob_spans.py || finish "setup: logprob_spans"
C=rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
for s in train dev test; do get /gpuq-runs/$C/rollouts_hotpot_fresh_$s.jsonl $G/$C/rollouts_hotpot_fresh_$s.jsonl || finish "setup: $s"; done
REF=rlcr-hotpot-rlvr-frozen/run-20261018T050522Z-eeac77918e/ref_topk_train
mkdir -p $G/$REF && for f in meta.json ids.int32 logp.float16; do get /gpuq-runs/$REF/$f $G/$REF/$f || finish "setup: ref $f"; done
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_ENABLE_HF_TRANSFER=1
python -c "from huggingface_hub import snapshot_download; snapshot_download('mehuldamani/hotpot-v2-correctness-7b')" > /dev/null || finish "setup: model"
echo "setup done $(date -Is)"; sync_run

( while true; do sleep 600; sync_run; done ) &  # a timer on purpose: a checkpoint cadence, not waiting for an event
SYNC=$!
cd $R
GPUQ_RUN_DIR=$OUT GEN_EVAL=0 TASK=hotpot-rlvr-11 PY=python bash scripts/$SCRIPT $ARGS --ref-topk $G/$REF
finish $?
