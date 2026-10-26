#!/bin/bash
# Start one Stage A run (TASK=hotpot-rlvr-11 data) on a RunPod community RTX 4090.
# usage: scripts/runpod/launch.sh NAME SCRIPT OBJECTIVE LR SEED [train_frozen.py args]
#   SCRIPT is run_frozen.sh (full run) or screen_frozen.sh (200 steps, one eval); generation checks are off.
#   e.g. scripts/runpod/launch.sh rps-screen-s0 screen_frozen.sh rps 3.7e-4 0 --answer-kl 0.02 --support-floor 0.053
# The working tree (tracked + untracked, not ignored) goes to the Modal volume as the pod's code, so uncommitted
# changes run as they are. The pod reads data and code from the volume with this machine's Modal token (passed as
# pod env vars, visible in the RunPod console), syncs its run to rlcr-exact-data:/runpod-runs/NAME every 10 minutes,
# and removes itself when done. Fetch with: modal volume get rlcr-exact-data /runpod-runs/NAME .
set -euo pipefail
name=$1 script=$2; shift 2
cd "$(dirname "$0")/../.."
V=rlcr-exact-data tgz=rlcr-exact-$name.tgz tmp=$(mktemp -d)
git ls-files -co --exclude-standard | tar czf $tmp/$tgz -T -
printf 'CODE_TGZ=%q\nSCRIPT=%q\nARGS=%q\n' "$tgz" "$script" "$*" > $tmp/job.env
modal volume put --force $V $tmp/$tgz /code/$tgz
modal volume put --force $V $tmp/job.env /code/jobs/$name.env
modal volume put --force $V scripts/runpod/pod_boot.sh /code/pod_boot.sh
rm -r $tmp
tok() { python -c "import tomllib; c = tomllib.load(open('$HOME/.modal.toml', 'rb')); print(next(p for p in c.values() if p.get('active'))['$1'])"; }
# RunPod restarts a container whose command exits, so the command ends in sleep (the pod removes itself when done).
# GPU / CLOUD (community|secure) / PORTS (e.g. 22/tcp; needs a host with a public IP) override the defaults.
runpodctl create pod --name "rlcr-$name" --gpuType "${GPU:-NVIDIA GeForce RTX 4090}" --${CLOUD:-community}Cloud \
  --imageName runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  --containerDiskSize 60 --mem 24 --vcpu 6 ${PORTS:+--ports $PORTS} \
  --env "RUN_NAME=$name" --env "MODAL_TOKEN_ID=$(tok token_id)" --env "MODAL_TOKEN_SECRET=$(tok token_secret)" \
  --args "bash -c '(/start.sh &); pip install -q modal && modal volume get --force $V /code/pod_boot.sh /root/pod_boot.sh && bash /root/pod_boot.sh; sleep infinity'"
