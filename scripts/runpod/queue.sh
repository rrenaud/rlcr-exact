#!/bin/bash
# Launch queued RunPod jobs as capacity allows: each line of QUEUE is `NAME SCRIPT OBJECTIVE LR SEED [args]` (launch.sh's
# arguments). Keeps at most MAX (default 4) rlcr- pods running; a launch refused for capacity is retried on the next pass.
# Launched lines move to QUEUE.launched. Passes run every 2 minutes until QUEUE is empty; append lines while it runs.
# usage: [MAX=4] [CLOUD=secure] scripts/runpod/queue.sh QUEUE
q=$1; cd "$(dirname "$0")/../.."
while [ -s "$q" ]; do
  running=$(runpodctl get pod 2>/dev/null | grep -c " rlcr-")
  if [ "$running" -lt "${MAX:-4}" ]; then
    line=$(head -1 "$q")
    if out=$(scripts/runpod/launch.sh $line 2>&1) && grep -q "created" <<<"$out"; then
      echo "$(date '+%H:%M') launched: $line ($(grep -o 'pod "[a-z0-9]*"' <<<"$out"))"
      echo "$line" >> "$q.launched"; sed -i 1d "$q"; continue
    fi
    echo "$(date '+%H:%M') no capacity for: ${line%% *}"
  fi
  sleep 120  # a retry cadence for capacity, which RunPod gives no event for
done
echo "$(date '+%H:%M') queue empty"
