#!/bin/bash
# gpuq payload gate: after_ok.sh GATE_RUN_DIR CMD ...  runs CMD only if the gate run (a smoke) finished cleanly: its
# stdout has SMOKE_DONE and no "exit N" with N != 0. Otherwise exits 3 without touching the GPU.
g=$1; shift
log=$g/logs/stdout.log
grep -q SMOKE_DONE "$log" 2>/dev/null || { echo "gate: $g did not finish"; exit 3; }
grep -E "exit [1-9]" "$log" && { echo "gate: $g had a failing stage"; exit 3; }
exec "$@"
