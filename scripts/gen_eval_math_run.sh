#!/bin/bash
# gpuq payload: gen_eval_math.sh for an already-finished math-rlvr pilot run.
# usage: gen_eval_math_run.sh PREP_DIR PROMPT RUN_DIR   (PREP_DIR = math_pilot_prep.sh run; base shared in PREP_DIR/PROMPT/gen_base)
prep=$1 prompt=$2 run=$3
bash /home/rrenaud/rlcr-exact/scripts/gen_eval_math.sh $prompt "$(ls -d $run/checkpoints/*/ | head -1)" $run $prep/$prompt/gen_base
