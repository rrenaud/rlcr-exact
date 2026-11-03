#!/bin/bash
# Submit pre-registered runs to gpuq at low priority: prereg_submit.sh CID[:ARM] ...  (no ARM = both arms).
# The description carries the commit the pre-registration was frozen at; refuse to submit from a dirty tree.
cd /home/rrenaud/rlcr-exact
git diff --quiet HEAD -- docs/prereg scripts src || { echo "commit the pre-registration first"; exit 1; }
rev=$(git rev-parse --short HEAD)
for item in "$@"; do
  cid=${item%%:*}; arms=${item#*:}; [ "$arms" = "$item" ] && arms="exact sampled"
  case $cid in A*) mins=150;; B*) mins=630;; esac
  for arm in $arms; do
    gpuq submit --experiment rlcr-prereg-${cid:0:1} --priority low --expected-minutes $mins --cwd /home/rrenaud/rlcr-exact \
      --description "prereg@$rev $cid $arm" -- /bin/bash scripts/prereg_run.sh $cid $arm 2>/dev/null
  done
done
