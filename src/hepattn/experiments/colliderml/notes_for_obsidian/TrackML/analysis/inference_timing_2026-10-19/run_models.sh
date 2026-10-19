#!/bin/bash
# Rerun only the trained-model part of the study (its timing hooks failed on the compiled modules the first time),
# three repeats one after the other, merged into results_repeat{1,2,3}.json
cd "$(dirname "$0")"
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
for r in 1 2 3; do
  echo "$(date '+%F %H:%M') models repeat $r start" >> status
  $P inference_timing.py $r results_repeat$r.json models > repeat${r}_models.log 2>&1
  echo "$(date '+%F %H:%M') models repeat $r exit $?" >> status
done
echo "$(date '+%F %H:%M') MODELSDONE" >> status
