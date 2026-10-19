#!/bin/bash
# Three repeats of the inference timing study, one process each, run one after the other (spread in time)
cd "$(dirname "$0")"
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
for r in 1 2 3; do
  echo "$(date '+%F %H:%M') repeat $r start" >> status
  $P inference_timing.py $r results_repeat$r.json > repeat$r.log 2>&1
  echo "$(date '+%F %H:%M') repeat $r exit $?" >> status
done
echo "$(date '+%F %H:%M') ALLDONE" >> status
