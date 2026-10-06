#!/bin/bash
# Pix0.6 window sweep queue: keeps MAX_JOBS training jobs on the GPU (counting A' and D's resumed runs),
# starting the next run whenever one finishes. Seed 42 for all four configs first, then seed 43.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
Q=sweep_runs
RUNS=(
  "flex-w256 42" "or-w256 42" "flex-w128 42" "or-w128 42"
  "flex-w256 43" "or-w256 43" "flex-w128 43" "or-w128 43"
)
echo 85954 > $Q/running.pids   # A' resumed
echo 85955 >> $Q/running.pids  # D resumed
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue.log; }
alive() { local n=0 keep=""; for p in $(cat $Q/running.pids); do
            if kill -0 $p 2>/dev/null; then n=$((n+1)); keep="$keep $p"; fi; done
          echo $keep | tr ' ' '\n' | grep . > $Q/running.pids; echo $n; }
log "queue started, ${#RUNS[@]} runs, max $MAX_JOBS jobs"
for r in "${RUNS[@]}"; do
  set -- $r; cfg=$1; seed=$2
  name=TRK-Pix0.6-DQMA-$(echo $cfg | sed 's/^or/OR/')-s$seed
  while [ "$(alive)" -ge $MAX_JOBS ]; do sleep 60; done
  nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > logs/${name}_full.out 2>&1 &
  pid=$!; echo $pid >> $Q/running.pids
  log "START $name pid=$pid"
  ( wait $pid 2>/dev/null; true ) &
  sleep 120
done
while [ "$(alive)" -gt 0 ]; do sleep 300; done
log "all runs finished"
