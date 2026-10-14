#!/bin/bash
# Pix0.6 small-window sweep (windows 64 and 32, flex+wrap baseline vs OR, seeds 42-44): keeps MAX_JOBS training jobs
# on the GPU and starts the next run whenever one finishes. All four configs per seed, seed by seed.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
Q=sweep_runs
PIDS=$Q/running_small.pids
RUNS=(
  "flex-w64 42" "or-w64 42" "flex-w32 42" "or-w32 42"
  "flex-w64 43" "or-w64 43" "flex-w32 43" "or-w32 43"
  "flex-w64 44" "or-w64 44" "flex-w32 44" "or-w32 44"
)
: > $PIDS
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue_small.log; }
alive() { local n=0 keep=""; for p in $(cat $PIDS); do
            if kill -0 $p 2>/dev/null; then n=$((n+1)); keep="$keep $p"; fi; done
          echo $keep | tr ' ' '\n' | grep . > $PIDS; echo $n; }
log "queue started, ${#RUNS[@]} runs, max $MAX_JOBS jobs"
for r in "${RUNS[@]}"; do
  set -- $r; cfg=$1; seed=$2
  name=TRK-Pix0.6-DQMA-$(echo $cfg | sed 's/^or/OR/')-s$seed
  while [ "$(alive)" -ge $MAX_JOBS ]; do sleep 60; done
  nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > logs/${name}_full.out 2>&1 &
  pid=$!; echo $pid >> $PIDS
  log "START $name pid=$pid"
  sleep 120
done
while [ "$(alive)" -gt 0 ]; do sleep 300; done
log "all runs finished"
