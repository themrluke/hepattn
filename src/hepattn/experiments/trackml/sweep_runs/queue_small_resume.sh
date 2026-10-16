#!/bin/bash
# Continues the small-window sweep after the notebook container restarted (16 Oct ~09:18, killing every process):
# resumes the two interrupted window-32 seed-42 runs from their last checkpoints (appending to their logs, so the
# evaluation loop sees them finish), then runs the remaining seed-43 and seed-44 runs, two jobs at a time.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
Q=sweep_runs
PIDS=$Q/running_small.pids
RUNS=(
  "resume flex-w32-s42" "resume OR-w32-s42"
  "flex-w64 43" "or-w64 43" "flex-w32 43" "or-w32 43"
  "flex-w64 44" "or-w64 44" "flex-w32 44" "or-w32 44"
)
: > $PIDS
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue_small.log; }
alive() { local n=0 keep=""; for p in $(cat $PIDS); do
            if kill -0 $p 2>/dev/null; then n=$((n+1)); keep="$keep $p"; fi; done
          echo $keep | tr ' ' '\n' | grep . > $PIDS; echo $n; }
log "queue resumed after the container restart, ${#RUNS[@]} runs, max $MAX_JOBS jobs"
for r in "${RUNS[@]}"; do
  set -- $r
  while [ "$(alive)" -ge $MAX_JOBS ]; do sleep 60; done
  if [ "$1" = resume ]; then
    name=TRK-Pix0.6-DQMA-$2
    dir=$(ls -d logs/${name}_2026* | head -1)
    ckpt=$(ls $dir/ckpts/epoch=*.ckpt | sort | tail -1)
    nohup $P run_tracking.py fit --config $dir/config.yaml --ckpt_path $ckpt >> logs/${name}_full.out 2>&1 &
    pid=$!; echo $pid >> $PIDS
    log "RESUME $name from $ckpt pid=$pid"
  else
    cfg=$1; seed=$2
    name=TRK-Pix0.6-DQMA-$(echo $cfg | sed 's/^or/OR/')-s$seed
    nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > logs/${name}_full.out 2>&1 &
    pid=$!; echo $pid >> $PIDS
    log "START $name pid=$pid"
  fi
  sleep 120
done
while [ "$(alive)" -gt 0 ]; do sleep 300; done
log "all runs finished"
