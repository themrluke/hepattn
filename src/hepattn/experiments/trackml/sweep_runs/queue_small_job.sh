#!/bin/bash
# Small-window sweep queue that can be restarted at any time (e.g. after the notebook session ends): for each planned
# run, skip it if finished, resume it (same Comet experiment) if it has checkpoints, otherwise start it fresh.
# Two jobs at a time. This copy is for running inside a Kubernetes Job (sweep_runs/k8s/small-sweep-job.yaml): it refuses
# to start while another queue is alive anywhere (heartbeat file on /shared, touched every minute).
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
Q=sweep_runs
PIDS=$Q/running_small.pids
RUNS=()
for seed in 42 43 44; do for cfg in flex-w64 or-w64 flex-w32 or-w32; do RUNS+=("$cfg $seed"); done; done
: > $PIDS
HB=$Q/queue_small.heartbeat
if [ -f $HB ] && [ $(( $(date +%s) - $(stat -c %Y $HB) )) -lt 600 ]; then
  echo "another queue is alive (heartbeat $(cat $HB)); not starting"; exit 1
fi
beat() { echo "$(hostname) $$ $(date '+%F %H:%M:%S')" > $HB; }
beat
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue_small.log; }
alive() { beat; local n=0 keep=""; for p in $(cat $PIDS); do
            if kill -0 $p 2>/dev/null; then n=$((n+1)); keep="$keep $p"; fi; done
          echo $keep | tr ' ' '\n' | grep . > $PIDS; echo $n; }
log "queue (re)started in $(hostname): ${#RUNS[@]} planned runs, max $MAX_JOBS jobs"
for r in "${RUNS[@]}"; do
  set -- $r; cfg=$1; seed=$2
  short=$(echo $cfg | sed 's/^or/OR/')-s$seed
  name=TRK-Pix0.6-DQMA-$short
  logf=logs/${name}_full.out
  if [ -f "$logf" ] && grep -q 'max_epochs=30. reached' "$logf"; then continue; fi
  while [ "$(alive)" -ge $MAX_JOBS ]; do sleep 60; done
  if ls logs/${name}_2026*/ckpts/epoch=*.ckpt >/dev/null 2>&1; then
    nohup $Q/resume_run.sh $short > /dev/null 2>&1 &
    pid=$!; log "RESUME $name pid=$pid"
  else
    nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > $logf 2>&1 &
    pid=$!; log "START $name pid=$pid"
  fi
  echo $pid >> $PIDS
  sleep 120
done
while [ "$(alive)" -gt 0 ]; do sleep 300; done
log "all runs finished"
rm -f $HB
