#!/bin/bash
# Learning-rate plateau study (Kubernetes Job sweep_runs/k8s/lr-plateau-job.yaml): window 64, flex and OR, seeds 42-44,
# peak learning rate 2e-4 then 1e-4 (default 5e-5; same seeds as the small-window sweep, so every run has a control).
# Stage 1: each run is stopped once its epoch-9 checkpoint exists (10 epochs: enough to see when it leaves the plateau);
# the schedule is still the full 30-epoch one, and the checkpoints stay, so a run can be continued later.
# Restart-safe like queue_small_job.sh: finished or stopped runs are skipped, interrupted ones resumed (same Comet
# experiment), runs training elsewhere left alone; two at a time; its own heartbeat file.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
STAGE_EPOCHS=10
MAX_RESUMES=5
Q=sweep_runs
RUNS=()
for lr in 2e-4 1e-4; do for seed in 42 43 44; do for cfg in flex-w64 or-w64; do RUNS+=("$cfg-lr$lr $seed"); done; done; done
HB=$Q/queue_lr.heartbeat
if [ -f $HB ] && [ $(( $(date +%s) - $(stat -c %Y $HB) )) -lt 600 ]; then
  echo "another LR queue is alive (heartbeat $(cat $HB)); not starting"; exit 1
fi
beat() { echo "$(hostname) $$ $(date '+%F %H:%M:%S')" > $HB; }
beat
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue_lr.log; }
declare -A OURS
running() { local n=0; for k in "${!OURS[@]}"; do kill -0 ${OURS[$k]} 2>/dev/null && n=$((n + 1)); done; echo $n; }
last_epoch_ckpt() { ls logs/$1_2026*/ckpts/epoch=*.ckpt 2>/dev/null | sed 's/.*epoch=0*\([0-9]*\)-.*/\1/' | sort -n | tail -1; }
stop_after_stage() {  # stop our runs that have finished stage 1
  for name in "${!OURS[@]}"; do
    pid=${OURS[$name]}
    kill -0 $pid 2>/dev/null || continue
    e=$(last_epoch_ckpt $name)
    if [ -n "$e" ] && [ "$e" -ge $((STAGE_EPOCHS - 1)) ]; then
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; sleep 30; kill -0 $pid 2>/dev/null && { pkill -KILL -P $pid; kill -KILL $pid; }
      echo "=== $(date '+%F %H:%M') stopped after $STAGE_EPOCHS epochs (stage 1 of the LR plateau study) ===" >> logs/${name}_full.out
      log "STOPPED $name after epoch $e (stage 1)"
    fi
  done
}
log "LR queue started in $(hostname): ${#RUNS[@]} planned runs, $STAGE_EPOCHS epochs each, max $MAX_JOBS at a time"
while true; do
  remaining=0
  for r in "${RUNS[@]}"; do
    set -- $r; cfg=$1; seed=$2
    short=$(echo $cfg | sed 's/^or/OR/')-s$seed
    name=TRK-Pix0.6-DQMA-$short
    logf=logs/${name}_full.out
    e=$(last_epoch_ckpt $name)
    [ -n "$e" ] && [ "$e" -ge $((STAGE_EPOCHS - 1)) ] && ! { [ -n "${OURS[$name]}" ] && kill -0 ${OURS[$name]} 2>/dev/null; } && continue
    [ -f "$logf" ] && grep -q 'max_epochs=30. reached' "$logf" && continue
    [ -f "$Q/$short.gaveup" ] && continue
    remaining=$((remaining + 1))
    [ -n "${OURS[$name]}" ] && kill -0 ${OURS[$name]} 2>/dev/null && continue
    if [ -z "${OURS[$name]}" ] && [ -f "$logf" ] && [ $(( $(date +%s) - $(stat -c %Y "$logf") )) -lt 900 ]; then
      continue
    fi
    n_resumes=$(grep -c '^=== .*resuming' "$logf" 2>/dev/null); n_resumes=${n_resumes:-0}
    if [ "$n_resumes" -ge $MAX_RESUMES ]; then
      log "giving up on $name after $MAX_RESUMES resumes"; touch $Q/$short.gaveup; continue
    fi
    while [ "$(running)" -ge $MAX_JOBS ]; do beat; stop_after_stage; sleep 60; done
    if [ -n "$e" ]; then
      nohup $Q/resume_run.sh $short > /dev/null 2>&1 &
      OURS[$name]=$!; log "RESUME $name pid=${OURS[$name]} (from epoch $e)"
    else
      nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > $logf 2>&1 &
      OURS[$name]=$!; log "START $name pid=${OURS[$name]}"
    fi
    sleep 120; beat
  done
  stop_after_stage
  [ $remaining -eq 0 ] && [ "$(running)" -eq 0 ] && break
  beat; sleep 60
done
log "LR queue: all planned runs done"
rm -f $HB
