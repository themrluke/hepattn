#!/bin/bash
# Learning-rate plateau study (Kubernetes Job sweep_runs/k8s/lr-plateau-job.yaml): window 64, flex and OR, seeds 42-44,
# peak learning rate 2e-4 then 1e-4 (default 5e-5; same seeds as the small-window sweep, so every run has a control).
# Stage 1: each run is stopped once its epoch-9 checkpoint exists (10 epochs, enough to see when it leaves the plateau);
# the schedule is still the full 30-epoch one, and the checkpoints stay.
#
# Every run's state is an explicit marker in sweep_runs/lr_state/ (v2 of this queue, 21 Oct: v1 resumed runs that had
# already finished, kept resuming diverged ones and skipped two):
#   <run>.done      reached epoch 10; never touched again
#   <run>.diverged  crashed with a CUDA device-side assert (what a blown-up run does here); a result, never resumed
# Otherwise: interrupted runs (session or job end) are resumed from their last checkpoint into the same Comet experiment,
# or started fresh if they have none. Two at a time; heartbeat file so only one LR queue runs.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
STAGE_EPOCHS=10
Q=sweep_runs
S=$Q/lr_state
mkdir -p $S
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
alive() { [ -n "${OURS[$1]}" ] && kill -0 ${OURS[$1]} 2>/dev/null; }
running() { local n=0; for k in "${!OURS[@]}"; do alive $k && n=$((n + 1)); done; echo $n; }
last_epoch() { ls logs/$1_2026*/ckpts/epoch=*.ckpt 2>/dev/null | sed 's/.*epoch=\([0-9]*\)-.*/\1/' | sed 's/^0*\([0-9]\)/\1/' | sort -n | tail -1; }
attempt() { tac "$1" | sed '/^=== /q'; }   # the log since the last resume/stop marker
stop_finished() {
  for name in "${!OURS[@]}"; do
    alive $name || continue
    e=$(last_epoch $name)
    if [ -n "$e" ] && [ "$e" -ge $((STAGE_EPOCHS - 1)) ]; then
      pid=${OURS[$name]}
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; sleep 30
      kill -0 $pid 2>/dev/null && { pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; }
      touch $S/${name#TRK-Pix0.6-DQMA-}.done
      echo "=== $(date '+%F %H:%M') stopped after $STAGE_EPOCHS epochs (stage 1 of the LR plateau study) ===" >> logs/${name}_full.out
      log "DONE $name (stopped after epoch $e)"
    fi
  done
}
log "LR queue v2 started in $(hostname): ${#RUNS[@]} planned runs, $STAGE_EPOCHS epochs each, max $MAX_JOBS at a time"
while true; do
  stop_finished
  remaining=0
  for r in "${RUNS[@]}"; do
    set -- $r; cfg=$1; seed=$2
    short=$(echo $cfg | sed 's/^or/OR/')-s$seed
    name=TRK-Pix0.6-DQMA-$short
    logf=logs/${name}_full.out
    [ -f $S/$short.done ] || [ -f $S/$short.diverged ] && continue
    remaining=$((remaining + 1))
    alive $name && continue
    e=$(last_epoch $name)
    if [ -n "$e" ] && [ "$e" -ge $((STAGE_EPOCHS - 1)) ]; then
      touch $S/$short.done; log "DONE $name (found finished, epoch $e)"; continue
    fi
    if [ -f "$logf" ] && attempt "$logf" | grep -q "device-side assert"; then
      touch $S/$short.diverged; log "DIVERGED $name (CUDA device-side assert; last epoch ${e:-none}); not resumed"; continue
    fi
    if [ -z "${OURS[$name]}" ] && [ -f "$logf" ] && [ $(( $(date +%s) - $(stat -c %Y "$logf") )) -lt 900 ]; then
      continue   # training somewhere else
    fi
    while [ "$(running)" -ge $MAX_JOBS ]; do beat; stop_finished; sleep 60; done
    if [ -n "$e" ]; then
      nohup $Q/resume_run.sh $short > /dev/null 2>&1 &
      OURS[$name]=$!; log "RESUME $name pid=${OURS[$name]} (from epoch $e)"
    else
      nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > $logf 2>&1 &
      OURS[$name]=$!; log "START $name pid=${OURS[$name]}"
    fi
    sleep 120; beat
  done
  [ $remaining -eq 0 ] && break
  beat; sleep 60
done
log "LR queue: all planned runs done or diverged"
rm -f $HB
