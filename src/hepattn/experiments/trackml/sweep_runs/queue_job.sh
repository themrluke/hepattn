#!/bin/bash
# Training queue for a study, run inside a Kubernetes Job (sweep_runs/k8s/<study>-job.yaml):
#
#   bash sweep_runs/queue_job.sh <study>
#
# The runs are listed in sweep_runs/<study>.runs, one "<config suffix> <seed>" per line, e.g. "or-w16 42" for
# configs/tracking-eta4-pt600-or-w16.yaml and run TRK-Pix0.6-DQMA-OR-w16-s42. Two trainings at a time, in list order.
#
# Every run's state is an explicit marker in sweep_runs/<study>_state/:
#   <run>.done      its current attempt reached max_epochs=30
#   <run>.diverged  its current attempt crashed with a CUDA device-side assert (what a blown-up run does here);
#                   a result, never resumed (the evaluation loop still tests its best checkpoint)
#   <run>.gaveup    interrupted MAX_RESUMES times
# Otherwise a run is resumed from its latest checkpoint into the same Comet experiment (resume_run.sh), or started.
# A run whose log was written in the last FRESH seconds and that this queue did not start is left alone (after a job
# restart, the old pod's runs are resumed once their logs go quiet).
#
# Written after the LR queue's bugs (21 Oct): every pass re-reads the state from scratch, and the helper functions
# keep their variables local, so nothing carries over between runs.
#
# QUEUE_ROOT, QUEUE_PY, QUEUE_RESUME, QUEUE_TICK and QUEUE_FRESH override the defaults (used by the dry-run test).
study=${1:?usage: queue_job.sh <study>}
cd ${QUEUE_ROOT:-/shared/projects/hepattn/src/hepattn/experiments/trackml} || exit 1
P=${QUEUE_PY:-/shared/projects/hepattn/.pixi/envs/default/bin/python}
Q=sweep_runs
RESUME=${QUEUE_RESUME:-$Q/resume_run.sh}
TICK=${QUEUE_TICK:-60}
FRESH=${QUEUE_FRESH:-900}
MAX_JOBS=2
MAX_RESUMES=5
S=$Q/${study}_state
mkdir -p $S
mapfile -t RUNS < <(grep -v '^\s*#' $Q/$study.runs | grep -v '^\s*$')
HB=$Q/queue_$study.heartbeat
if [ -f $HB ] && [ $(( $(date +%s) - $(stat -c %Y $HB) )) -lt $((10 * TICK)) ]; then
  echo "another $study queue is alive (heartbeat $(cat $HB)); not starting"; exit 1
fi
beat() { echo "$(hostname) $$ $(date '+%F %H:%M:%S')" > $HB; }
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue_$study.log; }
declare -A OURS   # run name -> pid of the training this queue started

short_of() { local cfg=$1 seed=$2; echo "$(echo $cfg | sed 's/^or/OR/')-s$seed"; }
attempt() { tac "$1" 2>/dev/null | sed '/^=== /q'; }   # the log since the last resume marker
alive() { [ -n "${OURS[$1]}" ] && kill -0 "${OURS[$1]}" 2>/dev/null; }
running() { local n=0 k; for k in "${!OURS[@]}"; do alive "$k" && n=$((n + 1)); done; echo $n; }
finished() { [ -f $S/$1.done ] || [ -f $S/$1.diverged ] || [ -f $S/$1.gaveup ]; }

# Mark a run that is not training (any more) from its log: done, diverged, or nothing (to be resumed).
classify() {
  local short=$1 logf=logs/TRK-Pix0.6-DQMA-$1_full.out
  [ -f "$logf" ] || return
  if attempt "$logf" | grep -q 'max_epochs=30. reached'; then
    touch $S/$short.done; log "DONE TRK-Pix0.6-DQMA-$short"
  elif attempt "$logf" | grep -q 'device-side assert'; then
    touch $S/$short.diverged; log "DIVERGED TRK-Pix0.6-DQMA-$short (CUDA device-side assert); not resumed"
  fi
}

# Start or resume one run; returns 1 if the run can't be started now.
launch() {
  local cfg=$1 seed=$2 short name logf n
  short=$(short_of $cfg $seed); name=TRK-Pix0.6-DQMA-$short; logf=logs/${name}_full.out
  if [ -f "$logf" ] && [ $(( $(date +%s) - $(stat -c %Y "$logf") )) -lt $FRESH ]; then
    return 1   # training somewhere else (or the previous pod's run, not yet quiet)
  fi
  if ls logs/${name}_2026*/ckpts/epoch=*.ckpt >/dev/null 2>&1; then
    n=$(grep -c '^=== .*resuming' "$logf" 2>/dev/null); n=${n:-0}
    if [ "$n" -ge $MAX_RESUMES ]; then
      touch $S/$short.gaveup; log "GAVE UP on $name after $MAX_RESUMES resumes"; return 1
    fi
    nohup $RESUME $short > /dev/null 2>&1 &
    OURS[$name]=$!; log "RESUME $name pid=${OURS[$name]}"
  else
    nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > $logf 2>&1 &
    OURS[$name]=$!; log "START $name pid=${OURS[$name]}"
  fi
}

beat
log "queue for $study started in $(hostname): ${#RUNS[@]} planned runs, max $MAX_JOBS at a time"
while true; do
  # 1. our runs that have stopped: mark them
  for name in "${!OURS[@]}"; do
    alive "$name" && continue
    classify "${name#TRK-Pix0.6-DQMA-}"; unset "OURS[$name]"
  done
  # 2. what is left, and start one run if there is a free slot
  left=0; started=0
  for r in "${RUNS[@]}"; do
    set -- $r
    short=$(short_of $1 $2); name=TRK-Pix0.6-DQMA-$short
    finished $short && continue
    alive "$name" && { left=$((left + 1)); continue; }
    classify $short; finished $short && continue       # finished while no queue was watching
    left=$((left + 1))
    if [ $started -eq 0 ] && [ "$(running)" -lt $MAX_JOBS ]; then
      launch $1 $2 && started=1
    fi
  done
  [ $left -eq 0 ] && break
  beat
  [ $started -eq 1 ] && sleep $((2 * TICK)) || sleep $TICK
done
log "queue for $study: every planned run is done, diverged or given up"
rm -f $HB
wait
