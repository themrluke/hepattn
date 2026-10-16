#!/bin/bash
# Small-window sweep queue for running inside a Kubernetes Job (sweep_runs/k8s/small-sweep-job.yaml).
# Restart-safe and shares the work with anything still training elsewhere (e.g. in the notebook session):
#   - finished runs (log has "max_epochs=30 reached") are skipped
#   - a run whose log was written in the last 15 min, and that this queue did not start, is training somewhere else:
#     left alone, and checked again on the next pass (if that session dies, the run is resumed here once its log goes quiet)
#   - otherwise the run is resumed from its latest checkpoint into the same Comet experiment (resume_run.sh), or started
# Two trainings at a time on this job's GPU; passes repeat every 5 min until every planned run has finished.
# Refuses to start while another job queue is alive (heartbeat file on /shared, touched every minute).
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
MAX_JOBS=2
MAX_RESUMES=5
Q=sweep_runs
RUNS=()
for seed in 42 43 44; do for cfg in flex-w64 or-w64 flex-w32 or-w32; do RUNS+=("$cfg $seed"); done; done
HB=$Q/queue_small.heartbeat
if [ -f $HB ] && [ $(( $(date +%s) - $(stat -c %Y $HB) )) -lt 600 ]; then
  echo "another queue is alive (heartbeat $(cat $HB)); not starting"; exit 1
fi
beat() { echo "$(hostname) $$ $(date '+%F %H:%M:%S')" > $HB; }
beat
log() { echo "$(date '+%F %H:%M') $*" >> $Q/queue_small.log; }
declare -A OURS   # run name -> pid of the training this queue started
running() { local n=0; for k in "${!OURS[@]}"; do kill -0 ${OURS[$k]} 2>/dev/null && n=$((n + 1)); done; echo $n; }
log "job queue started in $(hostname): ${#RUNS[@]} planned runs, max $MAX_JOBS at a time"
while true; do
  remaining=0
  for r in "${RUNS[@]}"; do
    set -- $r; cfg=$1; seed=$2
    short=$(echo $cfg | sed 's/^or/OR/')-s$seed
    name=TRK-Pix0.6-DQMA-$short
    logf=logs/${name}_full.out
    [ -f "$logf" ] && grep -q 'max_epochs=30. reached' "$logf" && continue
    [ -f "$Q/$short.gaveup" ] && continue
    remaining=$((remaining + 1))
    [ -n "${OURS[$name]}" ] && kill -0 ${OURS[$name]} 2>/dev/null && continue          # ours, still training
    if [ -z "${OURS[$name]}" ] && [ -f "$logf" ] && [ $(( $(date +%s) - $(stat -c %Y "$logf") )) -lt 900 ]; then
      continue                                                                           # training elsewhere
    fi
    n_resumes=$(grep -c '^=== .*resuming' "$logf" 2>/dev/null); n_resumes=${n_resumes:-0}
    if [ "$n_resumes" -ge $MAX_RESUMES ]; then
      log "giving up on $name after $MAX_RESUMES resumes"; touch $Q/$short.gaveup; continue
    fi
    while [ "$(running)" -ge $MAX_JOBS ]; do beat; sleep 60; done
    if ls logs/${name}_2026*/ckpts/epoch=*.ckpt >/dev/null 2>&1; then
      nohup $Q/resume_run.sh $short > /dev/null 2>&1 &
      OURS[$name]=$!; log "RESUME $name pid=${OURS[$name]} (job)"
    else
      nohup $P run_tracking.py fit --config configs/tracking-eta4-pt600-$cfg.yaml --seed_everything $seed --name $name > $logf 2>&1 &
      OURS[$name]=$!; log "START $name pid=${OURS[$name]} (job)"
    fi
    sleep 120; beat
  done
  [ $remaining -eq 0 ] && break
  beat; sleep 300
done
log "all planned runs finished (job queue)"
rm -f $HB
wait
