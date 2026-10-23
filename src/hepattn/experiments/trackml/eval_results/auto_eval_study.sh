#!/bin/bash
# Evaluate each run of a study once the queue has marked it (sweep_runs/queue_job.sh), in any order:
#
#   bash eval_results/auto_eval_study.sh <study>
#
# done or diverged -> best checkpoint by validation loss (over every attempt dir) -> test predictions -> UCL's
# evaluator at IoU 0.5 and 0 -> eval_results/<run>_iou{0.5,0.0}.txt. The queue's markers are the only judge of a
# run's state (the old loop marked runs failed from a traceback that a later resume made irrelevant).
# Ends once every planned run is evaluated, given up, or failed to evaluate twice.
study=${1:?usage: auto_eval_study.sh <study>}
cd ${QUEUE_ROOT:-/shared/projects/hepattn/src/hepattn/experiments/trackml} || exit 1
P=${QUEUE_PY:-/shared/projects/hepattn/.pixi/envs/default/bin/python}
S=sweep_runs/${study}_state
TICK=${QUEUE_TICK:-60}
mapfile -t RUNS < <(grep -v '^\s*#' sweep_runs/$study.runs | grep -v '^\s*$')
log() { echo "$(date '+%F %H:%M') $*" >> eval_results/auto_eval.log; }

evaluate() {
  local short=$1 name=TRK-Pix0.6-DQMA-$1 dir best h5 iou
  dir=$(ls -d logs/${name}_2026* | head -1)
  best=$(ls logs/${name}_2026*/ckpts/epoch=*-val_loss=*.ckpt | awk -F'val_loss=' '{v=$2; sub(/\.ckpt$/,"",v); print v, $0}' | sort -g | head -1 | cut -d' ' -f2-)
  [ -n "$best" ] || return 1
  $P - "$dir" <<'PY' || return 1
import sys, yaml
d = sys.argv[1]
c = yaml.safe_load(open(f"{d}/config.yaml"))
c.pop("ckpt_path", None)
c["trainer"]["logger"] = False
for cb in c["trainer"]["callbacks"]:
    if cb["class_path"].endswith("PredictionWriter"):
        cb["init_args"]["write_paper_compatible_test"] = True
yaml.safe_dump(c, open(f"{d}/test_config.yaml", "w"), sort_keys=False)
PY
  log "$short: testing best checkpoint $best"
  $P run_tracking.py test --config "$dir/test_config.yaml" --ckpt_path "$best" > logs/${short}_test.out 2>&1 || return 1
  h5="$PWD/${best%.ckpt}__test.h5"
  for iou in 0.5 0.0; do
    IOU=$iou $P eval_results/eval_runs.py "$short=$h5" 2>&1 | grep -v -i warn > eval_results/${short}_iou${iou}.tmp || return 1
    mv eval_results/${short}_iou${iou}.tmp eval_results/${short}_iou${iou}.txt
  done
  log "$short evaluated: eval_results/${short}_iou*.txt$([ -f $S/$short.diverged ] && echo ' (run had diverged)')"
}

log "evaluation loop for $study started in $(hostname)"
while true; do
  left=0
  for r in "${RUNS[@]}"; do
    set -- $r
    short=$(echo $1 | sed 's/^or/OR/')-s$2
    [ -f eval_results/${short}_iou0.0.txt ] && continue
    [ -f $S/$short.gaveup ] && continue
    n=$(cat $S/$short.evalfail 2>/dev/null || echo 0)
    [ "$n" -ge 2 ] && continue
    left=$((left + 1))
    [ -f $S/$short.done ] || [ -f $S/$short.diverged ] || continue
    if ! evaluate $short; then
      echo $((n + 1)) > $S/$short.evalfail; log "$short: evaluation FAILED (attempt $((n + 1)) of 2; see logs/${short}_test.out)"
    fi
  done
  [ $left -eq 0 ] && break
  sleep $((5 * TICK))
done
log "evaluation loop for $study finished"
