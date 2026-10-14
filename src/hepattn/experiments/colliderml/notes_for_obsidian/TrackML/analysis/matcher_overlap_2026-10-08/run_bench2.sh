#!/bin/bash
# Speed benchmark of the 14 Oct options (fast_permute, slice_valid_targets, scipy, single-product focal cost) on top of the 11 Oct best
# and 29 (late), 600 steps each, one run at a time on an otherwise idle GPU; then two profiled runs (late weights).
cd "$(dirname "$0")"
export PYTHONPATH=$(dirname $0)/shim HEPATTN_SRC=/shared/projects/hepattn-overlap/src
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
EARLY=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=000-val_loss=3.65789.ckpt
LATE=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=029-val_loss=0.29318.ckpt
mkdir -p timing2 profile
# Wait until nothing else uses the GPU (two checks a minute apart below 2 GB)
idle() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; }
until idle && sleep 60 && idle; do sleep 120; done
echo "$(date '+%F %H:%M') GPU idle, starting" >> bench2_status
run() { # stage ckpt variant rep [profile]
  INIT_WEIGHTS=$2 TIMING_OUT=timing2/$1_$3_$4.json PROFILE_OUT=${5:+profile/$1_$3} $P -u bench_run.py fit --config configs/$3.yaml > timing2/$1_$3_$4.log 2>&1
  echo "$(date '+%F %H:%M') $1 $3 rep$4 exit=$?" >> bench2_status
}
VARIANTS="otr otrps otrpss otrpssf"
for rep in 1 2; do for v in $VARIANTS; do run late $LATE $v $rep; done; done
for v in $VARIANTS; do run early $EARLY $v 1; done
for v in otrpss; do run late $LATE $v p 1; done
echo "$(date '+%F %H:%M') ALLDONE" >> bench2_status
