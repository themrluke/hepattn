#!/bin/bash
# Extra repeat of the reference variant (otr), after run_bench2.sh: its first repeat had 32 s of stalls (steps 155-280)
cd "$(dirname "$0")"
export PYTHONPATH=./shim HEPATTN_SRC=/shared/projects/hepattn-overlap/src
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
LATE=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=029-val_loss=0.29318.ckpt
until grep -q ALLDONE bench2_status; do sleep 20; done
INIT_WEIGHTS=$LATE TIMING_OUT=timing2/late_otr_3.json $P -u bench_run.py fit --config configs/otr.yaml > timing2/late_otr_3.log 2>&1
echo "$(date '+%F %H:%M') late otr rep3 exit=$?" >> bench2_status
echo "$(date '+%F %H:%M') ALLDONE2" >> bench2_status
