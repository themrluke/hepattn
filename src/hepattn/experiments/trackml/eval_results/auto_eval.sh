#!/bin/bash
# After a run finishes: best checkpoint (lowest val loss, original + resumed dirs) -> test predictions -> UCL evaluator.
# Usage: auto_eval.sh TAG RUN_GLOB TRAIN_LOG TEST_CONFIG
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
tag=$1; glob=$2; trainlog=$3; testcfg=$4
until grep -q -E 'max_epochs=30. reached|Traceback' "$trainlog"; do sleep 300; done
grep -q Traceback "$trainlog" && { echo "$(date '+%F %H:%M') $tag training FAILED, not evaluated" >> eval_results/auto_eval.log; exit 1; }
sleep 120
best=$(ls $glob/ckpts/epoch=*-val_loss=*.ckpt | awk -F'val_loss=' '{v=$2; sub(/\.ckpt$/,"",v); print v, $0}' | sort -g | head -1 | cut -d' ' -f2-)
echo "$(date '+%F %H:%M') $tag finished; best checkpoint $best" >> eval_results/auto_eval.log
$P run_tracking.py test --config "$testcfg" --ckpt_path "$best" > logs/${tag}_test.out 2>&1
h5="$PWD/${best%.ckpt}__test.h5"
for iou in 0.5 0.0; do IOU=$iou $P eval_results/eval_runs.py "$tag=$h5" 2>&1 | grep -v -i warn > eval_results/${tag}_iou${iou}.txt; done
echo "$(date '+%F %H:%M') $tag evaluated: eval_results/${tag}_iou*.txt" >> eval_results/auto_eval.log
