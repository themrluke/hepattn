#!/bin/bash
# Identity runs: late weights (A' epoch 29) off/off/on, early weights (A' epoch 0) off/on, 8 validation events each
cd $(dirname $0)
export PYTHONPATH=$(dirname $0)/shim HEPATTN_SRC=/shared/projects/hepattn-overlap/src
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
LATE=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261005-T152222/ckpts/epoch=029-val_loss=0.32403.ckpt
EARLY=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261002-T145644/ckpts/epoch=000-val_loss=3.80308.ckpt
for spec in late:off:1 late:off:2 late:on:1 early:off:1 early:on:1; do
  IFS=: read w mode rep <<< "$spec"
  ck=$([ $w = late ] && echo $LATE || echo $EARLY)
  $P identity.py $ck 8 $mode ${w}_${mode}_${rep}.pt > ${w}_${mode}_${rep}.log 2>&1
  echo "$(date '+%H:%M') $spec exit $?" >> status
done
