#!/bin/bash
# Second round, with gradient values saved: late weights off/off/on, and an early repeat of toggle 1
cd $(dirname $0)
export PYTHONPATH=$(dirname $0)/shim HEPATTN_SRC=/shared/projects/hepattn-overlap/src
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
LATE=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261005-T152222/ckpts/epoch=029-val_loss=0.32403.ckpt
EARLY=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261002-T145644/ckpts/epoch=000-val_loss=3.80308.ckpt
while ps -p 1008560 >/dev/null; do sleep 20; done
for spec in late:off:3 late:off:4 late:on:2 early:off:2; do
  IFS=: read w mode rep <<< "$spec"
  ck=$([ $w = late ] && echo $LATE || echo $EARLY)
  $P identity.py $ck 8 $mode ${w}_${mode}_${rep}.pt > ${w}_${mode}_${rep}.log 2>&1
  echo "$(date '+%H:%M') $spec exit $?" >> status
done
