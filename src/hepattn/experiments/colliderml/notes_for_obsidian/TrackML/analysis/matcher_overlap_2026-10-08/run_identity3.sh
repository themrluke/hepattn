#!/bin/bash
# Third round: toggle 2 stacked vs toggle 2 + overlap, late and early weights
cd $(dirname $0)
export PYTHONPATH=$(dirname $0)/shim HEPATTN_SRC=/shared/projects/hepattn-overlap/src
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
LATE=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261005-T152222/ckpts/epoch=029-val_loss=0.32403.ckpt
EARLY=/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261002-T145644/ckpts/epoch=000-val_loss=3.80308.ckpt
while ps -p 1015000 >/dev/null; do sleep 20; done
for spec in late:trimoff:1 late:trimon:1 early:trimoff:1 early:trimon:1; do
  IFS=: read w mode rep <<< "$spec"
  ck=$([ $w = late ] && echo $LATE || echo $EARLY)
  $P identity.py $ck 8 $mode ${w}_${mode}_${rep}.pt > ${w}_${mode}_${rep}.log 2>&1
  echo "$(date '+%H:%M') $spec exit $?" >> status
done
