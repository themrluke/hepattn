# Late training (epoch 29) x2 interleaved, then early training (epoch 0) x1. One run at a time.
cd <scratch>/timing
run() { # stage ckpt variant rep
  INIT_WEIGHTS=$2 TIMING_OUT=<scratch>/timing/$1_$3_$4.json PYTHONPATH=<scratch>/wt_boot /shared/projects/hepattn/.pixi/envs/default/bin/python -u run.py fit --config $3.yaml > $1_$3_$4.log 2>&1
  echo "$(date +%H:%M) $1 $3 rep$4 exit=$?" >> status
}
for rep in 1 2; do for v in original prep trim; do run late "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=029-val_loss=0.29318.ckpt" $v $rep; done; done
for v in original prep trim; do run early "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=000-val_loss=3.65789.ckpt" $v 1; done
echo ALLDONE >> status
