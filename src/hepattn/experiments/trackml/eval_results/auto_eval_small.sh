#!/bin/bash
# Evaluate every small-window sweep run as soon as it finishes (in any order), with the same auto_eval.sh as before.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
NAMES=()
for seed in 42 43 44; do for cfg in flex-w64 OR-w64 flex-w32 OR-w32; do NAMES+=("$cfg-s$seed"); done; done
while true; do
  left=0
  for name in "${NAMES[@]}"; do
    [ -f eval_results/${name}_iou0.5.txt ] || [ -f eval_results/${name}.failed ] && continue
    left=$((left + 1))
    log=logs/TRK-Pix0.6-DQMA-${name}_full.out
    # Only the current attempt counts: the log after the last "=== ... resuming" marker (sweep_runs/resume_run.sh)
    [ -f "$log" ] && tac "$log" | sed '/^=== .*resuming/q' | grep -q -E 'max_epochs=30. reached|Traceback' || continue
    if tac "$log" | sed '/^=== .*resuming/q' | grep -q Traceback; then
      echo "$(date '+%F %H:%M') $name training FAILED, not evaluated" >> eval_results/auto_eval.log
      touch eval_results/${name}.failed
      continue
    fi
    dir=$(ls -d logs/TRK-Pix0.6-DQMA-${name}_2026* | head -1)
    $P - "$dir" <<'PY'
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
    eval_results/auto_eval.sh "$name" "logs/TRK-Pix0.6-DQMA-${name}_2026*" "$log" "$dir/test_config.yaml"
  done
  [ $left -eq 0 ] && break
  sleep 300
done
echo "$(date '+%F %H:%M') small-window evaluation loop finished" >> eval_results/auto_eval.log
