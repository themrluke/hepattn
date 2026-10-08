#!/bin/bash
# Evaluate every window-sweep run as it finishes, in queue order, with the same auto_eval.sh as A' and D.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
for name in flex-w256-s42 OR-w256-s42 flex-w128-s42 OR-w128-s42 flex-w256-s43 OR-w256-s43 flex-w128-s43 OR-w128-s43; do
  log=logs/TRK-Pix0.6-DQMA-${name}_full.out
  [ -f eval_results/${name}_iou0.5.txt ] && continue                       # already evaluated
  until [ -f "$log" ] && ls -d logs/TRK-Pix0.6-DQMA-${name}_2026* >/dev/null 2>&1; do sleep 300; done
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
echo "$(date '+%F %H:%M') sweep evaluation loop finished" >> eval_results/auto_eval.log
