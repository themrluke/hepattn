#!/bin/bash
# Resume an interrupted run (e.g. `resume_run.sh flex-w32-s42`) from its latest checkpoint, in any of its run dirs,
# continuing the SAME Comet experiment (its key is the first Comet URL in the run's log), so the curves carry on.
# Output is appended to logs/<name>_full.out, so the evaluation loop sees the run finish.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
P=/shared/projects/hepattn/.pixi/envs/default/bin/python
name=TRK-Pix0.6-DQMA-$1
log=logs/${name}_full.out
ckpt=$(ls logs/${name}_2026*/ckpts/epoch=*.ckpt 2>/dev/null | awk -F'epoch=' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$ckpt" ] || { echo "no checkpoint for $name" >&2; exit 1; }
cfg=$(dirname "$(dirname "$ckpt")")/config.yaml
key=$(grep -ao "comet.com/[^/ ]*/[^/ ]*/[0-9a-f]\{32\}" "$log" | head -1 | sed 's|.*/||')
comet=()
[ -n "$key" ] && comet=(--trainer.logger.init_args.experiment_key "$key" --trainer.logger.init_args.mode get)
echo "=== $(date '+%F %H:%M') resuming from $ckpt, Comet experiment ${key:-NEW} ===" >> "$log"
exec $P run_tracking.py fit --config "$cfg" --ckpt_path "$ckpt" "${comet[@]}" >> "$log" 2>&1
