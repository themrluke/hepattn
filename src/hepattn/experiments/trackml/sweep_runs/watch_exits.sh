#!/bin/bash
# Records when each queued run's process ends and how its log ends (for the queue log).
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
while true; do
  for f in logs/TRK-Pix0.6-DQMA-*-s4[23]_full.out logs/TRK-Pix0.6-DQMA-{flex,or}_resume.out; do
    [ -f "$f" ] || continue
    grep -q "^$f" sweep_runs/finished.txt 2>/dev/null && continue
    if grep -q -E 'max_epochs=30. reached|Traceback' "$f"; then
      echo "$f $(date '+%F %H:%M') $(grep -q Traceback "$f" && echo FAILED || echo finished)" >> sweep_runs/finished.txt
    fi
  done
  sleep 300
done
