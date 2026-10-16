#!/bin/bash
# After the notebook session ends: one command brings the small-window sweep back (queue + evaluation loop), detached.
cd /shared/projects/hepattn/src/hepattn/experiments/trackml
# Running already? (checked by PID file, not pgrep -f, which can match the very command that started this)
p=$(cat sweep_runs/queue_small_auto.pid 2>/dev/null)
[ -n "$p" ] && kill -0 "$p" 2>/dev/null && grep -q queue_small_auto /proc/$p/cmdline 2>/dev/null && { echo "queue already running (pid $p)"; exit 1; }
setsid nohup sweep_runs/queue_small_auto.sh > sweep_runs/queue_small_auto.out 2>&1 < /dev/null &
running_eval=no
for q in $(ls /proc | grep -E '^[0-9]+$'); do grep -qa "eval_results/auto_eval_small.sh" /proc/$q/cmdline 2>/dev/null && grep -qa "^/bin/bash" /proc/$q/cmdline && running_eval=yes; done
[ $running_eval = yes ] || setsid nohup eval_results/auto_eval_small.sh > eval_results/auto_eval_small.out 2>&1 < /dev/null &
sleep 2; tail -3 sweep_runs/queue_small.log
