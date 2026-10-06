# Every 5 s: time, host load average, GPU memory used and utilization (device-wide, all users), our own process count
while true; do
  echo "$(date +%s.%N | cut -c1-14) $(cut -d' ' -f1-4 /proc/loadavg) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ') $(ps -e --no-headers | wc -l)"
  sleep 5
done
