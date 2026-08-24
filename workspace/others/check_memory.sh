#!/bin/bash

INTERVAL=${1:-5}
LOG_FILE="memory_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "开始监控内存，间隔 ${INTERVAL}s，日志: $LOG_FILE"
echo "timestamp,used_mem_mb,mem_percent" | tee "$LOG_FILE"

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    read TOTAL USED _ <<< $(free -m | awk 'NR==2{print $2,$3,$4}')
    MEM_PERCENT=$(awk "BEGIN{printf \"%.1f\", $USED/$TOTAL*100}")
    echo "$TIMESTAMP  used=${USED}MB  percent=${MEM_PERCENT}%" | tee -a "$LOG_FILE"
    sleep $INTERVAL
done