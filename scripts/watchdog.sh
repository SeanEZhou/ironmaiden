#!/usr/bin/env bash
# Training watchdog: restart the queue runner if it died, report GPU + job state.
cd "$(dirname "$0")/.."
Q=training/queue
alive=0
if [ -f $Q/runner.pid ] && kill -0 "$(cat $Q/runner.pid)" 2>/dev/null; then alive=1; fi
if [ $alive -eq 0 ]; then
  echo "RUNNER DEAD -> restarting"
  nohup .venv/bin/python -u training/queue_runner.py >> $Q/runner.log 2>&1 &
  sleep 2
fi
echo "== status"; cat $Q/status.json 2>/dev/null; echo
echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
echo "== pending"; ls $Q/pending 2>/dev/null
echo "== done";    for f in $Q/done/*.json; do [ -f "$f" ] && python3 -c "import json,sys;j=json.load(open('$f'));print(j['name'],j.get('minutes'),'min rc',j.get('returncode'))"; done
echo "== failed";  ls $Q/failed 2>/dev/null
job=$(python3 -c "import json;print(json.load(open('$Q/status.json')).get('job',''))" 2>/dev/null)
if [ -n "$job" ]; then echo "== $job (tail)"; grep -E "^\[|Error|Traceback|DONE|success_rate|Killed|XlaRuntimeError|out of memory" training/runs/$job/train.log | tail -6; fi
