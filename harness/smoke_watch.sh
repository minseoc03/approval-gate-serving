#!/bin/bash
# Wait for the vLLM serve job to come up, then run a small suspend/resume smoke.
JOB=2017796
for i in $(seq 1 240); do
  STATE=$(squeue -j $JOB -h -o "%T" 2>/dev/null)
  [ -z "$STATE" ] && { echo "job $JOB left queue without serving"; exit 1; }
  if [ "$STATE" = "RUNNING" ]; then
    SRV=$(cut -d' ' -f1 server_info.txt 2>/dev/null)
    if [ -n "$SRV" ] && curl -s --max-time 3 "http://$SRV/v1/models" | grep -q llama; then
      echo "=== server up at $SRV, running smoke ==="
      ../.venv/bin/python suspend_resume.py --server "$SRV" \
        --tiers pin discard --n-prompts 2 --repeats 3 \
        --out results/suspend_resume_smoke.jsonl
      exit $?
    fi
  fi
  sleep 30
done
echo "timed out waiting for server"
exit 1
