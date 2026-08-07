#!/bin/bash
# Download 8B, submit 1-GPU serve job, wait for it, run pin/discard smoke.
set -uo pipefail
WORK=$(cd "$(dirname "$0")/.." && pwd)
cd $WORK/harness

M8B=${M8B:?set M8B to the target checkpoint dir}
if [ ! -f "$M8B/config.json" ]; then
  echo "=== downloading Llama-3.1-8B-Instruct ==="
  $WORK/.venv-vllm/bin/hf download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir "$M8B" --exclude "original/*" || exit 1
fi

rm -f server_info_llama-3.1-8b.txt
JOB=$(MODEL=$M8B TP=1 PORT=8301 sbatch --parsable --gres=gpu:1 --mem=100G \
      --cpus-per-task=16 --time=01:00:00 --export=ALL,MODEL=$M8B,TP=1,PORT=8301 \
      serve_vllm.slurm) || exit 1
echo "=== submitted 8B serve job $JOB ==="

for i in $(seq 1 2880); do
  STATE=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
  [ -z "$STATE" ] && { echo "job $JOB exited before serving; tail of log:"; tail -20 logs/vllm-70b-$JOB.err 2>/dev/null; exit 1; }
  if [ "$STATE" = "RUNNING" ] && [ -f server_info_llama-3.1-8b.txt ]; then
    SRV=$(cut -d' ' -f1 server_info_llama-3.1-8b.txt)
    if curl -s --max-time 3 "http://$SRV/v1/models" | grep -q llama; then
      echo "=== 8B server up at $SRV, running pin/discard smoke ==="
      $WORK/.venv/bin/python suspend_resume.py --server "$SRV" \
        --model llama-3.1-8b --tiers pin discard --n-prompts 2 --repeats 3 \
        --out results/suspend_resume_smoke8b.jsonl
      RC=$?
      echo "=== smoke rc=$RC; cancelling serve job ==="
      scancel "$JOB"
      exit $RC
    fi
  fi
  sleep 30
done
echo "timed out"; scancel "$JOB" 2>/dev/null; exit 1
