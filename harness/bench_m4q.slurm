#!/bin/bash
#SBATCH --job-name=bench-m4q
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=nvl

# Self-contained q + M4 job on 70B TP4 (CLAUDE.md 10.1):
#   Phase A (q)  : Poisson background load with periodic cold-prefill
#                  "resume" injections, at three load points.
#   Phase B (M4) : alpha_HBM occupancy curve — emulate N suspended gate
#                  contexts by shrinking the KV pool via --kv-cache-memory,
#                  measure background goodput/TTFT at fixed load per N.
# KV arithmetic (measured, job 2017796): 51.94 GiB/GPU total KV;
# one median gate context = 8400 tok x 80 KiB/tok/GPU = 672 MiB/GPU.

set -uo pipefail
export VLLM_SERVER_DEV_MODE=1
WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
VENV=$WORK/.venv-vllm
PYCLI=$WORK/.venv/bin/python
MODEL=${MODEL:-/scratch/ajoshi34/models/llama-3.1-70b}
NAME=$(basename "$MODEL")
PORT=8303
TOTAL_KV_BYTES=55766715596          # 51.94 GiB per GPU, from job 2017796
CTX_BYTES=$((8400 * 81920))         # one gate context per GPU under TP4
cd $WORK/harness

# GPU sanity check — n07-style dead-GPU nodes waste the whole job
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -lt 4 ]; then echo "FATAL: only $NGPU GPUs visible on $(hostname); expected 4"; exit 1; fi

start_server () {  # $1 = kv-cache-memory bytes per GPU ("" = default util)
  local kvarg=()
  [ -n "$1" ] && kvarg=(--kv-cache-memory "$1")
  "$VENV/bin/vllm" serve "$MODEL" --served-model-name $NAME \
      --tensor-parallel-size 4 --max-model-len 32768 \
      --gpu-memory-utilization 0.92 --port $PORT "${kvarg[@]}" \
      > logs/server_m4q_${SLURM_JOB_ID}_$2.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 90); do
    curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q llama && return 0
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died ($2):"; tail -20 logs/server_m4q_${SLURM_JOB_ID}_$2.log; return 1; }
    sleep 10
  done
  return 1
}
stop_server () { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; sleep 5; }

echo "=== [$(date -Is)] Phase A: contention coefficient q"
if start_server "" qA; then
  for RATE in 1.0 2.0 4.0; do
    echo "--- q at rate=$RATE"
    $PYCLI bench_q.py --server localhost:$PORT --model $NAME \
      --rate $RATE --duration 300 --inject-every 30 --max-tokens 64 \
      --out results/bench_q_rate${RATE}_${SLURM_JOB_ID}.jsonl || echo "q rate=$RATE FAILED"
  done
else
  echo "Phase A server failed"
fi
stop_server

echo "=== [$(date -Is)] Phase B: alpha_HBM occupancy sweep"
for N in 0 25 50 65 72; do
  KV=$((TOTAL_KV_BYTES - N * CTX_BYTES))
  echo "--- N=$N suspended contexts emulated (kv-cache-memory=$KV)"
  if start_server "$KV" "N$N"; then
    $PYCLI loadgen.py --server localhost:$PORT --model $NAME \
      --rate 3.0 --duration 150 --max-tokens 128 \
      --out results/bench_occupancy_N${N}_${SLURM_JOB_ID}.jsonl || echo "N=$N FAILED"
  else
    echo "N=$N server failed to start"
  fi
  stop_server
done

echo "=== [$(date -Is)] all done"
