#!/bin/bash
#SBATCH --job-name=bench-m3-70b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=nvl

# Self-contained 70B TP4 job:
#   1. M1 transfer bench on this node (consistency with the serving platform)
#   2. start vLLM 70B TP4 (hbm tier)
#   3. M3: suffix-only re-prefill cost (bench_prefill.py)
#   4. 70B suspend/resume pin/discard demonstration (harness gate, real model)
# Everything runs inside the job; no login-node watcher.

set -uo pipefail
export VLLM_SERVER_DEV_MODE=1
WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
VENV=$WORK/.venv-vllm
PYCLI=$WORK/.venv/bin/python
MODEL=${MODEL:-/scratch/ajoshi34/models/llama-3.1-70b}
NAME=$(basename "$MODEL")
PORT=8302
cd $WORK/harness

# GPU sanity check — n07-style dead-GPU nodes waste the whole job
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -lt 4 ]; then echo "FATAL: only $NGPU GPUs visible on $(hostname); expected 4"; exit 1; fi

echo "=== node $(hostname) $(date -Is)"

echo "=== [1/4] M1+M2 on serving node"
$VENV/bin/python bench_transfer.py \
  --size-gb 2.75 --reps 10 --threads 8 --nvme-dir /tmp \
  --out results/bench_transfer_${SLURM_JOB_ID}.json || echo "M1/M2 FAILED (continuing)"

echo "=== [2/4] starting 70B TP4 server"
"$VENV/bin/vllm" serve "$MODEL" --served-model-name $NAME \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --port $PORT \
    > logs/server_70b_$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
UP=0
for i in $(seq 1 90); do
  curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q llama && { UP=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || break
  sleep 10
done
if [ "$UP" != 1 ]; then
  echo "server failed to start; log tail:"; tail -40 logs/server_70b_$SLURM_JOB_ID.log; exit 1
fi
echo "=== server up after ${i}0s"

echo "=== [3/4] M3 suffix-only re-prefill"
$PYCLI bench_prefill.py --server localhost:$PORT --model $NAME \
  --n-prompts 30 --reps 3 --n-gpus 4 \
  --out results/bench_prefill_${SLURM_JOB_ID}.json || echo "M3 FAILED (continuing)"

echo "=== [4/4] 70B suspend/resume pin/discard"
$PYCLI suspend_resume.py --server localhost:$PORT --model $NAME \
  --tiers pin discard --n-prompts 3 --repeats 5 \
  --out results/suspend_resume_70b_${SLURM_JOB_ID}.jsonl || echo "SR FAILED"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
echo "=== done $(date -Is)"
