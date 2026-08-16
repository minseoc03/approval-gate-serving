#!/bin/bash
#SBATCH --job-name=bench-m4real
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=nvl

# M4-real: occupancy curve with REAL suspended contexts held resident by
# keepalive touches (no --kv-cache-memory emulation). Validates or replaces
# the emulated Fig 1; also runs a sub-saturation rate for a clean cliff.

set -uo pipefail
export VLLM_SERVER_DEV_MODE=1
WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
VENV=$WORK/.venv-vllm
PYCLI=$WORK/.venv/bin/python
MODEL=${MODEL:-/scratch/ajoshi34/models/llama-3.1-70b}
NAME=$(basename "$MODEL")
PORT=8304
cd $WORK/harness

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -lt 4 ]; then echo "FATAL: only $NGPU GPUs on $(hostname)"; exit 1; fi

echo "=== node $(hostname) $(date -Is)"
"$VENV/bin/vllm" serve "$MODEL" --served-model-name $NAME \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --port $PORT \
    > logs/server_m4real_$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 90); do
  curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q llama && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -20 logs/server_m4real_$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done

$PYCLI bench_occupancy_real.py --server localhost:$PORT --model $NAME \
  --Ns 0 25 50 65 72 --rates 1.5 3.0 --duration 150 --keepalive 30 \
  --out results/bench_occupancy_real_$SLURM_JOB_ID.json || echo "M4-real FAILED"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
echo "=== done $(date -Is)"
