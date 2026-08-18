#!/bin/bash
#SBATCH --job-name=bench-m4pin
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=nvl

# M4-pinned (13.1 item 8b): alpha_HBM with ENFORCED residency via the
# BlockPool victim-exclusion patch. Policy control stays harness-side
# through VLLM_PIN_CONTROL; engine events land in VLLM_PIN_EVENTS.

set -uo pipefail
export VLLM_SERVER_DEV_MODE=1
WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
VENV=$WORK/.venv-vllm
PYCLI=$WORK/.venv/bin/python
MODEL=${MODEL:-/scratch/ajoshi34/models/llama-3.1-70b}
NAME=$(basename "$MODEL")
PORT=8305
cd $WORK/harness

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -lt 4 ]; then echo "FATAL: only $NGPU GPUs on $(hostname)"; exit 1; fi
if ! nvidia-smi --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
  echo "FATAL: NVML unhealthy on $(hostname)"; exit 1; fi
# n09-style: nvidia-smi lists 4 but CUDA initializes only 3 — probe the exact
# call that fails (torch get_device_properties on every index)
if ! $VENV/bin/python -c "import torch; assert torch.cuda.device_count() >= 4; [torch.cuda.get_device_properties(i) for i in range(4)]" >/dev/null 2>&1; then
  echo "FATAL: CUDA cannot initialize 4 GPUs on $(hostname)"; exit 1; fi
if ! grep -q "suspendserve pin patch" \
    $VENV/lib/python3.12/site-packages/vllm/v1/core/block_pool.py; then
  echo "FATAL: pin patch not applied to $VENV"; exit 1; fi

export VLLM_PIN_CONTROL=/tmp/pin_ctrl_$SLURM_JOB_ID.json
export VLLM_PIN_EVENTS=$WORK/harness/results/blockpool_events_$SLURM_JOB_ID.jsonl
echo '{"mode":"off","n":0}' > "$VLLM_PIN_CONTROL"

echo "=== node $(hostname) $(date -Is)"
"$VENV/bin/vllm" serve "$MODEL" --served-model-name $NAME \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --port $PORT \
    > logs/server_m4pin_$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 90); do
  curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q llama && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -20 logs/server_m4pin_$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done

$PYCLI bench_occupancy_pinned.py --server localhost:$PORT --model $NAME \
  --ctrl "$VLLM_PIN_CONTROL" --Ns 0 25 50 65 72 --rates 1.5 3.0 \
  --duration 150 --out results/bench_occupancy_pinned_$SLURM_JOB_ID.json \
  || echo "M4-pinned FAILED"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
echo "=== done $(date -Is)"
