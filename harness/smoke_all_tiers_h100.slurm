#!/bin/bash
#SBATCH --job-name=sr-smoke-8b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
##SBATCH --account=<your-account>   # default association is used if unset

# Self-contained 4-tier suspend/resume demonstration (the Aug 5-7 gate).
# Runs entirely inside one job: for each tier config, start an 8B vLLM
# server, wait for health, run suspend_resume.py against it, shut it down.
# No login-node watcher involved.

set -uo pipefail
export VLLM_SERVER_DEV_MODE=1

WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
VENV=$WORK/.venv-vllm
PYCLI=$WORK/.venv/bin/python          # login-side venv works on nvl too (httpx only)
MODEL=${MODEL:?set MODEL to a local HF checkpoint path}
NAME=llama-3.1-8b
PORT=8301
cd $WORK/harness

start_server () {  # $1 = tier
  local tier=$1
  EXTRA=()
  case "$tier" in
    hbm) ;;
    cpu)
      EXTRA+=(--kv-transfer-config
        '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":64000000000}}')
      ;;
    nvme)
      export LMCACHE_CONFIG_FILE=$WORK/harness/lmcache_nvme.yaml
      mkdir -p /tmp/lmcache_nvme
      EXTRA+=(--kv-transfer-config
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
      ;;
  esac
  echo "=== [$(date -Is)] starting server tier=$tier"
  "$VENV/bin/vllm" serve "$MODEL" --served-model-name $NAME \
      --tensor-parallel-size 1 --max-model-len 32768 \
      --gpu-memory-utilization ${GPU_UTIL:-0.92} --port $PORT "${EXTRA[@]}" \
      > logs/server_${tier}_$SLURM_JOB_ID.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 60); do
    if curl -s --max-time 2 "http://localhost:$PORT/v1/models" | grep -q llama; then
      echo "=== server tier=$tier up after ${i}0s"
      return 0
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "server tier=$tier died; log tail:"; tail -30 logs/server_${tier}_$SLURM_JOB_ID.log; return 1; }
    sleep 10
  done
  echo "server tier=$tier failed to become healthy"; return 1
}

stop_server () {
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
  unset LMCACHE_CONFIG_FILE
  sleep 5
}

run_tier_set () {  # $1 = server tier, rest = suspend_resume tiers
  local tier=$1; shift
  if start_server "$tier"; then
    "$PYCLI" suspend_resume.py --server "localhost:$PORT" --model $NAME \
        --tiers "$@" --n-prompts 2 --repeats 3 \
        --out results/suspend_resume_smoke8b_${SLURM_JOB_ID}.jsonl \
      && echo "=== tier=$tier smoke OK" || echo "=== tier=$tier smoke FAILED"
  fi
  stop_server
}


run_tier_set hbm  pin discard
run_tier_set cpu  pin cpu discard
run_tier_set nvme pin nvme discard

echo "=== [$(date -Is)] all done"
