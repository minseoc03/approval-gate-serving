#!/bin/bash
#SBATCH --job-name=vllm-70b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=nvl
##SBATCH --account=<your-account>   # default association is used if unset

# TIER selects the KV placement configuration this server exposes:
#   hbm  (default) : prefix caching only — pin + discard experiments
#   cpu            : + native OffloadingConnector (CPU DRAM tier)
#   nvme           : + LMCache with local-disk backend (NVMe tier)
TIER=${TIER:-hbm}

set -euo pipefail
export VLLM_SERVER_DEV_MODE=1

WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
VENV=$WORK/.venv-vllm
MODEL=${MODEL:?set MODEL to a local HF checkpoint path}
TP=${TP:-4}
PORT=${PORT:-8300}
MAX_LEN=${MAX_LEN:-32768}
MODEL_NAME=$(basename "$MODEL")

# ~60 GB of CPU offload space unless overridden (block size 16 tokens;
# 70B KV is 320 KiB/token -> 1 block = 5 MiB; 12000 blocks ≈ 60 GiB)
CPU_OFFLOAD_BYTES=${CPU_OFFLOAD_BYTES:-64000000000}

EXTRA_ARGS=()
case "$TIER" in
  hbm) ;;
  cpu)
    EXTRA_ARGS+=(--kv-transfer-config
      "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${CPU_OFFLOAD_BYTES}}}")
    ;;
  nvme)
    export LMCACHE_CONFIG_FILE=$WORK/harness/lmcache_nvme.yaml
    EXTRA_ARGS+=(--kv-transfer-config
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
    ;;
  *) echo "unknown TIER=$TIER" >&2; exit 1 ;;
esac

echo "[serve] node=$(hostname) model=$MODEL_NAME tp=$TP tier=$TIER port=$PORT started=$(date -Is)"
echo "$(hostname):$PORT $MODEL_NAME tier=$TIER" > $WORK/harness/server_info_${MODEL_NAME}.txt

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.92 \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"