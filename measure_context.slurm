#!/bin/bash
#SBATCH --job-name=tau2-ctx
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# --- fill these in for your cluster ---
##SBATCH --partition=cpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

# ---------------------------------------------------------------- config
WORK=${WORK:-$HOME/suspendserve}
REPO=$WORK/tau2-bench
VENV=$WORK/.venv
TOKENIZER=${TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}
# ungated mirror if you don't have the gated repo yet:
# TOKENIZER=NousResearch/Meta-Llama-3.1-8B-Instruct

# Keep HF cache off the home quota and on the parallel FS
export HF_HOME=${HF_HOME:-/weka/scratch/$USER/hf}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$WORK" "$HF_HOME"

# ---------------------------------------------------------------- setup
if [ ! -d "$REPO" ]; then
  echo "[setup] cloning tau2-bench"
  git clone --depth 1 https://github.com/sierra-research/tau2-bench.git "$REPO"
fi

if [ ! -d "$VENV" ]; then
  echo "[setup] creating venv"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet transformers
fi

# ---------------------------------------------------------------- run
echo "[run] node=$(hostname)  started=$(date -Is)"

"$VENV/bin/python" "$WORK/measure_tau2_context.py" \
  --repo "$REPO" \
  --domains retail airline telecom \
  --tokenizer "$TOKENIZER" \
  --suffix-tokens 4000 \
  --hbm-gb 80 --weights-gb 16 \
  --out "$WORK/context_report.json"

echo "[run] finished=$(date -Is)"
