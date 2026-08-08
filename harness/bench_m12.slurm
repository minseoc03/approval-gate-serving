#!/bin/bash
#SBATCH --job-name=bench-m12
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvl

# M1 (pinned HBM<->DRAM) + M2 (multi-threaded O_DIRECT NVMe) on one nvl node.

set -uo pipefail
WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
cd $WORK/harness

echo "=== node $(hostname) $(date -Is)"
echo "=== NVMe PCIe link: speed=$(cat /sys/block/nvme0n1/device/device/current_link_speed 2>/dev/null) width=x$(cat /sys/block/nvme0n1/device/device/current_link_width 2>/dev/null) (max $(cat /sys/block/nvme0n1/device/device/max_link_speed 2>/dev/null) x$(cat /sys/block/nvme0n1/device/device/max_link_width 2>/dev/null))"
$WORK/.venv-vllm/bin/python bench_transfer.py \
  --size-gb 2.75 --reps 10 --threads 8 --nvme-dir /tmp \
  --out results/bench_transfer_${SLURM_JOB_ID}.json
echo "=== done $(date -Is)"
