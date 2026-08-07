#!/bin/bash
#SBATCH --job-name=nvl-diag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvl
##SBATCH --account=<your-account>   # default association is used if unset

set -uo pipefail

echo "=== node: $(hostname)  date: $(date -Is) ==="

echo; echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo; echo "=== RAM ==="
free -h

echo; echo "=== block devices ==="
lsblk -d -o NAME,SIZE,ROTA,TYPE,MODEL

echo; echo "=== filesystems ==="
df -h /tmp / /dev/shm 2>/dev/null
echo
df -h | grep -Ei "nvme|ssd|local|scratch|weka" || true

echo; echo "=== /tmp write bandwidth ==="
if command -v fio >/dev/null 2>&1; then
  fio --name=w --rw=write --bs=1M --size=4G --direct=1 --filename=/tmp/fiotest --output-format=terse | awk -F';' '{print "fio write:", $48, "KB/s"}'
  rm -f /tmp/fiotest
else
  dd if=/dev/zero of=/tmp/ddtest bs=1M count=4096 oflag=direct 2>&1 | tail -1
  echo "--- read ---"
  dd if=/tmp/ddtest of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
  rm -f /tmp/ddtest
fi

echo; echo "=== /dev/shm write bandwidth ==="
dd if=/dev/zero of=/dev/shm/ddtest bs=1M count=4096 2>&1 | tail -1
rm -f /dev/shm/ddtest

echo; echo "=== internet check ==="
timeout 5 curl -sI https://huggingface.co >/dev/null 2>&1 && echo "outbound internet: YES" || echo "outbound internet: NO"

echo; echo "=== done ==="
