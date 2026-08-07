#!/bin/bash
#SBATCH --job-name=nvme-diag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvl

# Why is effective NVMe read ~0.9 GB/s when the Micron 7450 spec says 6.8?
# Candidates: Slurm cgroup io.max throttle, reduced PCIe link, qd starvation.

set -uo pipefail
WORK=${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/..}
cd $WORK/harness

echo "=== node $(hostname) $(date -Is)"

echo; echo "=== PCIe link status of the NVMe device ==="
NVME_PCI=$(readlink -f /sys/block/nvme0n1/device/device 2>/dev/null | xargs basename 2>/dev/null)
echo "pci addr: ${NVME_PCI:-unknown}"
cat /sys/block/nvme0n1/device/device/current_link_speed 2>/dev/null || true
cat /sys/block/nvme0n1/device/device/current_link_width 2>/dev/null || true
cat /sys/block/nvme0n1/device/device/max_link_speed 2>/dev/null || true
cat /sys/block/nvme0n1/device/device/max_link_width 2>/dev/null || true

echo; echo "=== block queue settings ==="
for f in scheduler nr_requests read_ahead_kb max_sectors_kb; do
  echo "$f: $(cat /sys/block/nvme0n1/queue/$f 2>/dev/null)"
done

echo; echo "=== cgroup I/O limits for this job ==="
CG=$(cat /proc/self/cgroup | head -1 | cut -d: -f3)
echo "cgroup: $CG"
for f in io.max io.bfq.weight io.stat; do
  echo "--- $f:"; cat /sys/fs/cgroup${CG}/$f 2>/dev/null | head -5 || echo "(absent)"
done
# walk up the hierarchy looking for io.max
P=/sys/fs/cgroup${CG}
while [ "$P" != "/sys/fs/cgroup" ] && [ -n "$P" ]; do
  if [ -s "$P/io.max" ]; then echo "io.max at $P:"; cat "$P/io.max"; fi
  P=$(dirname "$P")
done

echo; echo "=== qd sweep: threaded O_DIRECT read, 2.75 GB total ==="
for T in 1 4 8 16 32; do
  $WORK/.venv-vllm/bin/python bench_transfer.py --skip-m1 \
    --size-gb 2.75 --reps 3 --threads $T --nvme-dir /tmp \
    --out results/nvme_diag_t${T}_${SLURM_JOB_ID}.json 2>/dev/null | grep m2_
done

echo; echo "=== single large sequential read via dd (reference) ==="
dd if=/dev/zero of=/tmp/ddref bs=1M count=2816 oflag=direct 2>&1 | tail -1
dd if=/tmp/ddref of=/dev/null bs=16M iflag=direct 2>&1 | tail -1
rm -f /tmp/ddref

echo; echo "=== done $(date -Is)"
