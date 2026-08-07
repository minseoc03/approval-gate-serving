#!/usr/bin/env python3
"""
Engine-independent KV transfer microbenchmark (Table 1 inputs).

Measures, at KV-cache-sized granularity:
  - HBM -> CPU (D2H) and CPU -> HBM (H2D) bandwidth, pinned host memory
  - CPU -> NVMe write and NVMe -> CPU read (O_DIRECT, /tmp on local NVMe)

Sizes default to the measured gate-prompt KV range for 70B:
median ~8.4K tokens x 320 KiB/tok ~= 2.6 GB, so 0.5-8 GB brackets it.

Run on a GPU node with .venv-vllm (has torch + CUDA).
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def bench_gpu_cpu(size_bytes, repeats, device):
    import torch

    n = size_bytes // 2  # fp16/bf16 elements
    gpu = torch.empty(n, dtype=torch.bfloat16, device=device)
    cpu = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    torch.cuda.synchronize()

    def timed(fn):
        ts = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return ts

    d2h = timed(lambda: cpu.copy_(gpu, non_blocking=True))
    h2d = timed(lambda: gpu.copy_(cpu, non_blocking=True))
    del gpu, cpu
    return d2h, h2d


def bench_nvme(size_bytes, repeats, path):
    """O_DIRECT write/read of an aligned buffer."""
    align = 4096
    size = size_bytes - size_bytes % align
    buf = bytearray(os.urandom(size))
    mv = memoryview(buf)

    writes, reads = [], []
    for i in range(repeats):
        f = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
        t0 = time.perf_counter()
        written = 0
        while written < size:
            written += os.pwrite(f, mv[written:written + (64 << 20)], written)
        os.fsync(f)
        os.close(f)
        writes.append(time.perf_counter() - t0)

        f = os.open(path, os.O_RDONLY | os.O_DIRECT)
        t0 = time.perf_counter()
        got = 0
        while got < size:
            chunk = os.pread(f, 64 << 20, got)
            if not chunk:
                break
            got += len(chunk)
        os.close(f)
        reads.append(time.perf_counter() - t0)
    os.unlink(path)
    return writes, reads


def summarize(name, size_bytes, times):
    gbps = [size_bytes / t / 1e9 for t in times]
    return dict(op=name, size_gb=size_bytes / 1e9,
                median_s=statistics.median(times),
                gbps_median=statistics.median(gbps),
                gbps_min=min(gbps), gbps_max=max(gbps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-gb", nargs="+", type=float,
                    default=[0.5, 1, 2, 4, 8])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--nvme-path", default="/tmp/kvbench.bin")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--out", default="results/transfer_bench.jsonl")
    args = ap.parse_args()

    rows = []
    for gb in args.sizes_gb:
        size = int(gb * 1e9)
        if not args.skip_gpu:
            d2h, h2d = bench_gpu_cpu(size, args.repeats, args.device)
            rows.append(summarize("hbm_to_cpu", size, d2h))
            rows.append(summarize("cpu_to_hbm", size, h2d))
        w, r = bench_nvme(size, args.repeats, args.nvme_path)
        rows.append(summarize("cpu_to_nvme_write", size, w))
        rows.append(summarize("nvme_to_cpu_read", size, r))
        for row in rows[-4 if not args.skip_gpu else -2:]:
            print(f"{row['op']:18s} {row['size_gb']:5.2f} GB  "
                  f"{row['median_s']*1e3:8.1f} ms  "
                  f"{row['gbps_median']:6.2f} GB/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()