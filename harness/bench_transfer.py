#!/usr/bin/env python3
"""
M1 + M2 transfer microbenchmarks (CLAUDE.md 10.1).

M1  beta_CPU / P_CPU : HBM<->DRAM cudaMemcpy, pinned host memory, both
    directions, gate-context-sized payload (default 2.75 GB = median 70B
    gate KV), >=10 reps, median + IQR.
M2  beta_NVMe / P_NVMe : NVMe read/write with O_DIRECT, multi-threaded
    (fio is not installed on compute nodes; threads on separate file
    regions serve as the queue-depth knob). Buffers are mmap-allocated so
    they satisfy O_DIRECT alignment.

Output: one JSON file per run in results/, consumable by cost_model.py.
"""

import argparse
import json
import mmap
import os
import statistics
import threading
import time
from pathlib import Path

CHUNK = 64 << 20  # 64 MiB


def iqr(xs):
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def summarize(op, size_bytes, times, extra=None):
    gbps = [size_bytes / t / 1e9 for t in times]
    lo, hi = iqr(times) if len(times) >= 4 else (min(times), max(times))
    row = dict(op=op, size_gb=round(size_bytes / 1e9, 3), reps=len(times),
               median_s=statistics.median(times), iqr_s=[lo, hi],
               gbps_median=statistics.median(gbps),
               gbps_iqr=[size_bytes / hi / 1e9, size_bytes / lo / 1e9])
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------- M1
def bench_m1(size_bytes, reps, device):
    import torch

    n = size_bytes // 2
    gpu = torch.empty(n, dtype=torch.bfloat16, device=device)
    host = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    torch.cuda.synchronize()

    def timed(fn):
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return ts

    d2h = timed(lambda: host.copy_(gpu, non_blocking=True))
    h2d = timed(lambda: gpu.copy_(host, non_blocking=True))
    del gpu, host
    torch.cuda.empty_cache()
    return [summarize("m1_hbm_to_cpu_pinned", size_bytes, d2h),
            summarize("m1_cpu_to_hbm_pinned", size_bytes, h2d)]


# ---------------------------------------------------------------- M2
def _worker_write(path, total, out_times):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
    buf = mmap.mmap(-1, CHUNK)          # page-aligned -> O_DIRECT safe
    buf.write(os.urandom(1 << 20) * (CHUNK >> 20))
    mv = memoryview(buf)
    t0 = time.perf_counter()
    off = 0
    while off < total:
        n = min(CHUNK, total - off)
        os.pwritev(fd, [mv[:n]], off)
        off += n
    os.fsync(fd)
    out_times.append(time.perf_counter() - t0)
    os.close(fd)


def _worker_read(path, total, out_times):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, CHUNK)
    mv = memoryview(buf)
    t0 = time.perf_counter()
    off = 0
    while off < total:
        n = min(CHUNK, total - off)
        got = os.preadv(fd, [mv[:n]], off)
        if got <= 0:
            break
        off += got
    out_times.append(time.perf_counter() - t0)
    os.close(fd)


def bench_m2(size_bytes, reps, nthreads, dirpath):
    per_thread = (size_bytes // nthreads) // 4096 * 4096
    paths = [os.path.join(dirpath, f"kvbench_{i}.bin") for i in range(nthreads)]
    writes, reads = [], []
    for _ in range(reps):
        for phase, worker, acc in (("w", _worker_write, writes),
                                   ("r", _worker_read, reads)):
            times = []
            threads = [threading.Thread(target=worker, args=(p, per_thread, times))
                       for p in paths]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            acc.append(time.perf_counter() - t0)
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    total = per_thread * nthreads
    extra = dict(threads=nthreads, o_direct=True,
                 note="python threaded O_DIRECT; fio absent on compute nodes")
    return [summarize("m2_cpu_to_nvme_write", total, writes, extra),
            summarize("m2_nvme_to_cpu_read", total, reads, extra)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gb", type=float, default=2.75,
                    help="payload per rep; default = median 70B gate KV")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8, help="M2 parallelism")
    ap.add_argument("--nvme-dir", default="/tmp")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-m1", action="store_true")
    ap.add_argument("--skip-m2", action="store_true")
    ap.add_argument("--out", default="results/bench_transfer.json")
    args = ap.parse_args()

    size = int(args.size_gb * 1e9)
    rows, meta = [], dict(host=os.uname().nodename, size_gb=args.size_gb,
                          reps=args.reps, timestamp=time.strftime("%FT%T"))
    if not args.skip_m1:
        rows += bench_m1(size, args.reps, args.device)
    if not args.skip_m2:
        rows += bench_m2(size, args.reps, args.threads, args.nvme_dir)

    for r in rows:
        print(f"{r['op']:24s} {r['size_gb']:5.2f} GB  "
              f"{r['median_s']*1e3:8.1f} ms  {r['gbps_median']:6.2f} GB/s  "
              f"IQR [{r['gbps_iqr'][0]:.2f}, {r['gbps_iqr'][1]:.2f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, results=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
