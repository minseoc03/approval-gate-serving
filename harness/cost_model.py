#!/usr/bin/env python3
"""
Cost model: ingest M1-M4 measurements -> alpha/beta table -> break-evens ->
surviving tier set, under BOTH cost currencies (CLAUDE.md 10.1, 14).

Currencies:
  A. resume-stall seconds   -- what prior systems minimize (wake latency)
  B. stolen GPU-seconds     -- our objective: GPU compute taken from other
                               requests (transfers via DMA steal ~none)

Until M3 lands, beta_discard uses an estimate range (suffix prefill wall
seconds at 70B TP4); pass --m3 <json> to switch to measured values.

Pure arithmetic; no GPU.
"""

import argparse
import glob
import json
import statistics

GB = 1e9

# Platform constants (measured; CLAUDE.md 12, 13, 16)
GATE_BYTES = 2.75 * GB          # median gate context KV, 70B
HBM_KV_BYTES = 207.6 * GB       # 51.9 GiB x 4 GPUs, job 2017796
DRAM_BYTES = 1.1e12             # available CPU RAM for offload
N_GPUS = 4
WAIT_S = 1800.0                 # reference human wait (30 min)


def load_transfer(patterns):
    ops = {}
    for f in sorted(glob.glob(patterns)):
        d = json.load(open(f))
        for r in d["results"]:
            ops.setdefault(r["op"], []).append(r["median_s"])
    return {k: statistics.median(v) for k, v in ops.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer-glob", default="results/bench_transfer_*.json")
    ap.add_argument("--m3", default=None, help="bench_prefill JSON (measured)")
    ap.add_argument("--m3-est", nargs=2, type=float, default=[0.2, 0.6],
                    metavar=("LO", "HI"),
                    help="suffix re-prefill wall-s estimate range until M3")
    args = ap.parse_args()

    t = load_transfer(args.transfer_glob)
    h2d = t["m1_cpu_to_hbm_pinned"]            # CPU tier wake path
    d2h = t["m1_hbm_to_cpu_pinned"]            # CPU tier demote path
    nvme_rd = t["m2_nvme_to_cpu_read"]         # NVMe wake: disk -> DRAM
    nvme_wr = t["m2_cpu_to_nvme_write"]

    if args.m3:
        m3 = json.load(open(args.m3))["summary"]
        pf = [m3["ttft_suffix_s"]["median"]] * 2
        m3_src = f"measured ({args.m3})"
    else:
        pf = list(args.m3_est)
        m3_src = f"ESTIMATE {pf} (M3 pending)"

    # alpha: holding cost per second
    #   currency B: HBM occupancy steals a capacity share of node goodput
    alpha_hbm_B = GATE_BYTES / HBM_KV_BYTES * N_GPUS   # GPU-s stolen per s
    dram_cap = int(DRAM_BYTES / GATE_BYTES)            # CPU tier capacity

    # beta: wake-up cost, per currency  (lo, hi where ranged)
    beta = {
        # tier:      A resume-stall (s),        B stolen GPU-s
        "hbm":     ((0.0, 0.0),                 (0.0, 0.0)),
        "cpu":     ((h2d, h2d),                 (0.0, 0.0)),        # DMA
        "nvme":    ((nvme_rd + h2d,) * 2,       (0.0, 0.0)),        # DMA
        "discard": (tuple(pf),                  tuple(x * N_GPUS for x in pf)),
    }

    print(f"M3 source: {m3_src}\n")
    print("measured transfer medians (2.75 GB): "
          f"D2H {d2h:.3f}s  H2D {h2d:.3f}s  NVMe wr {nvme_wr:.2f}s rd {nvme_rd:.2f}s")
    print(f"alpha_HBM (currency B): {alpha_hbm_B:.4f} GPU-s stolen per second "
          f"-> {alpha_hbm_B*WAIT_S:.0f} GPU-s over a {WAIT_S:.0f}s wait")
    print(f"CPU tier capacity: ~{dram_cap} gate contexts "
          f"(DRAM {DRAM_BYTES/1e12:.1f} TB / {GATE_BYTES/GB:.2f} GB)\n")

    for cur, idx, unit in (("A: resume-stall", 0, "s"),
                           ("B: stolen GPU-seconds", 1, "GPU-s")):
        print(f"=== currency {cur} ===")
        for tier, b in beta.items():
            lo, hi = b[idx]
            rng = f"{lo:.3f}" if lo == hi else f"{lo:.3f}-{hi:.3f}"
            print(f"  beta_{tier:8s} {rng} {unit}")
        # dominance among alpha==0 tiers: smaller beta wins outright
        z = {k: beta[k][idx] for k in ("nvme", "discard")}
        nv_lo, dis_hi = z["nvme"][0], z["discard"][1]
        if nv_lo > dis_hi:
            print("  -> NVMe DOMINATED by discard (alpha both ~0, "
                  f"beta {nv_lo:.2f} > {dis_hi:.2f}) : NVMe drops out")
        cpu_hi, dis_lo = beta["cpu"][idx][1], beta["discard"][idx][0]
        if cpu_hi < dis_lo:
            print("  -> discard dominated by CPU on the UNCAPACITATED "
                  f"envelope (beta {cpu_hi:.3f} < {dis_lo:.3f}) — but CPU "
                  f"capacity is finite ({dram_cap} contexts):")
            lam = dram_cap / WAIT_S * 60
            print(f"     Little's law: DRAM saturates at lambda >= "
                  f"{lam:.1f} escalations/min at {WAIT_S:.0f}s waits; "
                  "beyond that, discard is the overflow tier -> hierarchy is "
                  "{HBM, CPU(capacitated), discard(overflow)}")
        # HBM vs CPU break-even (currency B: alpha_hbm vs ~0, beta 0 vs cpu)
        if idx == 1:
            t1 = beta["cpu"][0][0] and beta["cpu"][0][0]  # wake cost in A
            # In currency B the CPU wake is ~free; the HBM->CPU decision is
            # governed by alpha_hbm_B vs the demote+wake DMA (~0):
            print(f"  -> HBM->CPU break-even ~= (d2h+h2d)/alpha = "
                  f"{(d2h+h2d)/alpha_hbm_B:.1f} s of wait "
                  "(charging the DMA round-trip as currency-A seconds)")
        print()

    print("Aug-10 gate verdict (CLAUDE.md 14): NVMe eliminated on this "
          "platform (PCIe Gen3 x1). Hierarchy survives as 3 states via the "
          "CPU capacity constraint, not via the uncapacitated envelope. "
          "Verdict is beta-driven and platform-conditional: NVMe re-enters "
          f"if effective read bandwidth > {GATE_BYTES/GB / max(pf):.1f} GB/s "
          "(i.e., beta_NVMe < beta_discard).")


if __name__ == "__main__":
    main()
