#!/usr/bin/env python3
"""
Policy evaluation by replay over the composed cost model (CLAUDE.md 13 #6). v2

Event-level simulation, CPU only, no serving system.

v2 changes (fairness fixes):
  * The admission layer (CLAUDE.md 9.2) applies to EVERY policy: HBM holds at
    most C_HBM suspended contexts and DRAM at most C_CPU. A request that
    cannot hold its target tier is demoted further, whoever's policy it is.
    (v1 enforced HBM capacity only for the MORI baseline, which priced
    always_pin as if HBM were infinite — direction-reversing unfairness.)
  * MORI baseline is eviction-based: it keeps suspended contexts in HBM and
    demotes under capacity pressure — continuously, as real MORI does. At
    human waits its idleness ratio saturates (iota -> 1 for every suspended
    request), so the demotion ORDER is uninformative; we use FIFO among ties.
    Each context therefore resides in HBM for ~C_HBM/lambda before demotion.

Physical note: every request's KV is IN HBM at the gate (it was just
decoding). "HBM full" for the suspended set means the marginal suspended
context cannot linger: it is demoted at t=0 to CPU if there is space, else
discarded. No policy can hold what capacity does not allow.

Currencies (CLAUDE.md 6.1): B = GPU-seconds stolen from others (ours),
A = the request's own resume stall (prior work's). Both reported per row.
"""

import argparse
import heapq
import json
import math
import random
import statistics
from collections import deque
from pathlib import Path

import numpy as np

import policy as P

# Measured constants (CLAUDE.md 14 final verdict)
ALPHA_HBM_B = 0.0530        # GPU-s stolen per second pinned (M4)
BETA_DISCARD_B = 1.91       # GPU-s, suffix re-prefill (M3)
BETA_CPU_B = 0.02           # GPU-s, DMA overhead bound
BETA_CPU_A = 0.048          # s, CPU->HBM reload wall (M1)
BETA_DISCARD_A = 0.479      # s, suffix re-prefill wall (M3)
C_HBM = 75
C_CPU = 400
W_REF = 1800.0

TS_FULL = P.TierSystem([ALPHA_HBM_B, 0.0, 0.0],
                       [0.0, BETA_CPU_B, BETA_DISCARD_B],
                       ["hbm", "cpu", "discard"])
TS_HBM_DISCARD = P.TierSystem([ALPHA_HBM_B, 0.0],
                              [0.0, BETA_DISCARD_B],
                              ["hbm", "discard"])

ALPHA_B = {"hbm": ALPHA_HBM_B, "cpu": 0.0, "discard": 0.0}
BETA_B = {"hbm": 0.0, "cpu": BETA_CPU_B, "discard": BETA_DISCARD_B}
BETA_A = {"hbm": 0.0, "cpu": BETA_CPU_A, "discard": BETA_DISCARD_A}


def price(segments, end_tier):
    """segments: [(tier, duration)]"""
    b = sum(ALPHA_B[t] * d for t, d in segments) + BETA_B[end_tier]
    a = BETA_A[end_tier]
    return b, a


def sample_waits(dist, n, rng):
    if dist == "exp":
        return rng.exponential(W_REF, n)
    if dist == "lognorm":
        return np.exp(rng.normal(math.log(W_REF), 1.0, n))
    if dist == "mixture":
        short = rng.exponential(60.0, n)
        long_ = np.exp(rng.normal(math.log(2 * W_REF), 0.7, n))
        return np.where(rng.random(n) < 0.5, short, long_)
    raise ValueError(dist)


class Pool:
    """Capacity pool tracked as a departure-time heap."""
    def __init__(self, cap):
        self.cap, self.h = cap, []

    def tick(self, now):
        while self.h and self.h[0] <= now:
            heapq.heappop(self.h)

    def free(self):
        return len(self.h) < self.cap

    def hold(self, until):
        heapq.heappush(self.h, until)


def run_generic(fn, arrivals, waits, preds, rng):
    """Any policy.py-style policy under the common admission layer."""
    hbm, cpu = Pool(C_HBM), Pool(C_CPU)
    rowsB, rowsA, forced, discards = [], [], 0, 0
    for t0, wait, pred in zip(arrivals, waits, preds):
        hbm.tick(t0)
        cpu.tick(t0)
        if hbm.free():
            ts = TS_FULL if cpu.free() else TS_HBM_DISCARD
            switches = fn(ts, wait, pred, rng)
            # trajectory -> segments under this ts
            segs, cur, last = [], 0, 0.0
            for s in switches:
                if s >= wait:
                    break
                segs.append((ts.names[cur], s - last))
                cur, last = cur + 1, s
            segs.append((ts.names[cur], wait - last))
            end = ts.names[cur]
        else:
            forced += 1
            if cpu.free():
                segs, end = [("cpu", wait)], "cpu"
            else:
                segs, end = [("discard", wait)], "discard"
        t_acc = 0.0
        for t, d in segs:
            if t == "hbm" and d > 0:
                hbm.hold(t0 + t_acc + d)
            if t == "cpu" and d > 0:
                # slot freed when the trajectory LEAVES the tier — matters for
                # voluntary CPU->discard demotion (constant-alpha variants)
                cpu.hold(t0 + t_acc + d)
            t_acc += d
        if end == "discard":
            discards += 1
        b, a = price(segs, end)
        rowsB.append(b)
        rowsA.append(a)
    return rowsB, rowsA, forced, discards


def run_mori(arrivals, waits, rng):
    """Eviction-based MORI emulation. Suspended contexts stay in HBM; under
    capacity pressure the demotion ranking (idleness) is saturated at human
    waits, so eviction order is FIFO among ties. Evicted contexts go to CPU
    if there is space, else are discarded."""
    cpu = Pool(C_CPU)
    hbm = deque()               # (req_idx, entered_at)
    info = {}                   # req_idx -> dict(t0, wait, segs, end)
    rowsB, rowsA, discards = [], [], 0

    def resolve(idx, now, to_tier):
        d = info[idx]
        hbm_dur = now - d["t0"]
        segs = [("hbm", min(hbm_dur, d["wait"]))]
        rest = d["wait"] - segs[0][1]
        if rest <= 0:            # resumed while still pinned
            end = "hbm"
        elif to_tier == "cpu":
            segs.append(("cpu", rest))
            end = "cpu"
            cpu.hold(d["t0"] + d["wait"])
        else:
            segs.append(("discard", rest))
            end = "discard"
        d["segs"], d["end"] = segs, end

    for i, (t0, wait) in enumerate(zip(arrivals, waits)):
        cpu.tick(t0)
        # resumes: contexts whose wait ended leave HBM untouched
        while hbm and info[hbm[0][0]]["t0"] + info[hbm[0][0]]["wait"] <= t0:
            idx, _ = hbm.popleft()
            resolve(idx, info[idx]["t0"] + info[idx]["wait"], "hbm")
        info[i] = dict(t0=t0, wait=wait)
        hbm.append((i, t0))
        while len(hbm) > C_HBM:  # capacity pressure -> demote FIFO
            idx, _ = hbm.popleft()
            resolve(idx, t0, "cpu" if cpu.free() else "discard")
    for idx, _ in hbm:           # drain remaining at end of sim
        resolve(idx, info[idx]["t0"] + info[idx]["wait"], "hbm")

    for i in sorted(info):
        d = info[i]
        b, a = price(d["segs"], d["end"])
        rowsB.append(b)
        rowsA.append(a)
        if d["end"] == "discard":
            discards += 1
    return rowsB, rowsA, 0, discards


def fixed_policies():
    def pin(ts, w, p, rng):
        return []                       # stay in HBM as long as allowed

    def cpu_(ts, w, p, rng):
        return [0.0]                    # demote immediately to next tier

    def discard_(ts, w, p, rng):
        return [0.0] * (len(ts.alphas) - 1)
    return {"always_pin": pin, "always_cpu": cpu_, "always_discard": discard_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lam-mults", nargs="+", type=float,
                    default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--dists", nargs="+", default=["exp", "lognorm", "mixture"])
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/replay.json")
    args = ap.parse_args()

    lam_crit = C_CPU / W_REF
    rows = []
    for dist in args.dists:
        for lm in args.lam_mults:
            lam = lm * lam_crit
            nprng = np.random.default_rng(args.seed)
            waits = sample_waits(dist, args.n, nprng)
            arrivals = np.cumsum(nprng.exponential(1 / lam, args.n))
            # decision-liveness: probability mass below the pin/discard
            # break-even t* — adaptation can only pay when this is not ~0/~1
            t_star = BETA_DISCARD_B / ALPHA_HBM_B
            mass_below = float(np.mean(waits < t_star))
            for sm in args.sigmas:
                sig = sm * W_REF
                preds = np.maximum(0.0, waits + nprng.normal(0, sig, args.n)) \
                    if sig > 0 else waits.copy()
                eta = ALPHA_HBM_B * float(np.mean(np.abs(preds - waits)))

                pols = dict(fixed_policies())
                pols.update(P.make_policies())
                for name, fn in pols.items():
                    pred_free = name in ("always_pin", "always_cpu",
                                         "always_discard", "opt",
                                         "online_det", "online_rand")
                    if pred_free and sm != args.sigmas[0]:
                        continue
                    rng = random.Random(args.seed + hash(name) % 9973)
                    B, A, forced, disc = run_generic(
                        fn, arrivals, waits, preds, rng)
                    n = len(B)
                    rows.append(dict(
                        dist=dist, lam_mult=lm, sigma_mult=sm, eta_mean=eta, mass_below_be=mass_below,
                        policy=name, n=n,
                        meanB=statistics.mean(B),
                        p95B=sorted(B)[int(0.95 * n)],
                        meanA=statistics.mean(A),
                        forced_frac=forced / n, discard_frac=disc / n))
                if sm == args.sigmas[0]:
                    B, A, forced, disc = run_mori(arrivals, waits,
                                                  random.Random(args.seed))
                    n = len(B)
                    rows.append(dict(
                        dist=dist, lam_mult=lm, sigma_mult=sm, eta_mean=0.0, mass_below_be=mass_below,
                        policy="mori", n=n,
                        meanB=statistics.mean(B),
                        p95B=sorted(B)[int(0.95 * n)],
                        meanA=statistics.mean(A),
                        forced_frac=0.0, discard_frac=disc / n))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))

    print(f"lambda_crit = {lam_crit*60:.1f}/min;  n = {args.n};  "
          f"capacity C_HBM={C_HBM} C_CPU={C_CPU} enforced for ALL policies")
    for dist in args.dists:
        print(f"\n=== {dist} — mean GPU-s stolen/request (currency B, "
              f"sigma={args.sigmas[0]}) ===")
        pnames = sorted({r['policy'] for r in rows})
        print("policy".ljust(16) +
              "".join(f"λ×{m:<7}" for m in args.lam_mults))
        for pn in pnames:
            line = pn.ljust(16)
            for lm in args.lam_mults:
                c = [r for r in rows if r["dist"] == dist
                     and r["lam_mult"] == lm and r["policy"] == pn
                     and r["sigma_mult"] == args.sigmas[0]]
                line += f"{c[0]['meanB']:<9.2f}" if c else " " * 9
            print(line)
        # eta curve for the LA policy
        la = sorted({r["sigma_mult"] for r in rows})
        print("\n  eta curve (rhomu_1.1596 meanB by sigma_mult, λ×%s):"
              % args.lam_mults[-1])
        for sm in la:
            c = [r for r in rows if r["dist"] == dist
                 and r["lam_mult"] == args.lam_mults[-1]
                 and r["policy"] == "rhomu_1.1596" and r["sigma_mult"] == sm]
            if c:
                print(f"    sigma={sm:<5} eta={c[0]['eta_mean']:<8.3f} "
                      f"meanB={c[0]['meanB']:.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
