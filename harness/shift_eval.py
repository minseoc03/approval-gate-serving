#!/usr/bin/env python3
"""
Distribution-shift evaluation (CLAUDE.md 9 baselines, 16 open item).

The tuned baseline is a FIXED TTL tau (pin until tau, then discard) — the
Continuum-style decision rule, and the policy family that actually varies
with the wait distribution (tier choice does not: always-CPU wins whenever
DRAM is free, on every distribution). tau = 0 is always-discard, tau = inf
is always-pin.

Protocol:
  1. For each TRAIN distribution: grid-search tau on a training run
     (same simulator, lambda x2 where overflow makes the decision live).
  2. Evaluate tau*_train on every EVAL distribution.
  3. Compare against online_rand (prediction-FREE randomized DPM — needs no
     predictor at all) and per-distribution OPT.

Claim under test: adaptation does not beat a well-tuned fixed policy on the
distribution it was tuned for; it wins when the distribution MOVES — which
is what the worst-case guarantee buys.

Distributions include a reviewer-regime pair that makes shift realistic:
  active  : 0.8*Exp(20 s) + 0.2*Exp(600 s)   (reviewer at desk)
  offline : lognormal median 3600 s           (overnight queue)
"""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

import policy as P
import replay as R


def sample_waits(dist, n, rng):
    if dist == "active":
        short = rng.exponential(20.0, n)
        med = rng.exponential(600.0, n)
        return np.where(rng.random(n) < 0.8, short, med)
    if dist == "burst":
        # quick rubber-stamp approvals + an overnight tail heavy enough to
        # saturate DRAM: the long mode creates the overflow pressure, the
        # short mode then faces the live {HBM, discard} decision
        short = rng.exponential(20.0, n)
        long_ = np.exp(rng.normal(math.log(3600.0), 0.7, n))
        return np.where(rng.random(n) < 0.6, short, long_)
    if dist == "offline":
        return np.exp(rng.normal(math.log(3600.0), 0.7, n))
    return R.sample_waits(dist, n, rng)


def make_ttl(tau):
    def fn(ts, wait, pred, rng):
        if len(ts.alphas) == 3:
            return [0.0]                    # CPU free -> offload immediately
        return [] if tau == math.inf else [tau]   # overflow: pin-until-tau
    return fn


def mean_cost(fn, dist, n, lam_mult, seed, rng_name):
    lam = lam_mult * R.C_CPU / R.W_REF
    nprng = np.random.default_rng(seed)
    waits = sample_waits(dist, n, nprng)
    arrivals = np.cumsum(nprng.exponential(1 / lam, n))
    preds = waits.copy()                    # unused by ttl/online_rand/opt
    B, _, _, _ = R.run_generic(fn, arrivals, waits, preds,
                               random.Random(hash(rng_name) % 99991))
    return statistics.mean(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lam-mult", type=float, default=2.0)
    ap.add_argument("--dists", nargs="+",
                    default=["active", "exp", "lognorm", "mixture", "offline"])
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0, 5, 10, 20, 36, 60, 120, 300, 900, math.inf])
    ap.add_argument("--train-seed", type=int, default=1)
    ap.add_argument("--eval-seed", type=int, default=2)
    ap.add_argument("--out", default="results/shift_eval.json")
    args = ap.parse_args()

    # 1. tune tau per training distribution
    tuned = {}
    for d in args.dists:
        costs = {tau: mean_cost(make_ttl(tau), d, args.n, args.lam_mult,
                                args.train_seed, f"ttl{tau}")
                 for tau in args.taus}
        tuned[d] = min(costs, key=costs.get)
        print(f"train {d:8s}: tau* = {tuned[d]}  "
              f"(cost {costs[tuned[d]]:.3f})")

    # 2. evaluate everything on every eval distribution
    rows = []
    print("\n=== eval matrix: mean GPU-s stolen/request "
          f"(lambda x{args.lam_mult}) ===")
    hdr = "policy \\ eval".ljust(22) + "".join(f"{d:<10}" for d in args.dists)
    print(hdr)
    base = {}
    for d in args.dists:
        base[(d, "opt")] = mean_cost(P.opt, d, args.n, args.lam_mult,
                                     args.eval_seed, "opt")
        base[(d, "online_rand")] = mean_cost(
            P.online_rand_prudent, d, args.n, args.lam_mult,
            args.eval_seed, "onr")
    for name, key in (("OPT (oracle)", "opt"),
                      ("online_rand (no pred)", "online_rand")):
        line = name.ljust(22)
        for d in args.dists:
            line += f"{base[(d, key)]:<10.3f}"
        print(line)
    for train in args.dists:
        fn = make_ttl(tuned[train])
        line = f"ttl*({train})={tuned[train]:<6}".ljust(22)
        for ev in args.dists:
            c = mean_cost(fn, ev, args.n, args.lam_mult,
                          args.eval_seed, f"ttl{train}{ev}")
            rows.append(dict(train=train, eval=ev, tau=tuned[train], meanB=c))
            line += f"{c:<10.3f}"
        print(line)

    # 3. headline: worst-case ratio to OPT, on-diagonal vs off-diagonal
    print("\n=== ratio to per-distribution OPT ===")
    diag, off, onr_r = [], [], []
    for r in rows:
        ratio = r["meanB"] / base[(r["eval"], "opt")]
        (diag if r["train"] == r["eval"] else off).append(ratio)
    for d in args.dists:
        onr_r.append(base[(d, "online_rand")] / base[(d, "opt")])
    print(f"tuned TTL on its own distribution : median {statistics.median(diag):.2f}x  worst {max(diag):.2f}x")
    print(f"tuned TTL under SHIFT             : median {statistics.median(off):.2f}x  worst {max(off):.2f}x")
    print(f"online_rand (no tuning, no pred)  : median {statistics.median(onr_r):.2f}x  worst {max(onr_r):.2f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        tuned={k: (None if v == math.inf else v) for k, v in tuned.items()},
        rows=rows,
        baselines={f"{d}/{k}": v for (d, k), v in base.items()}), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
