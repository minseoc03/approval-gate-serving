#!/usr/bin/env python3
"""
Falsification test for the constant-alpha_CPU proposal (CLAUDE.md 9.2 ext).

The proposal: give the DPM a constant alpha_CPU > 0 (a congestion/shadow
price) so t2 is finite, voluntary CPU->discard demotion happens, and the
3-state guarantee holds unconditionally IN THE MODEL.

The test: policies DECIDE using the constant-alpha tier system, but are
SCORED on physical costs (alpha_CPU = 0; DRAM slot freed on departure;
admission layer unchanged). If voluntary demotion lowers aggregate physical
cost at lambda > lambda_crit (by sparing later arrivals forced discards), the
constant is earning its keep; if it never does, alpha_CPU = 0 is empirically
justified for Paper 1.

Principled constant candidate: at load lambda, the marginal displacement rate
per occupied slot-second is ~(lambda - lambda_crit)/C_CPU, so
    alpha_CPU* = (lambda - lambda_crit)/C_CPU * beta_discard
At lambda = 2*lambda_crit: 0.222/400 * 1.91 = 0.00106 GPU-s/s (t2 ~ 30 min).
We sweep constants bracketing it (t2 = 10 min / 30 min / 60 min).
"""

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np

import policy as P
import replay as R


def make_const_alpha(base_fn, ac):
    ts3 = P.TierSystem([R.ALPHA_HBM_B, ac, 0.0],
                       [0.0, R.BETA_CPU_B, R.BETA_DISCARD_B],
                       ["hbm", "cpu", "discard"])
    def fn(ts, wait, pred, rng):
        if len(ts.alphas) == 3:
            return base_fn(ts3, wait, pred, rng)
        return base_fn(ts, wait, pred, rng)   # overflow system unchanged
    return fn


def run(dist, lam_mult, fn, n, seed, tag):
    lam = lam_mult * R.C_CPU / R.W_REF
    nprng = np.random.default_rng(seed)
    waits = R.sample_waits(dist, n, nprng)
    arrivals = np.cumsum(nprng.exponential(1 / lam, n))
    preds = waits.copy()
    B, _, forced, disc = R.run_generic(
        fn, arrivals, waits, preds, random.Random(hash(tag) % 99991))
    return statistics.mean(B), forced / n, disc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lam-mults", nargs="+", type=float,
                    default=[0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--dists", nargs="+", default=["exp", "mixture"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/const_alpha_eval.json")
    args = ap.parse_args()

    beta_step = R.BETA_DISCARD_B - R.BETA_CPU_B
    variants = {
        "ac=0 (physical)": None,
        "ac: t2=10min": beta_step / 600,
        "ac: t2=30min": beta_step / 1800,
        "ac: t2=60min": beta_step / 3600,
    }
    rows = []
    for dist in args.dists:
        print(f"\n=== {dist}: mean PHYSICAL GPU-s/request "
              "(decide with constant alpha, score with alpha_CPU=0) ===")
        print("variant".ljust(18) + "".join(f"λ×{m:<8}" for m in args.lam_mults))
        for name, ac in variants.items():
            fn = (P.online_rand_prudent if ac is None
                  else make_const_alpha(P.online_rand_prudent, ac))
            line = name.ljust(18)
            for lm in args.lam_mults:
                m, ff, df = run(dist, lm, fn, args.n, args.seed,
                                f"{name}{dist}{lm}")
                rows.append(dict(dist=dist, lam_mult=lm, variant=name,
                                 alpha_cpu=ac or 0.0, meanB=m,
                                 forced_frac=ff, discard_frac=df))
                line += f"{m:<10.3f}"
            print(line)
        # context: forced-overflow fraction at each lambda for ac=0
        line = "  (forced_frac)".ljust(18)
        for lm in args.lam_mults:
            c = [r for r in rows if r["dist"] == dist and r["lam_mult"] == lm
                 and r["variant"] == "ac=0 (physical)"]
            line += f"{c[0]['forced_frac']:<10.3f}"
        print(line)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
