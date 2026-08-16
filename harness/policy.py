#!/usr/bin/env python3
"""
Multi-state DPM placement policies, parameterized by measured alpha/beta.

Adapted from the reference implementation by Antoniadis, Coester, Elias,
Polak, Simon (github.com/adampolak/dpm; arXiv:2110.13116) — their code uses
module-global POWER_CONSUMPTIONS / WAKE_UP_COSTS; this adaptation makes the
tier system a parameter and keeps only what the paper needs:

  opt / ftp                  : offline optimum, follow-the-prediction
  online_det                 : deterministic break-even (prediction-free, <=4)
  online_rand_prudent        : randomized e/(e-1), prediction-free, PRUDENT
  rhomu_prudent              : learning-augmented (rho, mu), PRUDENT
                               (Antoniadis et al. Alg. of Sec. 2.1 + Sec. 4.1
                               prudence conversion — mandatory per their Fig 9)
  cost_of                    : cost of a switch history for a realized wait

Units are OURS: alpha_j in GPU-seconds stolen per second of wait, beta_j in
GPU-seconds at wake (currency B, CLAUDE.md 6.1/9). All algorithms are
scale-free — the reference's per-step rescaling (scale_pred = pred * a/b)
carries any units.

States must satisfy alpha_0 > ... > alpha_k >= 0, beta_0 < ... < beta_k,
beta_0 = 0. Degenerate steps (alpha step == 0) mean the deeper state is never
entered via the envelope — the admission layer (replay.py) handles capacity-
forced tier sets instead (CLAUDE.md 9.2).
"""

import math
import random

import numpy as np
from scipy.special import lambertw

E_RATIO = math.e / (math.e - 1)          # 1.582, RANDOMIZED ratio


class TierSystem:
    def __init__(self, alphas, betas, names=None):
        assert len(alphas) == len(betas)
        assert betas[0] == 0
        self.alphas = list(map(float, alphas))
        self.betas = list(map(float, betas))
        self.names = names or [f"s{i}" for i in range(len(alphas))]
        # per-transition (rent-rate, buy) pairs; alpha step may be 0
        self.step_a = [self.alphas[i] - self.alphas[i + 1]
                       for i in range(len(alphas) - 1)]
        self.step_b = [self.betas[i + 1] - self.betas[i]
                       for i in range(len(alphas) - 1)]

    def opt_state(self, wait):
        costs = [a * wait + b for a, b in zip(self.alphas, self.betas)]
        return min(range(len(costs)), key=costs.__getitem__)

    def opt_cost(self, wait):
        return min(a * wait + b for a, b in zip(self.alphas, self.betas))

    def cost_of(self, wait, switches):
        """Cost of a trajectory: switches[i] = time of transition to state i+1."""
        cost, cur, last = 0.0, 0, 0.0
        for s in switches:
            if s >= wait:
                break
            cost += self.alphas[cur] * (s - last)
            cur += 1
            last = s
        cost += self.alphas[cur] * (wait - last)
        cost += self.betas[cur]
        return cost


# ---------------------------------------------------------------- policies
# Each policy returns a list of switch times (possibly empty = stay in s0).

def opt(ts, wait, pred=None, rng=None):
    return [0.0] * ts.opt_state(wait)


def ftp(ts, wait, pred, rng=None):
    return [0.0] * ts.opt_state(pred)


def online_det(ts, wait, pred=None, rng=None):
    """Deterministic break-even: enter state j+1 when accumulated rent equals
    the incremental buy. (<=4-competitive for multislope; 2 for two states.)"""
    out, t = [], 0.0
    for a, b in zip(ts.step_a, ts.step_b):
        if a <= 0:
            break
        t = max(t, b / a)
        out.append(t)
    return out


# --- randomized machinery, adapted from adampolak/dpm (CDF, ParetoMu, Bp) ---

def _cdf_classic(x):
    """Prediction-free randomized ski rental CDF on scaled time x in [0,1]."""
    return min(1.0, max(0.0, math.expm1(x) / math.expm1(1)))


def _pareto_mu(rho):
    if rho > 1.1596:
        return (1 - rho * (math.e - 1) / math.e) / math.log(2)
    w = lambertw(-math.sqrt((rho - 1) / (4 * rho))).real
    return (rho - 1) / (2 * w) * (1 + 1 / (2 * w))


def _exp(x):
    """exp with the argument capped — our unscaled times (waits in seconds x
    alpha/beta scale) can be huge; the reference operated on ~O(1) inputs."""
    return math.exp(min(x, 700.0))


def _g0(mu, tp):
    return tp * mu


def _g1int(mu, rho, tp, end):
    return max(0.0, (rho - 1 - mu + mu * tp) * (_exp(end) - 1))


def _g2int(rho, start, end):
    return max(0.0, rho * (_exp(end - 1) - _exp(start - 1)))


def _cdf_rhomu(mu, rho, tp, t):
    """(rho,mu) CDF from Antoniadis et al. Sec 2.1 (three cases).
    All returns clamped to [0,1] (reference relied on small scaled inputs)."""
    if tp >= 1:
        if _g0(mu, tp) >= 1:
            return 1.0
        temp = _g0(mu, tp) + _g1int(mu, rho, tp, t)
        ref = _g0(mu, tp) + _g1int(mu, rho, tp, tp - 1)
        if ref > 1:
            return min(1.0, temp)
        if t < tp - 1:
            return min(1.0, temp)
        if ref > 1 - mu:
            return min(1.0, ref)
        return min(temp, 1 - mu)
    if (rho - 1 - mu + mu * tp) * math.exp(0) <= 0:
        p0 = tp * (rho - 1) / (1 - tp)
        p1 = min(mu, 1 - p0)
        tlim = 1 + math.log((rho - 1 + p0 + p1) / rho)
        if t < tlim:
            return p0
        return p0 + _g2int(rho, tlim, min(t, 1))
    p0 = _g0(mu, tp)
    p1 = min(mu, 1 - p0)
    tlim = 1 + math.log((rho - 1 + p0 + p1 + _g1int(mu, rho, tp, tp)) / rho)
    if t < tlim:
        return min(1 - p1, p0 + _g1int(mu, rho, tp, min(tp, t)))
    return min(1 - p1, p0 + _g1int(mu, rho, tp, tp) + _g2int(rho, tlim, min(t, 1)))


def _inverse(f, y, lo=0.0, hi=1.0, iters=40):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _prudent_bp(ts, cdf_at):
    """B_p(x) = sum_j beta-step_j * CDF_j(x) — cumulative expected buy mass.
    cdf_at(j, x) must give transition-j CDF at unscaled time x."""
    def bp(x):
        tot = 0.0
        for j, (a, b) in enumerate(zip(ts.step_a, ts.step_b)):
            if a <= 0:
                continue
            tot += b * cdf_at(j, x)
        return tot
    return bp


def _prudent_switches(ts, wait, cdf_at, rng, horizon):
    """Prudence conversion (Lotker et al. Thm 4.2 via Antoniadis et al. 4.1):
    switch to state i when B_p reaches beta[i-1] + p_i*(beta[i]-beta[i-1])."""
    bp = _prudent_bp(ts, cdf_at)
    out = []
    for i in range(1, len(ts.alphas)):
        p = rng.random()
        target = ts.betas[i - 1] + p * (ts.betas[i] - ts.betas[i - 1])
        if bp(0.0) >= ts.betas[i]:
            out.append(0.0)
            continue
        if bp(horizon) < target:
            break
        s = _inverse(bp, target, 0.0, horizon)
        if s >= wait:
            break
        out.append(s)
    return out


def online_rand_prudent(ts, wait, pred=None, rng=None, horizon=None):
    """Prediction-free randomized multi-state DPM, prudent. e/(e-1)-competitive
    (RANDOMIZED — state this in the paper; deterministic is 2 / <=4)."""
    rng = rng or random
    horizon = horizon or _default_horizon(ts)

    def cdf_at(j, x):
        scale = ts.step_a[j] / ts.step_b[j]
        return _cdf_classic(min(1.0, x * scale))
    return _prudent_switches(ts, wait, cdf_at, rng, horizon)


def rhomu_prudent(ts, wait, pred, rng=None, rho=1.1596, horizon=None):
    """Learning-augmented (rho, mu(rho)) multi-state DPM, prudent.
    cost <= rho*OPT + mu*eta, eta = alpha_0*|pred - wait|."""
    rng = rng or random
    horizon = horizon or _default_horizon(ts)
    mu = _pareto_mu(rho)

    def cdf_at(j, x):
        scale = ts.step_a[j] / ts.step_b[j]
        return _cdf_rhomu(mu, rho, pred * scale, min(1.0, x * scale))
    return _prudent_switches(ts, wait, cdf_at, rng, horizon)


def _default_horizon(ts):
    """Beyond the last break-even nothing changes; 3x the max is plenty."""
    bes = [b / a for a, b in zip(ts.step_a, ts.step_b) if a > 0]
    return 3 * max(bes) if bes else 1.0


# ---------------------------------------------------------------- registry

def make_policies(rhos=(1.05, 1.1596, 1.3, 1.5)):
    pols = {
        "opt": opt,
        "ftp": ftp,
        "online_det": online_det,
        "online_rand": online_rand_prudent,
    }
    for rho in rhos:
        pols[f"rhomu_{rho}"] = (
            lambda ts, wait, pred, rng=None, _r=rho:
            rhomu_prudent(ts, wait, pred, rng, rho=_r))
    return pols
