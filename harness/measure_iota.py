#!/usr/bin/env python3
"""
MORI's idleness ratio iota at human-approval waits (CLAUDE.md 6.1 Axis test).

iota = T_acting / (T_reasoning + T_acting) over a k=5 cycle window (MORI 4.1).
From recorded tau2 trajectories: per-message durations from timestamps;
assistant-message production time -> reasoning, tool/user turns -> acting.
At the gate, the ongoing human wait W adds to T_acting. We report the iota
distribution across suspended requests as W grows: if the spread collapses,
the ranking carries no information at human timescales.
"""
import json, re, sys, statistics
from datetime import datetime
from pathlib import Path
sys.path.insert(0, "../tau2-bench/src")
from tau2.data_model.simulation import Results
from build_gate_prompts import get_write_tools

K = 5
repo = Path("../tau2-bench/data/tau2/results/final")
writes = {d: get_write_tools(d) for d in ("retail", "airline")}

def parse(ts): return datetime.fromisoformat(ts) if isinstance(ts, str) else ts

samples = []
for f in sorted(repo.glob("*.json")):
    dom = next((d for d in ("retail", "airline") if f"_{d}_" in f.name), None)
    if not dom: continue
    for sim in Results.load(f).simulations:
        msgs = sim.messages
        k_idx = None
        for i, m in enumerate(msgs):
            if getattr(m, "role", "") == "assistant" and getattr(m, "tool_calls", None):
                if any(tc.name in writes[dom] for tc in m.tool_calls):
                    k_idx = i; break
        if k_idx is None or k_idx < 2: continue
        reason, act, cycles = 0.0, 0.0, []
        for i in range(1, k_idx):
            prev, cur = msgs[i-1], msgs[i]
            try:
                dt = (parse(cur.timestamp) - parse(prev.timestamp)).total_seconds()
            except Exception:
                continue
            if dt < 0 or dt > 300: continue
            if getattr(cur, "role", "") == "assistant":
                cycles.append((dt, 0.0))
            else:
                if cycles: cycles[-1] = (cycles[-1][0], cycles[-1][1] + dt)
                else: cycles.append((0.0, dt))
        win = cycles[-K:]
        r = sum(c[0] for c in win); a = sum(c[1] for c in win)
        if r + a > 0: samples.append((r, a))

print(f"gate trajectories with usable timing: {len(samples)}")
print(f"{'W_elapsed':>10s} {'iota mean':>10s} {'std':>8s} {'p5':>8s} {'p95':>8s} {'spread':>8s}")
for W in [0, 10, 60, 600, 1800]:
    iotas = [ (a + W) / (r + a + W) for r, a in samples ]
    q = statistics.quantiles(iotas, n=20)
    print(f"{W:>10d} {statistics.mean(iotas):>10.4f} {statistics.pstdev(iotas):>8.4f} "
          f"{q[0]:>8.4f} {q[18]:>8.4f} {q[18]-q[0]:>8.4f}")
