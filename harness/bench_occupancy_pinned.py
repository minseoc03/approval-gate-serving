#!/usr/bin/env python3
"""
M4-pinned: HBM occupancy with ENFORCED residency (CLAUDE.md 13.1 item 8b).

Requires the BlockPool victim-exclusion patch (harness/patches/, verified
2026-08-16). The engine pins the first N finished requests when the control
file says {"mode":"first_n","n":N}; policy stays harness-side.

Per (N, rate) point:
  1. ctrl off  -> reset_prefix_cache (unpins + clears)
  2. ctrl first_n=N -> send N suspended gate prompts sequentially
     (each is pinned by the engine as it finishes)
  3. ctrl off  -> background Poisson load from a DISJOINT prompt pool
  4. resume probe: re-send each suspended prompt (max_tokens=1); TTFT below
     --hit-thresh = resident.  With the patch this should be ~100 % — the
     stock-engine arm measured 15 % at N=72/rate 3.0 (job 2062769).

The bg TTFT curve vs N under enforced residency IS alpha_HBM measured
properly (the 2026-08-16b correction). Engine-side events land in
VLLM_PIN_EVENTS jsonl for pin/touch/unpin attribution.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from bench_occupancy_real import send_one
from loadgen import load_prompts, one_request


def set_ctrl(path, mode, n=0):
    Path(path).write_text(json.dumps({"mode": mode, "n": n}))
    time.sleep(0.2)   # engine polls mtime per request-free; slack for clock


async def run_point(args, N, rate, sus_pool, bg_pool, arm="pinned"):
    """arm='pinned': engine pins the N suspended contexts (first_n).
    arm='stock'  : identical protocol, ctrl stays off -> plain LRU.
    Residency is measured for BOTH arms by the same end-of-load resume
    probe (item 12 fix: the stock keepalive hit rate was a different,
    period-confounded instrument -- effective period 124 s at N=72)."""
    base = f"http://{args.server}"
    url = f"{base}/v1/chat/completions"
    async with httpx.AsyncClient() as client:
        set_ctrl(args.ctrl, "off")
        (await client.post(f"{base}/reset_prefix_cache",
                           timeout=60)).raise_for_status()
        if arm == "pinned":
            set_ctrl(args.ctrl, "first_n", N)
        sus = sus_pool[:N]
        for p in sus:                       # sequential: finish order == send order
            await send_one(client, url, args.model, p["messages"])
        set_ctrl(args.ctrl, "off")

        results = []
        import random as _r
        rng = _r.Random(args.seed)
        t_start = time.perf_counter()
        tasks, t_next, i = [], t_start, 0
        while time.perf_counter() - t_start < args.duration:
            now = time.perf_counter()
            if now < t_next:
                await asyncio.sleep(t_next - now)
            p = rng.choice(bg_pool)
            meta = dict(req_id=i, t_send=time.perf_counter() - t_start)
            tasks.append(asyncio.create_task(one_request(
                client, url, args.model, p["messages"], args.max_tokens,
                meta, results)))
            i += 1
            t_next += rng.expovariate(rate)
        await asyncio.gather(*tasks)

        # resume probe — measures realized residency under enforced pinning
        probe_ttfts = []
        for p in sus:
            try:
                t = await send_one(client, url, args.model, p["messages"])
                if t is not None:
                    probe_ttfts.append(t)
            except Exception:
                pass

    # item 13: symmetric warm-up exclusion across every condition and arm
    ok = [r for r in results
          if r.get("ttft_s") and r.get("t_send", 0) >= args.warmup]
    tt = sorted(r["ttft_s"] for r in ok)
    hits = [t for t in probe_ttfts if t < args.hit_thresh]
    return dict(
        arm=arm, N=N, rate=rate, n_bg=len(ok),
        bg_ttft_median=statistics.median(tt) if tt else None,
        bg_ttft_p95=tt[int(0.95 * len(tt))] if tt else None,
        out_tok_s=sum(r.get("completion_tokens") or 0 for r in ok) / args.duration,
        probe_n=len(probe_ttfts),
        resume_hit_rate=(len(hits) / len(probe_ttfts)) if probe_ttfts else None,
        resume_ttft_median=(statistics.median(probe_ttfts)
                            if probe_ttfts else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctrl", required=True, help="VLLM_PIN_CONTROL path")
    ap.add_argument("--prompts", default="results/gate_prompts.jsonl")
    ap.add_argument("--Ns", nargs="+", type=int, default=[0, 25, 50, 65, 72])
    ap.add_argument("--rates", nargs="+", type=float, default=[1.5, 3.0])
    ap.add_argument("--duration", type=float, default=150.0)
    ap.add_argument("--hit-thresh", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=30.0,
                    help="seconds of bg samples to discard, symmetric (item 13)")
    ap.add_argument("--arms", nargs="+", default=["stock", "pinned"])
    ap.add_argument("--max-prompt-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench_occupancy_pinned.json")
    args = ap.parse_args()

    pool = [p for p in load_prompts(args.prompts)
            if p["n_tokens"] <= args.max_prompt_tokens]
    sus_pool = [p for p in pool if hash(str(p["task_id"])) % 2 == 0]
    bg_pool = [p for p in pool if hash(str(p["task_id"])) % 2 == 1]
    assert len(sus_pool) >= max(args.Ns)

    rows = []
    for rate in args.rates:
        for N in args.Ns:
          for arm in args.arms:
            r = asyncio.run(run_point(args, N, rate, sus_pool, bg_pool, arm))
            rows.append(r)
            hr = r["resume_hit_rate"]
            print(f"[{arm:6s}] N={N:3d} rate={rate}: bg TTFT med {r['bg_ttft_median']:.3f} "
                  f"p95 {r['bg_ttft_p95']:.3f}  tok/s {r['out_tok_s']:.0f}  "
                  f"resume hit {('%.0f%%' % (100*hr)) if hr is not None else 'n/a'} "
                  f"(med {r['resume_ttft_median'] and round(r['resume_ttft_median'],3)}s)",
                  flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
