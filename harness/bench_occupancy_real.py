#!/usr/bin/env python3
"""
M4-real: HBM occupancy with REAL suspended contexts (no emulation).

Replaces the --kv-cache-memory shrinkage emulation: N distinct gate prompts
are prefilled (max_tokens=1) so their KV actually sits in the prefix cache,
and a keepalive loop re-touches each every --keepalive seconds so LRU cannot
evict them under background pressure. Background load runs from a DISJOINT
prompt pool. Secondary interference (block-table pressure, scheduler
overhead, keepalive traffic itself) is therefore physical, not modeled.

Residency is monitored for free: a keepalive TTFT below --hit-thresh means
the context is still resident (prefix hit); a spike to ~full-prefill time
means vLLM silently evicted it — i.e., the engine converted pin into discard.
We report the eviction onset N alongside the contention curve.

Output per (N, rate): bg TTFT median/p95, output tok/s, keepalive hit rate.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from loadgen import load_prompts, one_request


def to_wire_pool(pool):
    return pool  # loadgen.load_prompts already wire-converts tool_calls


async def send_one(client, url, model, messages, max_tokens=1):
    t0 = time.perf_counter()
    ttft = None
    body = dict(model=model, messages=messages, max_tokens=max_tokens,
                temperature=0.0, stream=True)
    async with client.stream("POST", url, json=body, timeout=600) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            chunk = json.loads(p)
            if chunk.get("choices"):
                d = chunk["choices"][0].get("delta", {})
                if d.get("content") and ttft is None:
                    ttft = time.perf_counter() - t0
    return ttft


async def run_point(args, N, rate, sus_pool, bg_pool):
    base = f"http://{args.server}"
    url = f"{base}/v1/chat/completions"
    async with httpx.AsyncClient() as client:
        # fresh cache per point
        (await client.post(f"{base}/reset_prefix_cache",
                           timeout=60)).raise_for_status()
        # 1. materialize N real suspensions
        sus = sus_pool[:N]
        for p in sus:
            await send_one(client, url, args.model, p["messages"])
        # 2. keepalive loop + 3. background load, concurrently
        ka_ttfts = []
        stop = asyncio.Event()

        async def keepalive():
            i = 0
            while not stop.is_set() and sus:
                p = sus[i % len(sus)]
                try:
                    t = await send_one(client, url, args.model, p["messages"])
                    if t is not None:
                        ka_ttfts.append(t)
                except Exception:
                    pass
                i += 1
                # cycle the whole set every --keepalive seconds
                await asyncio.sleep(max(0.05, args.keepalive / max(len(sus), 1)))

        results = []
        import random as _r
        rng = _r.Random(args.seed)
        ka_task = asyncio.create_task(keepalive())
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
        stop.set()
        await ka_task

    ok = [r for r in results if r.get("ttft_s")]
    tt = sorted(r["ttft_s"] for r in ok)
    hits = [t for t in ka_ttfts if t < args.hit_thresh]
    return dict(
        N=N, rate=rate, n_bg=len(ok),
        bg_ttft_median=statistics.median(tt) if tt else None,
        bg_ttft_p95=tt[int(0.95 * len(tt))] if tt else None,
        out_tok_s=sum(r.get("completion_tokens") or 0 for r in ok) / args.duration,
        ka_n=len(ka_ttfts),
        ka_hit_rate=(len(hits) / len(ka_ttfts)) if ka_ttfts else None,
        ka_ttft_median=statistics.median(ka_ttfts) if ka_ttfts else None,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="results/gate_prompts.jsonl")
    ap.add_argument("--Ns", nargs="+", type=int, default=[0, 25, 50, 65, 72])
    ap.add_argument("--rates", nargs="+", type=float, default=[1.5, 3.0])
    ap.add_argument("--duration", type=float, default=150.0)
    ap.add_argument("--keepalive", type=float, default=30.0)
    ap.add_argument("--hit-thresh", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench_occupancy_real.json")
    args = ap.parse_args()

    pool = [p for p in load_prompts(args.prompts)
            if p["n_tokens"] <= args.max_prompt_tokens]
    # disjoint pools: even task-hash -> suspended, odd -> background
    sus_pool = [p for p in pool if hash(str(p["task_id"])) % 2 == 0]
    bg_pool = [p for p in pool if hash(str(p["task_id"])) % 2 == 1]
    assert len(sus_pool) >= max(args.Ns), "not enough distinct suspended prompts"

    rows = []
    for rate in args.rates:
        for N in args.Ns:
            r = asyncio.run(run_point(args, N, rate, sus_pool, bg_pool))
            rows.append(r)
            print(f"N={N:3d} rate={rate}: bg TTFT med {r['bg_ttft_median']:.3f} "
                  f"p95 {r['bg_ttft_p95']:.3f}  tok/s {r['out_tok_s']:.0f}  "
                  f"keepalive hit {100*(r['ka_hit_rate'] or 0):.0f}% "
                  f"(med {r['ka_ttft_median'] if r['ka_ttft_median'] is None else round(r['ka_ttft_median'],3)}s, n={r['ka_n']})",
                  flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
