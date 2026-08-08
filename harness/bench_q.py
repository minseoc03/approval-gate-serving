#!/usr/bin/env python3
"""
Contention coefficient q (CLAUDE.md 10, Observation 2).

Runs steady Poisson background load, then periodically injects a "resume":
a cold full-context prefill (a gate prompt never seen by the server, so no
prefix-cache hit beyond the shared system prefix). Records every request
with timestamps so the analysis can compare background TTFT inside
[t_inject, t_inject + window] against baseline windows.

The prompt pool is split by task hash: even -> background (reused, so their
KV is warm after first use), odd -> injections (each used at most once, so
every injection pays a real suffix prefill).

Output: JSONL rows {kind: bg|inject, t_send, ttft_s, e2e_s, ...}.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


def load_pools(path, domains=None):
    bg, inj = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if domains and d["domain"] not in domains:
                continue
            for m in d["messages"]:
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    if isinstance(fn.get("arguments"), (dict, list)):
                        fn["arguments"] = json.dumps(fn["arguments"])
            (bg if hash(str(d["task_id"])) % 2 == 0 else inj).append(d)
    return bg, inj


async def one(client, url, model, messages, max_tokens, meta, results):
    t0 = time.perf_counter()
    ttft, usage = None, None
    body = dict(model=model, messages=messages, max_tokens=max_tokens,
                temperature=0.0, stream=True,
                stream_options={"include_usage": True})
    try:
        async with client.stream("POST", url, json=body, timeout=600) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode(errors="replace")[:200]
                results.append(dict(meta, error=f"{r.status_code}: {text}"))
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("choices"):
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content") and ttft is None:
                        ttft = time.perf_counter() - t0
    except Exception as e:
        results.append(dict(meta, error=repr(e)))
        return
    results.append(dict(
        meta, ttft_s=ttft, e2e_s=time.perf_counter() - t0,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        completion_tokens=(usage or {}).get("completion_tokens"),
    ))


async def run(args):
    bg_pool, inj_pool = load_pools(args.prompts, args.domains)
    if not bg_pool or not inj_pool:
        raise SystemExit(f"pool too small: bg={len(bg_pool)} inj={len(inj_pool)}")
    rng = random.Random(args.seed)
    rng.shuffle(inj_pool)
    url = f"http://{args.server}/v1/chat/completions"
    results, tasks = [], []
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        async def injector():
            i = 0
            # let the background load reach steady state first
            await asyncio.sleep(args.inject_every)
            while time.perf_counter() - t0 < args.duration and i < len(inj_pool):
                p = inj_pool[i]
                meta = dict(kind="inject", inj_id=i, task_id=p["task_id"],
                            n_tokens=p["n_tokens"],
                            t_send=time.perf_counter() - t0)
                tasks.append(asyncio.create_task(one(
                    client, url, args.model, p["messages"],
                    args.max_tokens, meta, results)))
                i += 1
                await asyncio.sleep(args.inject_every)

        inj_task = asyncio.create_task(injector())
        i = 0
        t_next = time.perf_counter()
        while time.perf_counter() - t0 < args.duration:
            now = time.perf_counter()
            if now < t_next:
                await asyncio.sleep(t_next - now)
            p = rng.choice(bg_pool)
            meta = dict(kind="bg", req_id=i, task_id=p["task_id"],
                        n_tokens=p["n_tokens"], t_send=time.perf_counter() - t0)
            tasks.append(asyncio.create_task(one(
                client, url, args.model, p["messages"],
                args.max_tokens, meta, results)))
            i += 1
            t_next += rng.expovariate(args.rate)
        await inj_task
        await asyncio.gather(*tasks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(json.dumps(dict(kind="meta", rate=args.rate,
                                duration=args.duration,
                                inject_every=args.inject_every,
                                max_tokens=args.max_tokens)) + "\n")
        for r in sorted(results, key=lambda r: r.get("t_send", 0)):
            f.write(json.dumps(r) + "\n")

    # quick on-line summary: bg TTFT inside vs outside injection windows
    import statistics as st
    inj_times = [r["t_send"] for r in results if r.get("kind") == "inject"]
    bg = [r for r in results if r.get("kind") == "bg" and r.get("ttft_s")]
    win = args.window
    inside = [r["ttft_s"] for r in bg
              if any(t <= r["t_send"] <= t + win for t in inj_times)]
    outside = [r["ttft_s"] for r in bg
               if not any(t <= r["t_send"] <= t + win for t in inj_times)]
    print(f"bg requests: {len(bg)}  injections: {len(inj_times)}  "
          f"errors: {sum(1 for r in results if 'error' in r)}")
    if inside and outside:
        mi, mo = st.median(inside), st.median(outside)
        print(f"bg TTFT median inside {win}s windows: {mi:.3f}s  "
              f"outside: {mo:.3f}s  ratio q~{mi/mo:.2f}")
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="results/gate_prompts.jsonl")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--inject-every", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench_q.jsonl")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
