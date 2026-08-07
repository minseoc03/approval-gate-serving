#!/usr/bin/env python3
"""
Open-loop Poisson load generator against the vLLM OpenAI endpoint.

Sends gate prompts (from build_gate_prompts.py output) at a fixed arrival
rate, streams responses, and records per-request TTFT / TPOT / E2E.
This is the shared load tool for the q-coefficient measurement and the
Fig 1 suspension-count sweep.

Usage:
  python loadgen.py --rate 2.0 --duration 120 --max-tokens 64 \
      --server $(cut -d' ' -f1 server_info.txt) \
      --out results/load_run.jsonl
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx


def load_prompts(path, domains=None):
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if domains and d["domain"] not in domains:
                continue
            prompts.append(d)
    return prompts


async def one_request(client, url, model, messages, max_tokens, meta, results):
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    body = dict(model=model, messages=messages, max_tokens=max_tokens,
                temperature=0.0, stream=True,
                stream_options={"include_usage": True})
    usage = None
    try:
        async with client.stream("POST", url, json=body, timeout=600) as r:
            if r.status_code != 200:
                text = await r.aread()
                results.append(dict(meta, error=f"{r.status_code}: {text[:200]}"))
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
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_chunks += 1
    except Exception as e:
        results.append(dict(meta, error=repr(e)))
        return
    e2e = time.perf_counter() - t0
    out_tokens = (usage or {}).get("completion_tokens", n_chunks)
    tpot = (e2e - ttft) / max(out_tokens - 1, 1) if ttft is not None else None
    results.append(dict(
        meta,
        ttft_s=ttft, e2e_s=e2e,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        completion_tokens=out_tokens,
        tpot_s=tpot,
    ))


async def run(args):
    prompts = load_prompts(args.prompts, args.domains)
    if not prompts:
        raise SystemExit("no prompts loaded")
    rng = random.Random(args.seed)
    url = f"http://{args.server}/v1/chat/completions"
    results = []
    tasks = []
    t_start = time.perf_counter()
    i = 0
    async with httpx.AsyncClient() as client:
        t_next = t_start
        while time.perf_counter() - t_start < args.duration:
            now = time.perf_counter()
            if now < t_next:
                await asyncio.sleep(t_next - now)
            p = rng.choice(prompts)
            meta = dict(req_id=i, task_id=p["task_id"], domain=p["domain"],
                        n_tokens_est=p["n_tokens"],
                        t_send=time.perf_counter() - t_start)
            tasks.append(asyncio.create_task(one_request(
                client, url, args.model, p["messages"], args.max_tokens,
                meta, results)))
            i += 1
            t_next += rng.expovariate(args.rate)
        await asyncio.gather(*tasks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in sorted(results, key=lambda r: r.get("req_id", 0)):
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if "error" not in r and r.get("ttft_s")]
    err = [r for r in results if "error" in r]
    if ok:
        import statistics as st
        ttfts = [r["ttft_s"] for r in ok]
        print(f"requests: {len(results)}  ok: {len(ok)}  err: {len(err)}")
        print(f"TTFT s  median {st.median(ttfts):.3f}  "
              f"p95 {sorted(ttfts)[int(0.95 * len(ttfts))]:.3f}  "
              f"mean {st.mean(ttfts):.3f}")
        thr = sum(r.get("completion_tokens") or 0 for r in ok) / args.duration
        print(f"output tok/s (whole run): {thr:.1f}")
    if err:
        print("first error:", err[0]["error"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port")
    ap.add_argument("--model", default="llama-3.1-70b")
    ap.add_argument("--prompts", default="results/gate_prompts.jsonl")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--rate", type=float, default=1.0, help="arrivals/sec")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/load_run.jsonl")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()