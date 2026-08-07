#!/usr/bin/env python3
"""
M3: re-prefill cost at resume (CLAUDE.md 10.1).

beta_discard = F = GPU-seconds to re-prefill the PRIVATE SUFFIX only.
The ~5.4K-token shared prefix is warm in the prefix cache in any realistic
deployment, so charging the full context would inflate F by ~2.8x.

Protocol per sampled gate prompt (exclusive server, no other load):
  1. POST /reset_prefix_cache
  2. warm the shared prefix: send the system message alone (max_tokens=1)
  3. send the full gate prompt          -> TTFT_suffix  (prefix cache hit)
  4. POST /reset_prefix_cache
  5. send the full gate prompt again    -> TTFT_full    (cold, whole context)

GPU-seconds = TTFT x n_gpus (TP occupies all ranks during prefill).

Output JSON rows per prompt with both measures; summary with median + IQR.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import httpx



def to_wire(messages):
    """OpenAI wire format: tool_call function.arguments must be a JSON string."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = json.loads(json.dumps(m))
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), (dict, list)):
                    fn["arguments"] = json.dumps(fn["arguments"])
        out.append(m)
    return out

def stream_ttft(client, base, model, messages, max_tokens=1):
    body = dict(model=model, messages=messages, max_tokens=max_tokens,
                temperature=0.0, stream=True,
                stream_options={"include_usage": True})
    t0 = time.perf_counter()
    ttft, usage = None, None
    with client.stream("POST", f"{base}/v1/chat/completions", json=body,
                       timeout=600) as r:
        if r.status_code != 200:
            detail = r.read().decode(errors="replace")[:400]
            raise RuntimeError(f"HTTP {r.status_code}: {detail}")
        for line in r.iter_lines():
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
    return ttft, usage


def reset(client, base):
    client.post(f"{base}/reset_prefix_cache", timeout=60).raise_for_status()
    time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="results/gate_prompts.jsonl")
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-gpus", type=int, default=4,
                    help="TP degree; converts wall seconds to GPU-seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench_prefill.json")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.prompts)]
    # stratified sample across the length distribution
    pool.sort(key=lambda p: p["n_tokens"])
    idx = [round(i * (len(pool) - 1) / max(args.n_prompts - 1, 1))
           for i in range(args.n_prompts)]
    sample = [pool[i] for i in sorted(set(idx))]
    random.Random(args.seed).shuffle(sample)

    base = f"http://{args.server}"
    rows = []
    with httpx.Client() as client:
        for p in sample:
            wire_msgs = to_wire(p["messages"])
            system_only = [wire_msgs[0]]
            for rep in range(args.reps):
                reset(client, base)
                stream_ttft(client, base, args.model, system_only)   # warm prefix
                t_suffix, usage = stream_ttft(client, base, args.model,
                                              wire_msgs)
                reset(client, base)
                t_full, _ = stream_ttft(client, base, args.model, wire_msgs)
                rows.append(dict(
                    task_id=p["task_id"], domain=p["domain"], rep=rep,
                    n_tokens=p["n_tokens"], prefix_tokens=p["prefix_tokens"],
                    suffix_tokens=p["n_tokens"] - p["prefix_tokens"],
                    prompt_tokens_reported=(usage or {}).get("prompt_tokens"),
                    ttft_suffix_s=t_suffix, ttft_full_s=t_full,
                    gpu_s_suffix=t_suffix * args.n_gpus if t_suffix else None,
                    gpu_s_full=t_full * args.n_gpus if t_full else None,
                ))
                print(f"{p['task_id']:>6} rep{rep} n={p['n_tokens']:5d}: "
                      f"suffix {t_suffix:.3f}s  full {t_full:.3f}s")

    ok = [r for r in rows if r["ttft_suffix_s"] and r["ttft_full_s"]]
    def q(key):
        xs = [r[key] for r in ok]
        qs = statistics.quantiles(xs, n=4) if len(xs) >= 4 else [min(xs)]*3
        return dict(median=statistics.median(xs), iqr=[qs[0], qs[2]])
    summary = dict(
        n_rows=len(ok), n_gpus=args.n_gpus, model=args.model,
        ttft_suffix_s=q("ttft_suffix_s"), ttft_full_s=q("ttft_full_s"),
        gpu_s_suffix=q("gpu_s_suffix"), gpu_s_full=q("gpu_s_full"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
