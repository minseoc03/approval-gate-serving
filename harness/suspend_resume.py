#!/usr/bin/env python3
"""
Suspend/resume demonstration across KV placement tiers (the Aug 5-7 gate:
"suspension actually happens").

Protocol per (tier, prompt, repeat):
  1. warm    : send the gate prompt (max_tokens=1) -> KV materialized
  2. suspend : idle for --wait seconds (short; long waits are composed
               analytically per the cost model, never run in wall-clock)
  3. intervene, by tier:
       pin     : nothing (KV stays in HBM prefix cache)
       discard : POST /reset_prefix_cache (GPU cache dropped)
       cpu     : POST /reset_prefix_cache (GPU dropped; OffloadingConnector's
                 CPU copy should survive and promote on resume)
       nvme    : POST /reset_prefix_cache (GPU dropped; LMCache disk copy
                 should survive and reload on resume)
  4. resume  : append the approval user turn, stream, measure TTFT
  5. cold    : reset everything, send the same resume prompt -> full-prefill
               TTFT baseline for this prompt

Expected ordering of resume TTFT: pin < cpu <= nvme < discard ~= cold.
The server must be launched (serve_vllm.slurm) with a TIER that provides
the tier under test: hbm covers {pin, discard}; cpu covers {pin, cpu,
discard}; nvme covers {pin, nvme, discard}.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

APPROVAL_TURN = {"role": "user",
                 "content": "Approved — please proceed with the action now."}


def to_wire(messages):
    """OpenAI wire format: tool_call function.arguments must be a JSON string.

    gate_prompts.jsonl stores arguments as dicts (HF chat-template convention
    for offline tokenization); convert on the way out.
    """
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


def stream_ttft(client, base, model, messages, max_tokens):
    """Send one streaming request; return (ttft_s, e2e_s, usage)."""
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
    return ttft, time.perf_counter() - t0, usage


def reset_prefix_cache(client, base):
    r = client.post(f"{base}/reset_prefix_cache", timeout=60)
    r.raise_for_status()


def run_tier(client, base, model, tier, prompts, wait_s, max_tokens, repeats):
    rows = []
    for p in prompts:
        warm_msgs = to_wire(p["messages"])
        resume_msgs = warm_msgs + [APPROVAL_TURN]
        for rep in range(repeats):
            # make sure this prompt's KV is not already resident from the
            # previous repeat: full reset before every trial
            reset_prefix_cache(client, base)
            time.sleep(0.5)

            # 1. warm
            stream_ttft(client, base, model, warm_msgs, 1)
            # 2. suspend
            time.sleep(wait_s)
            # 3. intervene
            if tier in ("discard", "cpu", "nvme"):
                reset_prefix_cache(client, base)
                time.sleep(0.2)
            # 4. resume
            ttft, e2e, usage = stream_ttft(client, base, model,
                                           resume_msgs, max_tokens)
            # 5. cold baseline
            reset_prefix_cache(client, base)
            time.sleep(0.5)
            cold_ttft, _, _ = stream_ttft(client, base, model,
                                          resume_msgs, max_tokens)

            rows.append(dict(
                tier=tier, task_id=p["task_id"], domain=p["domain"],
                rep=rep, n_tokens=p["n_tokens"],
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                resume_ttft_s=ttft, cold_ttft_s=cold_ttft, resume_e2e_s=e2e,
            ))
            print(f"  {tier} {p['task_id']} rep{rep}: "
                  f"resume TTFT {ttft:.3f}s  cold {cold_ttft:.3f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port")
    ap.add_argument("--model", default="llama-3.1-70b")
    ap.add_argument("--prompts", default="results/gate_prompts.jsonl")
    ap.add_argument("--tiers", nargs="+", default=["pin", "discard"],
                    choices=["pin", "cpu", "nvme", "discard"])
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--wait", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/suspend_resume.jsonl")
    args = ap.parse_args()

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
            if len(prompts) >= args.n_prompts:
                break

    base = f"http://{args.server}"
    all_rows = []
    with httpx.Client() as client:
        for tier in args.tiers:
            print(f"[tier] {tier}")
            all_rows += run_tier(client, base, args.model, tier, prompts,
                                 args.wait, args.max_tokens, args.repeats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print("\n=== resume TTFT by tier (s) ===")
    for tier in args.tiers:
        t = [r["resume_ttft_s"] for r in all_rows
             if r["tier"] == tier and r["resume_ttft_s"]]
        c = [r["cold_ttft_s"] for r in all_rows
             if r["tier"] == tier and r["cold_ttft_s"]]
        if t:
            print(f"{tier:8s} median {statistics.median(t):.3f}  "
                  f"mean {statistics.mean(t):.3f}  "
                  f"(cold baseline median {statistics.median(c):.3f})")


if __name__ == "__main__":
    main()