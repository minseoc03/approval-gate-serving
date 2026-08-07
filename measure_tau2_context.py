#!/usr/bin/env python3
"""
Stage 1: static context measurement for tau2-bench.

Answers three questions before any GPU work starts:
  1. How many tokens is the shared prefix (policy doc + tool schemas)?
  2. How many tokens of KV does one suspended request actually hold?
  3. What fraction of actions are read-only (speculable) vs DB-modifying
     (approval gates)?

Tokenizer only. No model weights, no GPU.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# KV bytes/token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes
MODEL_CFG = {
    "llama-3.1-8b": dict(layers=32, kv_heads=8, head_dim=128, dtype_bytes=2),
    "llama-3.1-70b": dict(layers=80, kv_heads=8, head_dim=128, dtype_bytes=2),
}

# Actions that mutate the DB require explicit user confirmation per the
# tau-bench policy documents. Everything else is read-only.
WRITE_PREFIXES = (
    "book_", "cancel_", "update_", "modify_", "return_", "exchange_",
    "send_", "transfer_", "create_", "delete_", "set_", "add_", "remove_",
)
READ_PREFIXES = (
    "get_", "find_", "search_", "list_", "calculate_", "check_", "think",
)


def kv_bytes_per_token(name):
    c = MODEL_CFG[name]
    return 2 * c["layers"] * c["kv_heads"] * c["head_dim"] * c["dtype_bytes"]


def classify(action_name):
    if action_name.startswith(WRITE_PREFIXES):
        return "write"
    if action_name.startswith(READ_PREFIXES):
        return "read"
    return "unknown"


def load_tokenizer(model_id):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed: pip install transformers")
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        sys.exit(
            f"could not load tokenizer '{model_id}': {e}\n"
            "If the repo is gated, either accept the license and set HF_TOKEN, "
            "or pass --tokenizer NousResearch/Meta-Llama-3.1-8B-Instruct"
        )


def get_tool_schemas(domain, repo_root):
    """Try the real registry first; fall back to parsing tool source."""
    sys.path.insert(0, str(Path(repo_root) / "src"))
    try:
        from tau2.registry import registry
        env = registry.get_env_constructor(domain)()
        tools = env.get_tools()
        schemas = []
        for t in tools:
            if hasattr(t, "openai_schema"):
                schemas.append(t.openai_schema)
            elif isinstance(t, dict):
                schemas.append(t)
            else:
                schemas.append({"name": getattr(t, "name", str(t)),
                                "description": getattr(t, "description", "")})
        return schemas, "registry"
    except Exception:
        pass

    # Fallback: count docstrings in tools.py as a rough schema proxy
    src = Path(repo_root) / "src" / "tau2" / "domains" / domain / "tools.py"
    if not src.exists():
        return [], "unavailable"
    text = src.read_text()
    docs = re.findall(r'"""(.*?)"""', text, re.S)
    names = re.findall(r"def\s+(\w+)\s*\(", text)
    schemas = [{"name": n, "description": d.strip()}
               for n, d in zip(names, docs)]
    return schemas, "source-fallback"


def measure_domain(domain, repo_root, tok):
    data_dir = Path(repo_root) / "data" / "tau2" / "domains" / domain
    policy_path = data_dir / "policy.md"
    if not policy_path.exists():
        policy_path = data_dir / "main_policy.md"
    if not policy_path.exists():
        return None

    policy_text = policy_path.read_text()
    policy_tok = len(tok.encode(policy_text))

    schemas, schema_src = get_tool_schemas(domain, repo_root)
    schema_tok = len(tok.encode(json.dumps(schemas))) if schemas else 0

    tasks = json.loads((data_dir / "tasks.json").read_text())

    instr_tok, n_actions, action_names = [], [], Counter()
    for t in tasks:
        scen = (t.get("user_scenario") or {}).get("instructions") or {}
        blob = " ".join(str(v) for v in scen.values() if v)
        if blob:
            instr_tok.append(len(tok.encode(blob)))
        acts = (t.get("evaluation_criteria") or {}).get("actions") or []
        n_actions.append(len(acts))
        for a in acts:
            action_names[a["name"]] += 1

    kinds = Counter(classify(n) for n in action_names)
    kind_counts = Counter()
    for name, c in action_names.items():
        kind_counts[classify(name)] += c

    return dict(
        domain=domain,
        policy_tokens=policy_tok,
        schema_tokens=schema_tok,
        schema_source=schema_src,
        n_tools=len(schemas),
        shared_prefix_tokens=policy_tok + schema_tok,
        n_tasks=len(tasks),
        instruction_tokens_median=int(statistics.median(instr_tok)) if instr_tok else 0,
        actions_per_task_median=statistics.median(n_actions) if n_actions else 0,
        actions_per_task_max=max(n_actions) if n_actions else 0,
        distinct_actions=len(action_names),
        action_kind_types=dict(kinds),
        action_kind_calls=dict(kind_counts),
        top_actions=action_names.most_common(10),
    )


def project_kv(shared_tok, suffix_tok):
    """KV footprint for shared prefix vs request-specific suffix."""
    out = {}
    for m in MODEL_CFG:
        b = kv_bytes_per_token(m)
        out[m] = dict(
            bytes_per_token=b,
            shared_gb=shared_tok * b / 1e9,
            suffix_gb=suffix_tok * b / 1e9,
            total_gb=(shared_tok + suffix_tok) * b / 1e9,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to tau2-bench checkout")
    ap.add_argument("--domains", nargs="+", default=["retail", "airline", "telecom"])
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--suffix-tokens", type=int, default=4000,
                    help="assumed conversation length at the approval gate; "
                         "replace with the measured value after stage 2")
    ap.add_argument("--hbm-gb", type=float, default=80.0)
    ap.add_argument("--weights-gb", type=float, default=16.0)
    ap.add_argument("--out", default="context_report.json")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    results = []
    for d in args.domains:
        r = measure_domain(d, args.repo, tok)
        if r:
            results.append(r)
        else:
            print(f"[skip] {d}: no policy file found", file=sys.stderr)

    print(f"\ntokenizer: {args.tokenizer}")
    print(f"assumed conversation suffix at gate: {args.suffix_tokens} tokens\n")

    header = f"{'domain':10s} {'policy':>8s} {'schemas':>8s} {'prefix':>8s} {'tools':>6s} {'tasks':>6s} {'acts/task':>10s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['domain']:10s} {r['policy_tokens']:8d} {r['schema_tokens']:8d} "
              f"{r['shared_prefix_tokens']:8d} {r['n_tools']:6d} {r['n_tasks']:6d} "
              f"{r['actions_per_task_median']:10.0f}")

    print("\nKV footprint per suspended request")
    for r in results:
        proj = project_kv(r["shared_prefix_tokens"], args.suffix_tokens)
        r["kv_projection"] = proj
        print(f"\n  {r['domain']}  (prefix {r['shared_prefix_tokens']} + suffix {args.suffix_tokens})")
        usable = args.hbm_gb - args.weights_gb
        for m, p in proj.items():
            n_fit = usable / p["suffix_gb"] if p["suffix_gb"] > 0 else float("inf")
            print(f"    {m:14s} total {p['total_gb']:.2f} GB "
                  f"(shared {p['shared_gb']:.2f} + private {p['suffix_gb']:.2f})   "
                  f"~{n_fit:.0f} private suffixes fit in {usable:.0f} GB")

    print("\nAction mix (read-only is speculable; writes are approval gates)")
    for r in results:
        calls = r["action_kind_calls"]
        tot = sum(calls.values()) or 1
        print(f"  {r['domain']:10s} " + "  ".join(
            f"{k}: {v} ({100*v/tot:.0f}%)" for k, v in sorted(calls.items())))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
