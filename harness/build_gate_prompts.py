#!/usr/bin/env python3
"""
Build gate-time prompts from tau2-bench recorded trajectories (path A).

For every SimulationRun in data/tau2/results/final/*.json, find the first
assistant message that calls a WRITE tool (the approval gate), truncate the
conversation just before it, prepend the reconstructed system prompt, and
render with the Llama chat template (tools passed via `tools=`).

Output: JSONL, one gate prompt per line, plus a token-length summary that
replaces the stage-2 "assume 2-4K conversation" projection with measured data.

CPU only. Run with the project venv (.venv) that has tau2 + transformers.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def get_write_tools(domain):
    """Tool names decorated @is_tool(ToolType.WRITE) for this domain."""
    from tau2.environment.toolkit import ToolType
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)()
    return {
        name
        for name in env.tools.get_tools()
        if env.tools.tool_type(name) == ToolType.WRITE
    }


def build_system_prompt(domain, repo_root, tools):
    """tau2 system prompt with tool schemas embedded.

    The Llama 3.1 chat template injects `tools=` into the first USER message,
    but tau2 conversations open with an assistant greeting, so that path
    errors out. Embedding the schemas in the system message keeps the token
    accounting consistent between offline measurement and serving (where we
    send messages only, no `tools` param).
    """
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    policy = (
        Path(repo_root) / "data" / "tau2" / "domains" / domain / "policy.md"
    ).read_text()
    base = SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION.strip(), domain_policy=policy.strip()
    )
    tools_block = "\n<tools>\n" + json.dumps(tools, indent=2) + "\n</tools>"
    return base + tools_block


def get_tools_schema(domain):
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)()
    return [t.openai_schema for t in env.get_tools()]


def to_openai_messages(messages):
    """tau2 Message objects -> dicts the Llama chat template accepts.

    Normalizations needed on top of litellm format:
    - content None -> "" (the template calls len()/trim on it)
    - tool_call function.arguments JSON string -> dict (HF template convention)
    - drop the non-standard top-level "name" litellm adds to tool calls
    """
    from tau2.utils.llm_utils import to_litellm_messages

    out = []
    for m in to_litellm_messages(messages):
        # drop keys that are None entirely — the Llama template len()s them
        for key in [k for k, v in m.items() if v is None]:
            del m[key]
        if "content" not in m:
            m["content"] = ""
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tc.pop("name", None)
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
        out.append(m)
    return split_parallel_tool_calls(out)


def split_parallel_tool_calls(msgs):
    """The Llama 3.1 template rejects >1 tool call per assistant message.

    Rewrite [assistant(tc1,tc2), tool1, tool2] as
            [assistant(tc1), tool1, assistant(tc2), tool2],
    pairing tool results by tool_call_id. Token-count impact is a few
    header tokens per extra message; semantics are unchanged.
    """
    out, i = [], 0
    while i < len(msgs):
        m = msgs[i]
        calls = m.get("tool_calls") or []
        if m.get("role") != "assistant" or len(calls) <= 1:
            out.append(m)
            i += 1
            continue
        # collect the tool results that answer these calls
        want = {tc["id"] for tc in calls}
        results = {}
        j = i + 1
        while j < len(msgs) and msgs[j].get("role") == "tool" and want:
            tid = msgs[j].get("tool_call_id")
            if tid in want:
                results[tid] = msgs[j]
                want.discard(tid)
            j += 1
        for idx, tc in enumerate(calls):
            out.append(dict(role="assistant",
                            content=m.get("content", "") if idx == 0 else "",
                            tool_calls=[tc]))
            r = results.get(tc["id"])
            if r:
                out.append(r)
        i = j
    return out


def find_gate_index(messages, write_tools):
    """Index of first assistant message whose tool_calls hit a WRITE tool."""
    for i, m in enumerate(messages):
        if getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc.name in write_tools:
                    return i, tc
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../tau2-bench")
    ap.add_argument("--domains", nargs="+", default=["retail", "airline"])
    ap.add_argument("--tokenizer", required=True, help="local tokenizer path or HF id")
    ap.add_argument("--out", default="results/gate_prompts.jsonl")
    ap.add_argument("--summary", default="results/gate_prompts_summary.json")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "src"))

    from tau2.data_model.simulation import Results
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    results_dir = repo / "data" / "tau2" / "results" / "final"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    seen = set()
    lengths = defaultdict(list)  # domain -> [n_tokens]
    conv_lengths = defaultdict(list)  # domain -> conversation-only tokens
    gate_counts = defaultdict(lambda: defaultdict(int))  # domain -> task_id -> n

    with out_path.open("w") as fout:
        for domain in args.domains:
            write_tools = get_write_tools(domain)
            tools = get_tools_schema(domain)
            system_prompt = build_system_prompt(domain, repo, tools)

            # tokens of system-side prefix alone (system prompt incl. tools),
            # measured through the same chat template for consistency
            prefix_ids = tok.apply_chat_template(
                [{"role": "system", "content": system_prompt}],
                add_generation_prompt=False, tokenize=True, return_dict=False,
            )
            prefix_tokens = len(prefix_ids)

            files = sorted(results_dir.glob(f"*_{domain}_*.json"))
            if not files:
                print(f"[warn] no result files for {domain}", file=sys.stderr)
                continue

            for f in files:
                res = Results.load(f)
                model = res.info.agent_info.llm if res.info else f.stem
                for sim in res.simulations:
                    k, gate_tc = find_gate_index(sim.messages, write_tools)
                    if k is None:
                        continue
                    key = (domain, sim.task_id, gate_tc.name,
                           json.dumps(gate_tc.arguments, sort_keys=True))
                    if key in seen:
                        continue
                    seen.add(key)

                    history = to_openai_messages(sim.messages[:k])
                    msgs = [{"role": "system", "content": system_prompt}] + history
                    try:
                        ids = tok.apply_chat_template(
                            msgs, add_generation_prompt=True, tokenize=True, return_dict=False
                        )
                    except Exception as e:
                        print(f"[skip] {sim.id}: template error {e}", file=sys.stderr)
                        continue

                    n_tokens = len(ids)
                    lengths[domain].append(n_tokens)
                    conv_lengths[domain].append(n_tokens - prefix_tokens)
                    gate_counts[domain][sim.task_id] += 1

                    fout.write(json.dumps(dict(
                        domain=domain,
                        task_id=sim.task_id,
                        source_file=f.name,
                        source_model=str(model),
                        gate_action=dict(name=gate_tc.name,
                                         arguments=gate_tc.arguments),
                        n_history_messages=len(history),
                        prefix_tokens=prefix_tokens,
                        n_tokens=n_tokens,
                        messages=msgs,
                    )) + "\n")
                    n_written += 1

    summary = {}
    for domain in args.domains:
        L, C = lengths[domain], conv_lengths[domain]
        if not L:
            continue
        summary[domain] = dict(
            n_gate_prompts=len(L),
            n_tasks_covered=len(gate_counts[domain]),
            prompt_tokens=dict(
                min=min(L), p25=int(statistics.quantiles(L, n=4)[0]),
                median=int(statistics.median(L)),
                p75=int(statistics.quantiles(L, n=4)[2]),
                p95=int(statistics.quantiles(L, n=20)[18]), max=max(L),
                mean=int(statistics.mean(L)),
            ),
            conversation_tokens=dict(
                median=int(statistics.median(C)), max=max(C),
            ),
        )

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"wrote {n_written} gate prompts -> {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
