# Approval-Gate Serving

A measurement harness for studying **LLM serving of agents suspended at
human-approval gates** — the regime where an agentic request pauses for
minutes to hours waiting for a human to approve a side-effecting action,
while its KV cache occupies GPU memory.

Existing agentic serving work optimizes for second-scale stalls (tool calls),
where resume latency dominates. At human-approval time scales the resume
stall becomes negligible and the dominant cost is what the suspended request
takes away from every other request: HBM capacity (if pinned), PCIe/NVMe
bandwidth (if offloaded), or prefill compute (if discarded and recomputed).
This harness measures those three contentions and demonstrates
suspend/resume across a four-tier KV placement hierarchy
(HBM / CPU DRAM / local NVMe / discard) on top of vLLM.

## Components

| File | Purpose |
|---|---|
| `measure_tau2_context.py` | Static context measurement over [τ²-bench](https://github.com/sierra-research/tau2-bench): tokenizes policy documents and tool schemas, projects per-request KV footprints, classifies read vs. write (approval-gated) actions. CPU-only. |
| `harness/build_gate_prompts.py` | Builds gate-time prompts offline by truncating τ²-bench's recorded simulation trajectories at the first WRITE tool call — the approval gate, immediately after the user's explicit confirmation turn. Renders through the Llama chat template. Produces 449 unique gate prompts (retail + airline) with measured token-length distributions. |
| `harness/serve_vllm.slurm` | Slurm launcher for a vLLM OpenAI-compatible server with per-tier KV configurations: prefix caching only (`hbm`), native CPU offloading via `OffloadingConnector` (`cpu`), or LMCache with a local-NVMe backend (`nvme`). |
| `harness/suspend_resume.py` | Demonstrates one suspend→resume cycle per tier and measures resume TTFT against a cold-prefill baseline. Expected ordering: pin < cpu ≤ nvme < discard ≈ cold. |
| `harness/loadgen.py` | Open-loop Poisson load generator (async, streaming) recording per-request TTFT / TPOT / end-to-end latency; used for contention and throughput-collapse sweeps. |
| `harness/microbench_transfer.py` | Engine-independent transfer microbenchmarks at KV-cache granularity: pinned-memory D2H/H2D bandwidth and O_DIRECT NVMe read/write. |
| `harness/smoke_all_tiers.slurm` | Self-contained single-GPU job that cycles a server through all three tier configurations and runs the suspend/resume demonstration against each. |
| `node_diag.slurm` | Compute-node diagnostics: GPU model/HBM, RAM, local-disk topology and bandwidth, outbound connectivity. |

## Method notes

- **No wall-clock waiting.** Hour-long suspensions are never run in real time;
  component costs (occupancy, transfer, recompute, contention) are measured
  separately and composed under a cost model.
- **Gates come from the benchmark, not from us.** τ²-bench policies mandate
  explicit user confirmation before any database-modifying action, and write
  tools are tagged in the benchmark source (`ToolType.WRITE`), so approval
  gates are identified without running an agent.
- **Real conversations, no agent LLM in the loop.** Gate prompts are cut from
  trajectories recorded with strong models, preserving realistic conversation
  shape and length (median ≈ 8.4K tokens at the gate).

## Requirements

vLLM ≥ 0.11 (KV offloading connector), optionally LMCache for the NVMe tier,
`transformers` for offline tokenization, and a local Llama-3.1 checkpoint.
Cluster scripts assume Slurm; paths and model locations are passed via
environment variables (`MODEL`, `WORK`, `TIER`, …).
