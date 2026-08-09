#!/usr/bin/env python3
"""
Approval rate at gates, from recorded tau2-bench trajectories (keyword proxy).

For every assistant message that asks for confirmation (no tool call, text
matches an ask pattern), classify the next user reply as approve / decline /
unclear. Grounds the Paper-2 premise ("speculating the yes-branch has high
expected value") with a measured number. Approximate by construction —
report as such.

CPU-only; run with .venv (tau2 installed).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ASK = re.compile(
    r"(please confirm|can you confirm|do you confirm|shall i (?:go ahead|proceed)"
    r"|would you like (?:me )?to proceed|do you want (?:me )?to proceed"
    r"|confirm (?:with (?:a )?)?[\"']?yes[\"']?|say [\"']?yes[\"']?"
    r"|(?:reply|respond) (?:with )?[\"']?yes[\"']?|\(yes\)|proceed\?)",
    re.I)
YES = re.compile(
    r"^\s*(yes|yep|yeah|sure|confirm|confirmed|correct|go ahead|please (?:do|proceed)"
    r"|sounds good|that works|ok(?:ay)?[,. ])|(^|\W)i confirm\b", re.I)
NO = re.compile(
    r"^\s*(no\b|nope|don'?t|do not|cancel|stop|wait|hold on|actually,? no"
    r"|not yet|never mind)|(\W)(don'?t|do not) (proceed|do (?:that|it))", re.I)


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else "../tau2-bench").resolve()
    sys.path.insert(0, str(repo / "src"))
    from tau2.data_model.simulation import Results

    counts = Counter()
    per_domain = {}
    for f in sorted((repo / "data/tau2/results/final").glob("*.json")):
        domain = next((d for d in ("retail", "airline") if f"_{d}_" in f.name), None)
        if not domain:
            continue
        res = Results.load(f)
        for sim in res.simulations:
            msgs = sim.messages
            for i, m in enumerate(msgs[:-1]):
                if getattr(m, "role", "") != "assistant" or getattr(m, "tool_calls", None):
                    continue
                if not m.content or not ASK.search(m.content):
                    continue
                nxt = msgs[i + 1]
                if getattr(nxt, "role", "") != "user" or not nxt.content:
                    continue
                if YES.search(nxt.content):
                    kind = "approve"
                elif NO.search(nxt.content):
                    kind = "decline"
                else:
                    kind = "unclear"
                counts[(domain, kind)] += 1
                per_domain.setdefault(domain, Counter())[kind] += 1

    print("Confirmation-ask -> next-user-reply classification (keyword proxy):\n")
    for domain, c in per_domain.items():
        tot = sum(c.values())
        dec = c["approve"] + c["decline"]
        print(f"  {domain:8s} asks={tot:5d}  approve={c['approve']:5d}  "
              f"decline={c['decline']:4d}  unclear={c['unclear']:4d}")
        if dec:
            print(f"           approval rate (excl. unclear): "
                  f"{100*c['approve']/dec:.1f}%")
    both = Counter()
    for c in per_domain.values():
        both.update(c)
    dec = both["approve"] + both["decline"]
    if dec:
        print(f"\n  OVERALL approval rate (excl. unclear): "
              f"{100*both['approve']/dec:.1f}%  "
              f"({both['approve']}/{dec} classified; {both['unclear']} unclear)")


if __name__ == "__main__":
    main()
