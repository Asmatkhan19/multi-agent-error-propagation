#!/usr/bin/env python3
"""Summarize pilot results.

Metrics (per condition, over all tasks x seeds):
  accuracy@r      share of runs whose answer after round r is correct
  adoption@r      (injected only) share whose answer matches the injected false answer
  correction rate (injected) among runs that adopted the false answer at round 0,
                  the share that no longer hold it at the final round
  late adoption   (injected) among runs that did NOT adopt it at round 0,
                  the share that hold it at the final round (the loop introduced it)

Usage: python analyze.py results/pilot.jsonl
"""
import json
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "  n/a "


def main(path):
    rows = load(path)
    if not rows:
        sys.exit("No results found.")

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    n_rounds = max(len(r["rounds"]) for r in rows)
    print(f"{len(rows)} runs, model(s): {sorted({r.get('model', '?') for r in rows})}\n")

    for cond, runs in sorted(by_cond.items()):
        print(f"== condition: {cond} (n={len(runs)})")
        for rnd in range(n_rounds):
            at = [r["rounds"][rnd] for r in runs if len(r["rounds"]) > rnd]
            acc = sum(x["correct"] for x in at)
            line = f"  round {rnd}: accuracy {pct(acc, len(at))}"
            if cond == "injected":
                adopted = sum(x["adopted_false"] for x in at)
                line += f" | adopted false answer {pct(adopted, len(at))}"
            print(line)

        if cond == "injected":
            adopted0 = [r for r in runs if r["rounds"][0]["adopted_false"]]
            clean0 = [r for r in runs if not r["rounds"][0]["adopted_false"]]
            corrected = sum(not r["rounds"][-1]["adopted_false"] for r in adopted0)
            late = sum(r["rounds"][-1]["adopted_false"] for r in clean0)
            print(f"  correction rate: {pct(corrected, len(adopted0))} "
                  f"({corrected}/{len(adopted0)} runs adopted at round 0)")
            print(f"  late adoption:   {pct(late, len(clean0))} "
                  f"({late}/{len(clean0)} runs were clean at round 0)")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pilot.jsonl")
