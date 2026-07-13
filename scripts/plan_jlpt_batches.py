#!/usr/bin/env python3
"""Partition the JLPT grammar seed list into generation batches, weighted
toward N2-N1 per CLAUDE.md ("dồn ngân sách token cho N2–N1").

Weighting is done two ways at once:
  - more sentences per grammar point at higher levels, and
  - grammar points split across more parallel batches at higher levels.

Each batch spec is written to data/synthetic/batches/plan_<id>.json; a
generation subagent reads one spec, produces the pairs, and writes
data/synthetic/batches/jlpt_<id>.jsonl.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEEDS = ROOT / "data" / "synthetic" / "seeds" / "jlpt_grammar.json"
BATCH_DIR = ROOT / "data" / "synthetic" / "batches"

# level -> (sentences per grammar point, number of parallel batches)
PLAN = {
    "N5": (4, 1),
    "N4": (6, 1),
    "N3": (9, 3),
    "N2": (14, 4),
    "N1": (16, 5),
}


def chunk(lst, n):
    """Split lst into n roughly-equal contiguous chunks."""
    k = math.ceil(len(lst) / n)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def main():
    grammar = json.load(open(SEEDS, encoding="utf-8"))
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batches = []
    bid = 0
    grand_total = 0
    for level, (per_point, n_batches) in PLAN.items():
        points = grammar[level]
        for slice_points in chunk(points, n_batches):
            bid += 1
            target = per_point * len(slice_points)
            grand_total += target
            batch_id = f"{level}_{bid:02d}"
            spec = {
                "batch_id": batch_id,
                "level": level,
                "sentences_per_point": per_point,
                "target_pairs": target,
                "grammar_points": slice_points,
                "out_file": str(BATCH_DIR / f"jlpt_{batch_id}.jsonl"),
            }
            (BATCH_DIR / f"plan_{batch_id}.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            batches.append((batch_id, level, len(slice_points), target))

    print(f"{len(batches)} batches, ~{grand_total} target raw pairs")
    print(f"{'batch':<10} {'level':<5} {'points':>7} {'target':>7}")
    by_level = {}
    for batch_id, level, npts, target in batches:
        print(f"{batch_id:<10} {level:<5} {npts:>7} {target:>7}")
        by_level[level] = by_level.get(level, 0) + target
    print("per-level totals:", by_level)
    n2n1 = by_level["N2"] + by_level["N1"]
    print(f"N2+N1 share: {n2n1}/{grand_total} = {100*n2n1/grand_total:.0f}%")


if __name__ == "__main__":
    main()
