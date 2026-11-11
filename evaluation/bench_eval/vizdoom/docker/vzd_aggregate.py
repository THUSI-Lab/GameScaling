#!/usr/bin/env python3
"""Aggregate one model's ViZDoom episodes into a table + summary json.

usage: vzd_aggregate.py RUNS_ROOT SERVED_NAME

Only `result.json` files whose `agent.model` equals SERVED_NAME are counted, so
many models can share one runs tree -- which is also why two batches of the same
checkpoint MUST use different `--served-name` values.

All four maps report the paper's metric, cumulative episode reward: `battle*`
carry a reward script worth +1 per kill (so reward == frags there, verified equal
on every episode), `defend_the_*` are +1 per kill and -1 for dying.

The maps do not share a *scale* though -- `defend_the_line` reaches ~25 while
`battle2` sits near 3 -- so any average over the four is mostly one map. Figure 8
plots the maps separately and reports no aggregate; the per-map means are the
result. `overall` (mean of map means) and `overall_pooled` are kept in the json
only as coarse regression tripwires for the same suite.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

SUITE_ORDER = ["battle1", "battle2", "defend_the_center", "defend_the_line"]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root, served = sys.argv[1], sys.argv[2]

    rows: list[tuple[str, float, dict]] = []
    for rj in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
        try:
            d = json.load(open(rj))
        except (OSError, ValueError):
            continue
        if (d.get("agent") or {}).get("model") != served:
            continue
        metrics = d.get("metrics") or {}
        score = (metrics.get("game") or {}).get("score")
        if score is None:
            continue
        game = d.get("game") or os.path.relpath(rj, root).split(os.sep)[0]
        rows.append((game, float(score), d))

    by: dict[str, list[float]] = defaultdict(list)
    detail: dict[str, list[dict]] = defaultdict(list)
    for game, score, d in rows:
        by[game].append(score)
        detail[game].append(d)

    summary = {"model": served, "maps": {}}
    print(f"\n================ VIZDOOM RESULTS ({served}) ================")
    print(f"  {'map':20} {'from':>8} {'n':>3} {'reward':>8} {'std':>7} {'steps':>7} "
          f"{'unpars':>7} {'noop':>6} {'fire':>6}  scores")
    map_means: list[float] = []
    pooled: list[float] = []
    for game in SUITE_ORDER + sorted(set(by) - set(SUITE_ORDER)):
        if game not in by:
            continue
        scores = by[game]
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        ds = detail[game]

        def avg(path: tuple[str, ...]) -> float:
            vals = []
            for d in ds:
                cur = d.get("metrics") or {}
                for key in path:
                    cur = (cur or {}).get(key)
                if isinstance(cur, (int, float)):
                    vals.append(float(cur))
            return statistics.mean(vals) if vals else 0.0

        steps = avg(("steps",))
        unparsed = avg(("action", "unparsed_rate"))
        noop = avg(("action", "noop_rate"))
        fire = avg(("action", "attack_rate"))
        map_means.append(mean)
        pooled += scores
        summary["maps"][game] = {
            "n": len(scores),
            "mean": mean,
            "std": std,
            "scores": scores,
            "score_mode": ((ds[0].get("metrics") or {}).get("game") or {}).get("score_mode"),
            "mean_steps": steps,
            "mean_unparsed_rate": unparsed,
            "mean_noop_rate": noop,
            "mean_attack_rate": fire,
            "terminations": [d.get("termination") for d in ds],
        }
        # Both modes are cumulative episode reward; this only says which channel
        # it was read from (see the module docstring).
        src = {"killcount": "kills", "total_reward": "engine"}.get(
            summary["maps"][game]["score_mode"] or "", "?")
        print(f"  {game:20} {src:>8} {len(scores):>3} {mean:>8.2f} {std:>7.2f} "
              f"{steps:>7.0f} {unparsed:>7.2f} {noop:>6.2f} {fire:>6.2f}  "
              f"{[round(s, 1) for s in scores]}")

    summary["overall"] = statistics.mean(map_means) if map_means else None
    summary["overall_pooled"] = statistics.mean(pooled) if pooled else None
    summary["n_episodes"] = len(pooled)
    print(f"  {'-' * 84}")
    # Deliberately not called a score. The maps share a unit but not a scale, so
    # any average of the four is mostly defend_the_line; Figure 8 plots them
    # separately and reports no aggregate. Report the row, compare the row.
    print("  mean episode reward per map is the result "
          "(paper Fig. 8 reports no aggregate)")
    print(f"  unweighted mean of {len(map_means)} maps = "
          f"{summary['overall'] if summary['overall'] is None else round(summary['overall'], 3)}"
          f"   [{len(pooled)} episodes]  -- scales differ, regression tripwire only")

    out = os.path.abspath(os.path.join(root, "..", "..", f"eval_summary_{served}.json"))
    try:
        json.dump(summary, open(out, "w"), indent=2)
        print(f"  summary -> {out}")
    except OSError as exc:
        print(f"  (summary write failed: {exc})")
    print("===========================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
