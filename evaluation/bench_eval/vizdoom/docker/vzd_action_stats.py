#!/usr/bin/env python3
"""Behaviour report for one ViZDoom run: what the policy actually did.

usage: vzd_action_stats.py RUNS_ROOT SERVED_NAME [--markdown]

The score alone cannot distinguish "the model played badly" from "the model
never fired", "the model never aimed" or "the model narrated instead of acting".
This prints the per-map behaviour columns behind the score, plus the histogram of
keys the parser actually accepted -- a model that spends its chunks on keys Doom
does not use (`Tab`, `F1`, `Q`) shows up here and nowhere else.

`--markdown` emits a Markdown row for a results table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter, defaultdict

SUITE_ORDER = ["battle1", "battle2", "defend_the_center", "defend_the_line"]


def load(root: str, served: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for rj in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
        try:
            d = json.load(open(rj))
        except (OSError, ValueError):
            continue
        if (d.get("agent") or {}).get("model") != served:
            continue
        d["_dir"] = os.path.dirname(rj)
        by[d.get("game") or "?"].append(d)
    return by


def _mean(ds: list[dict], *path: str) -> float:
    vals = []
    for d in ds:
        cur: object = d.get("metrics") or {}
        for key in path:
            cur = (cur or {}).get(key) if isinstance(cur, dict) else None
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    return statistics.mean(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("served")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    by = load(args.root, args.served)
    if not by:
        print(f"no results for {args.served} under {args.root}")
        return 1

    games = [g for g in SUITE_ORDER if g in by] + sorted(set(by) - set(SUITE_ORDER))
    all_ds = [d for g in games for d in by[g]]
    params = (all_ds[0].get("parameters") or {}) if all_ds else {}

    print(f"\n=============== BEHAVIOUR ({args.served}) ===============")
    print(f"  history={params.get('history_len')} frames  policy sees "
          f"{params.get('resize')}  rendered {params.get('resolution')}  "
          f"temp={params.get('temperature')}  prompt={params.get('prompt_style')}  "
          f"clock={params.get('clock_mode')}  max_steps={params.get('max_steps')}")
    header = (f"  {'map':19}{'n':>3}{'score':>8}{'steps':>7}{'unpars':>8}{'noop':>7}"
              f"{'fire':>7}{'aim':>7}{'|X|':>7}{'think':>7}{'thchr':>7}{'lat_ms':>8}")
    print(header)
    rows = {}
    for g in games:
        ds = by[g]
        scores = [float(((d.get("metrics") or {}).get("game") or {}).get("score", 0.0)) for d in ds]
        rows[g] = statistics.mean(scores) if scores else 0.0
        print(f"  {g:19}{len(ds):>3}{rows[g]:>8.2f}{_mean(ds, 'steps'):>7.0f}"
              f"{_mean(ds, 'action', 'unparsed_rate'):>8.2f}"
              f"{_mean(ds, 'action', 'noop_rate'):>7.2f}"
              f"{_mean(ds, 'action', 'attack_rate'):>7.2f}"
              f"{_mean(ds, 'action', 'mouse_nonzero_rate'):>7.2f}"
              f"{_mean(ds, 'action', 'mean_abs_x'):>7.0f}"
              f"{_mean(ds, 'reasoning', 'think_rate'):>7.2f}"
              f"{_mean(ds, 'reasoning', 'mean_think_chars'):>7.0f}"
              f"{_mean(ds, 'latency', 'mean_ms'):>8.0f}")

    keys: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    terms: Counter[str] = Counter()
    for d in all_ds:
        action = (d.get("metrics") or {}).get("action") or {}
        keys.update(action.get("key_uses") or {})
        dropped.update(action.get("dropped_tokens") or {})
        terms[d.get("termination") or "?"] += 1
    total_keys = sum(keys.values()) or 1
    print(f"  {'-' * 92}")
    print("  keys used   : " + ", ".join(
        f"{k} {v * 100 / total_keys:.0f}%" for k, v in keys.most_common(12)) or "  keys used   : none")
    if dropped:
        print("  keys dropped: " + ", ".join(f"{k} x{v}" for k, v in dropped.most_common(10))
              + "   (not bound in Doom -- wasted chunks)")
    print("  terminations: " + ", ".join(f"{k} x{v}" for k, v in terms.most_common()))
    overall = statistics.mean(list(rows.values())) if rows else 0.0
    print(f"  OVERALL (mean of {len(rows)} map means) = {overall:.3f}")
    print("=========================================================\n")

    if args.markdown:
        cells = " | ".join(f"{rows.get(g, float('nan')):.2f}" for g in SUITE_ORDER)
        print("| model | battle1 | battle2 | defend_center | defend_line | OVERALL | n/map | history | sees | prompt | clock |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        print(f"| `{args.served}` | {cells} | **{overall:.2f}** | "
              f"{len(all_ds) // max(1, len(rows))} | {params.get('history_len')} | "
              f"{params.get('resize')} | {params.get('prompt_style')} | "
              f"{params.get('clock_mode')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
