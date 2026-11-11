#!/usr/bin/env python3
"""Aggregate one checkpoint's SOLO episode scores into a table + summary json.

usage: oga_aggregate.py RUNS_ROOT CKPT_ID

RUNS_ROOT is the benchmark output root (…/omnigamearena/runs/yourvla). Only
result.json files whose agent.model equals CKPT_ID are counted, so several
checkpoints can share the tree.
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from oga_craft_score import solo_craft_score  # noqa: E402


def main():
    root, ckpt = sys.argv[1], sys.argv[2]
    rows = []
    # Batch jobs use flat_output directories carrying the served model name.
    # Opt in explicitly; legacy callers still inspect the complete result tree.
    tag = os.environ.get("OGA_RUN_DIR_TAG")
    if tag:
        if tag != ckpt:
            raise ValueError("OGA_RUN_DIR_TAG must equal the requested checkpoint")
        candidates = glob.glob(os.path.join(root, "*", "*" + glob.escape(tag) + "*", "result.json"))
    else:
        candidates = glob.glob(os.path.join(root, "**", "result.json"), recursive=True)
    for rj in candidates:
        try:
            d = json.load(open(rj))
        except Exception:
            continue
        if (d.get("agent") or {}).get("model") != ckpt:
            continue
        # PvP/Coop write one result.json per player under the match dir. They
        # carry the same agent.model, so without this they would land here as
        # two extra "episodes" of a game whose score means something else
        # entirely (a coop pair would also be counted twice instead of summed).
        # oga_aggregate_2p.py reports those. Solo results have no `mode` key.
        if d.get("mode") in {"pvp", "coop"}:
            continue
        game = os.path.relpath(rj, root).split(os.sep)[0]
        m = d.get("metrics") or {}
        score = (m.get("game") or {}).get("score")      # scores live in metrics.game.score
        if score is None:
            score = (d.get("terminal_info") or {}).get("score")
        if score is None:
            score = d.get("score")
        score = solo_craft_score(game, score, m.get("game"))
        rows.append((game, score))

    by = defaultdict(list)
    for g, s in rows:
        by[g].append(s)

    summary = {"checkpoint": ckpt, "games": {}}
    alls = []
    print(f"\n================= EVAL RESULTS ({ckpt}) =================")
    for g in sorted(by):
        nums = [s for s in by[g] if isinstance(s, (int, float))]
        alls += nums
        mean = sum(nums) / len(nums) if nums else None
        summary["games"][g] = {"n": len(by[g]), "mean": mean, "scores": by[g]}
        print(f"  {g:18} n={len(by[g])} mean={mean}")
    overall = sum(alls) / len(alls) if alls else None
    summary["overall_mean"] = overall
    print(f"  OVERALL mean_score = {overall}  ({len(alls)} scored episodes)")

    out = os.path.abspath(os.path.join(root, "..", "..", "..", f"eval_summary_{ckpt}.json"))
    try:
        json.dump(summary, open(out, "w"), indent=2)
        print(f"  summary -> {out}")
    except Exception as exc:
        print(f"  (summary write failed: {exc})")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
