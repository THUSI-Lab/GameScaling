#!/usr/bin/env python3
"""Aggregate one checkpoint's PvP / Coop matches into a table + summary json.

usage: oga_aggregate_2p.py RUNS_ROOT CKPT_ID

Reads ``match_result.json`` (one per match) rather than the per-player
``result.json`` files, because the match level is where the two sides mean
something together:

  pvp  -- both sides are the same checkpoint in self-play, so the win rate is
          ~50% by construction and carries no signal. The table reports the
          pooled per-player score (n = 2 x matches) and keeps the winner split
          only as a sanity check that the sides are symmetric.

**Read the `start` column before reading the score.** The two families run in
opposite directions and UE5 reports both through the same `get_score`:

  start 1.0 (crystal_guard, sky_duel) -- the score is what is LEFT of your own
      crystal / your own HP, and only ever falls (0.1 = one 10 HP hit; a 1.0 ->
      0.0 jump in sky_duel is a fall off the platform). A completely passive
      agent therefore scores 1.0, so a HIGH score here means little damage was
      landed, not that the model played well. The offensive reading is the
      opponent's loss: damage dealt = 1 - opponent score, and pooled over both
      self-play sides that is exactly 1 - the score column.
  start 0.0 (the CookHouse maps) -- the score accumulates deliveries, so higher
      is better in the usual sense.
  coop -- the pair is the unit. ``coop_score_aggregation`` decides whether the
          team score is the sum of both players or the shared reading taken
          once, and ``match_result.coop_total_score`` already applies it (the
          runner deletes any ``team_score`` key before writing); counting the
          two players separately would double-count a shared score.

Never mix these numbers with the solo table: split-screen halves the frame
(observed 960x270), so the same checkpoint sees a different world here.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def _first_step_score(match_dir, label):
    """Step 0's score, which is what tells the two score conventions apart.

    Rebuilt from the match dir rather than the recorded ``run_dir``: that field
    holds the pod-side absolute path, which does not exist wherever the tree is
    read back from.
    """
    if not label:
        return None
    player_dir = os.path.join(match_dir, str(label))
    try:
        with open(os.path.join(player_dir, "summary.json")) as fh:
            records = json.load(fh)
        return (records[0].get("info") or {}).get("score")
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root, ckpt = sys.argv[1], sys.argv[2]

    per_game = defaultdict(lambda: {
        "matches": 0, "player_scores": [], "team_scores": [], "starts": [],
        "winners": Counter(), "status": Counter(), "steps": [], "mode": "",
    })

    for mj in glob.glob(os.path.join(root, "**", "match_result.json"), recursive=True):
        try:
            d = json.load(open(mj))
        except Exception:
            continue
        players = (d.get("player_results") or {}).values()
        models = {(p.get("model") or "") for p in players}
        if ckpt not in models:
            continue
        game = os.path.relpath(mj, root).split(os.sep)[0]
        row = per_game[game]
        row["mode"] = d.get("mode") or row["mode"]
        row["matches"] += 1
        row["status"][d.get("status") or "?"] += 1
        if d.get("winner"):
            row["winners"][d["winner"]] += 1
        if d.get("coop_total_score") is not None:
            row["team_scores"].append(d["coop_total_score"])
        for p in players:
            row["player_scores"].append(p.get("score"))
            row["steps"].append(p.get("steps"))
            row["starts"].append(
                _first_step_score(os.path.dirname(mj), p.get("player_label")))

    if not per_game:
        print(f"\n[2p-agg] no PvP/Coop matches found for {ckpt} under {root}")
        return

    summary = {"checkpoint": ckpt, "games": {}}
    print(f"\n============= TWO-PLAYER RESULTS ({ckpt}) =============")
    print(f"{'game':<16}{'mode':<6}{'n':>3}{'start':>6}{'player mean':>13}"
          f"{'team mean':>11}{'steps':>8}  status / winners")
    for g in sorted(per_game):
        r = per_game[g]
        pm, tm, sm = (_mean(r["player_scores"]), _mean(r["team_scores"]),
                      _mean(r["starts"]))
        st = ",".join(f"{k}={v}" for k, v in sorted(r["status"].items()))
        wn = ",".join(f"{k}={v}" for k, v in sorted(r["winners"].items()))
        print(f"{g:<16}{r['mode']:<6}{r['matches']:>3}"
              f"{(sm if sm is not None else float('nan')):>6.1f}"
              f"{(pm if pm is not None else float('nan')):>13.4f}"
              f"{(tm if tm is not None else float('nan')):>11.4f}"
              f"{(_mean(r['steps']) or 0):>8.0f}  {st}  {wn}")
        summary["games"][g] = {
            "mode": r["mode"], "matches": r["matches"],
            "start_score": sm,
            "player_mean_score": pm, "team_mean_score": tm,
            "mean_steps": _mean(r["steps"]),
            "status": dict(r["status"]), "winners": dict(r["winners"]),
        }
    print("  start=1.0 -> score is your own crystal/HP left, falls only; a passive "
          "agent scores 1.0 and damage dealt = 1 - score")
    print("  start=0.0 -> score accumulates deliveries; higher is better")

    dst = os.path.join(os.path.dirname(os.path.normpath(root)),
                       f"eval_summary_2p_{ckpt}.json")
    try:
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"  summary -> {dst}")
    except Exception as exc:  # noqa: BLE001
        print(f"  summary write failed: {exc}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
