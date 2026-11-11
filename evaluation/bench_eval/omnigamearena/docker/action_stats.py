#!/usr/bin/env python3
"""Standardized stats for an OmniGameArena <|action_start|>-format eval.

Usage:
  # filter by agent.model name under runs/yourvla
  python3 docker/action_stats.py --model <ckpt_name>
  python3 docker/action_stats.py --model <ckpt_name> --markdown \
      --date 2026-07-25 --idx 1 --think-mode 'open <think>'

Reports: overall mean score + per-game means (7 SOLO games), mouse histogram
(0 / 1-9 / 10-99 / 100+ on norm |dx|/|dy|), key distribution (IDLE + multi-key
+ singles), think/format/mouse/key rates matching the results-table columns.

Score source: result.json metrics.game.score (same as eval_summary_*.json).
Action source: summary.json step info.vlm_response.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from oga_craft_score import solo_craft_score  # noqa: E402

GAMES = [
    "obstacle_run_3d",
    "obstacle_run_2d",
    "last_stand",
    "monster_shoot",
    "scene_escape",
    "cue_chase",
    "solo_craft",
]
MOUSE_BUCKETS = (("0", 0, 0), ("1-9", 1, 9), ("10-99", 10, 99), ("100+", 100, 10**9))
_ACTION_TAG_RE = re.compile(
    r"<\|action_start\|[>}](.*?)<\|action_end\|[>}]", re.DOTALL,
)
_DEFAULT_RUNS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "omnigamearena", "runs", "yourvla",
)


def pct(n, d) -> str:
    return f"{100.0 * n / max(d, 1):.1f}%"


def mouse_bucket(m: float) -> str:
    for name, lo, hi in MOUSE_BUCKETS:
        if lo <= m <= hi:
            return name
    return "100+"


def split_think(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    if "</think>" in text:
        pre, _, post = text.rpartition("</think>")
        reasoning = pre.replace("<think>", "").strip()
        return reasoning, post.strip()
    if "<think>" in text:
        _, _, rest = text.partition("<think>")
        return rest.strip(), ""
    return "", text.strip()


def parse_norm_action(text: str):
    """Parse action body -> (ok, norm_dx, norm_dy, norm_dz, key_groups).

    Uses per-mille mouse (training units), not pixels. ok=False on no payload.
    """
    if not text:
        return False, 0.0, 0.0, 0.0, []
    m = _ACTION_TAG_RE.search(text)
    if m:
        body = m.group(1)
    elif re.match(r"^\s*-?\d", text) and ";" in text:
        body = text
    else:
        return False, 0.0, 0.0, 0.0, []
    parts = [seg.strip() for seg in body.split(";")]
    if not parts or parts[0] == "":
        return False, 0.0, 0.0, 0.0, []
    xyz = parts[0].split()
    try:
        nx = float(xyz[0]) if len(xyz) > 0 else 0.0
        ny = float(xyz[1]) if len(xyz) > 1 else 0.0
        nz = float(xyz[2]) if len(xyz) > 2 else 0.0
    except ValueError:
        return False, 0.0, 0.0, 0.0, []
    groups = [g.split() if g else [] for g in parts[1:]]
    return True, nx, ny, nz, groups


def vlm_text(step: dict) -> str:
    info = step.get("info") or {}
    t = info.get("vlm_response")
    if isinstance(t, str) and t.strip():
        return t
    # fallback: action JSON chunk_text
    raw = step.get("action")
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return obj.get("chunk_text") or ""
        except Exception:
            return raw
    if isinstance(raw, dict):
        return raw.get("chunk_text") or ""
    return ""


def collect(runs_root: str, model: str):
    by_game = defaultdict(list)  # game -> [score, ...]
    run_dirs = []  # dirs belonging to this model
    temps = Counter()
    reason_flags = Counter()
    dates = []

    for rj in glob.glob(os.path.join(runs_root, "**", "result.json"), recursive=True):
        try:
            d = json.load(open(rj))
        except Exception:
            continue
        m = (d.get("agent") or {}).get("model")
        if m != model:
            continue
        game = d.get("game") or os.path.relpath(rj, runs_root).split(os.sep)[0]
        metrics = d.get("metrics") or {}
        score = (metrics.get("game") or {}).get("score")
        if score is None:
            score = (d.get("terminal_info") or {}).get("score")
        if score is None:
            score = d.get("score")
        score = solo_craft_score(game, score, metrics.get("game"))
        by_game[game].append(score)
        run_dirs.append(os.path.dirname(rj))
        params = d.get("parameters") or {}
        if params.get("temperature") is not None:
            temps[str(params["temperature"])] += 1
        extra = ((d.get("agent") or {}).get("extra") or {})
        if "reasoning" in extra:
            reason_flags[bool(extra["reasoning"])] += 1
        ts = d.get("timestamp") or ""
        if len(ts) >= 8 and ts[:8].isdigit():
            dates.append(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}")

    # overall mean over numeric scores
    nums = [s for scores in by_game.values() for s in scores if isinstance(s, (int, float))]
    overall = (sum(nums) / len(nums)) if nums else None

    per_game = {}
    for g in GAMES:
        scores = by_game.get(g, [])
        gnums = [s for s in scores if isinstance(s, (int, float))]
        per_game[g] = {
            "n": len(scores),
            "mean": (sum(gnums) / len(gnums)) if gnums else None,
            "scores": scores,
        }
    for g, scores in by_game.items():
        if g not in per_game:
            gnums = [s for s in scores if isinstance(s, (int, float))]
            per_game[g] = {
                "n": len(scores),
                "mean": (sum(gnums) / len(gnums)) if gnums else None,
                "scores": scores,
            }

    # step-level action / think stats from summary.json
    steps = parse_ok = think_nonempty = mouse_nz = key_active = idle = multi = 0
    keys = Counter()
    mag = Counter()
    for rd in run_dirs:
        sf = os.path.join(rd, "summary.json")
        if not os.path.isfile(sf):
            continue
        try:
            recs = json.load(open(sf))
        except Exception:
            continue
        if not isinstance(recs, list):
            continue
        for step in recs:
            text = vlm_text(step)
            steps += 1
            reasoning, _ = split_think(text)
            if reasoning:
                think_nonempty += 1
            ok, dx, dy, dz, groups = parse_norm_action(text)
            if not ok:
                continue
            parse_ok += 1
            toks = [t for g in groups for t in g]
            uniq = set(toks)
            m = max(abs(dx), abs(dy))
            mag[mouse_bucket(m)] += 1
            if m > 0:
                mouse_nz += 1
            if toks:
                key_active += 1
            if not toks and dx == 0 and dy == 0 and dz == 0:
                idle += 1
            if len(uniq) >= 2:
                multi += 1
            for k in uniq:
                keys[k] += 1

    temp = temps.most_common(1)[0][0] if temps else "0.9"
    # infer think mode hint from agent.extra.reasoning if present
    if reason_flags.get(True, 0) > reason_flags.get(False, 0):
        think_hint = "open <think>"
    elif steps and think_nonempty == 0:
        think_hint = "none"
    else:
        think_hint = ""

    date_hint = max(dates) if dates else ""

    return {
        "model": model,
        "runs_root": runs_root,
        "n_episodes": sum(v["n"] for v in per_game.values()),
        "n_scored": len(nums),
        "overall": overall,
        "per_game": per_game,
        "temp": temp,
        "think_hint": think_hint,
        "date_hint": date_hint,
        "steps": steps,
        "parse_ok": parse_ok,
        "think_nonempty": think_nonempty,
        "mouse_nz": mouse_nz,
        "key_active": key_active,
        "idle": idle,
        "multi": multi,
        "keys": keys,
        "mag": mag,
    }


def fmt_mean(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


def print_human(s):
    print(f"### OMNIGAMEARENA SOLO — model={s['model']}")
    print(
        f"  episodes={s['n_episodes']} scored={s['n_scored']}  "
        f"overall_mean={fmt_mean(s['overall'])}  temp={s['temp']}"
    )
    print("  by game (mean score):")
    for g in GAMES:
        d = s["per_game"].get(g) or {"n": 0, "mean": None, "scores": []}
        print(f"    {g:18} n={d['n']:3d}  mean={fmt_mean(d['mean'])}  scores={d['scores']}")

    print(f"\n### STEP RATES  (raw_steps={s['steps']} parse_ok={s['parse_ok']})")
    print(f"  P(think non-empty)  {pct(s['think_nonempty'], s['steps'])}")
    print(f"  P(action parsed)    {pct(s['parse_ok'], s['steps'])}")
    print(f"  P(mouse≠0)          {pct(s['mouse_nz'], s['parse_ok'])}")
    print(f"  P(key non-idle)     {pct(s['key_active'], s['parse_ok'])}")

    print("\n### Mouse displacement histogram max(|norm_dx|,|norm_dy|)")
    for name, _, _ in MOUSE_BUCKETS:
        print(f"  {name:8}: {pct(s['mag'].get(name, 0), s['parse_ok'])}")

    print("\n### Key distribution (incl. IDLE / multi-key)")
    print(f"  {'(IDLE)':12}: {pct(s['idle'], s['parse_ok'])}")
    print(f"  {'multi-key':12}: {pct(s['multi'], s['parse_ok'])}")
    for k, n in s["keys"].most_common(20):
        print(f"  {k:12}: {pct(n, s['parse_ok'])}")


def print_markdown(s, *, model="", date="", think_mode="", temp="", idx="N", short=""):
    model = model or s["model"]
    date = date or s["date_hint"] or "YYYY-MM-DD"
    think_mode = think_mode or s["think_hint"] or "none"
    temp = temp or s["temp"]
    label = short or s["model"]
    overall = fmt_mean(s["overall"])
    row = (
        f"| {idx} | {date} | {model} | {overall} | {temp} | {think_mode} | "
        f"{pct(s['think_nonempty'], s['steps'])} | {pct(s['parse_ok'], s['steps'])} | "
        f"{pct(s['mouse_nz'], s['parse_ok'])} | {pct(s['key_active'], s['parse_ok'])} |"
    )
    print("<!-- ===== paste into summary table ===== -->")
    print(row)
    print()
    print(f"### #{idx} · {date} · {label}")
    print()
    print(f"- **Model (full name)**: `{model}`")
    print(f"- **Results**: `omnigamearena/runs/yourvla/*` (agent.model=`{s['model']}`)")
    print(f"- **Config**: temp={temp} · think={think_mode} · clock=pdq · episodes/game≈3")
    print(f"- **Overall score**: overall_mean={overall}  (scored {s['n_scored']}/{s['n_episodes']} eps)")
    print()
    print("#### Per-game scores, 7 games (mean / n / scores)")
    print()
    print("| game | n | mean | scores |")
    print("|---|---:|---:|---|")
    for g in GAMES:
        d = s["per_game"].get(g) or {"n": 0, "mean": None, "scores": []}
        sc = ", ".join("null" if x is None else str(x) for x in d["scores"])
        print(f"| {g} | {d['n']} | {fmt_mean(d['mean'])} | [{sc}] |")
    print()
    print("#### Mouse displacement histogram `max(|norm_dx|,|norm_dy|)`")
    print()
    print("| bucket | share |")
    print("|---|---:|")
    labels = {"0": "0 (still)", "1-9": "1–9", "10-99": "10–99", "100+": "100+"}
    for name, _, _ in MOUSE_BUCKETS:
        print(f"| {labels[name]} | {pct(s['mag'].get(name, 0), s['parse_ok'])} |")
    print()
    print("#### Key distribution (incl. IDLE / multi-key)")
    print()
    print("| key | share of steps |")
    print("|---|---:|")
    print(f"| (IDLE) | {pct(s['idle'], s['parse_ok'])} |")
    print(f"| multi-key | {pct(s['multi'], s['parse_ok'])} |")
    for k, n in s["keys"].most_common(20):
        print(f"| {k} | {pct(n, s['parse_ok'])} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="agent.model name in result.json")
    ap.add_argument("--runs-root", default=_DEFAULT_RUNS)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--date", default="")
    ap.add_argument("--think-mode", default="",
                    help='e.g. \'open <think>\' / \'closed <think>\\n\\n</think>\' / none')
    ap.add_argument("--temp", default="")
    ap.add_argument("--idx", default="N")
    ap.add_argument("--short", default="")
    ap.add_argument("--full-model-name", default="",
                    help="override display name in markdown (default = --model)")
    args = ap.parse_args()

    s = collect(args.runs_root, args.model)
    if s["n_episodes"] == 0:
        print(f"[warn] no result.json with agent.model={args.model!r} under {args.runs_root}",
              file=sys.stderr)
    if args.markdown:
        print_markdown(
            s,
            model=args.full_model_name or args.model,
            date=args.date,
            think_mode=args.think_mode,
            temp=args.temp,
            idx=args.idx,
            short=args.short,
        )
    else:
        print_human(s)


if __name__ == "__main__":
    main()
