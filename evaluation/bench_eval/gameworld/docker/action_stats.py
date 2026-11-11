#!/usr/bin/env python3
"""Standardized stats for an <|action_start|>-format eval run.

Usage:
  python3 docker/action_stats.py results/<run_root>
  python3 docker/action_stats.py results/<run_root> --markdown
      # emit a summary-table row + detail section for a results table

Reports: game performance (overall + by genre), mouse histogram
(0 / 1-9 / 10-99 / 100+), key distribution (IDLE + multi-key + singles),
think/format/mouse/key rates matching the results-table columns.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

GW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GW_ROOT))
from agents.mm_agents.qwen_vl.sft_action_parser import parse_sft_action as P  # noqa

GENRE = {
    "runner": [
        "04_boxel-rebound", "07_chrome-dino", "09_cubefield", "11_edge-surf",
        "13_flappy-bird", "21_ns-shaft", "26_run-3", "28_temple-run-2",
    ],
    "arcade": [
        "05_breakout", "08_core-ball", "15_google-snake", "23_pacman",
        "25_rocket-league-2d", "33_worlds-hardest-game", "34_worlds-hardest-game-2",
    ],
    "platformer": [
        "02_another-gentlemans-adventure", "06_captaincallisto", "10_doodle-jump",
        "14_geodash", "17_mario-game", "22_ovo", "24_restless-wing-syndrome", "30_vex-3",
    ],
    "puzzle": [
        "01_2048", "03_astray", "16_hextris", "19_minesweeper",
        "27_stack", "29_tetris", "32_wordle",
    ],
    "simulation": [
        "12_fireboy-and-watergirl", "18_minecraft-clone-glm",
        "20_monkey-mart", "31_wolf3d",
    ],
}
GAME_TO_GENRE = {g: genre for genre, games in GENRE.items() for g in games}
MOUSE_BUCKETS = (("0", 0, 0), ("1-9", 1, 9), ("10-99", 10, 99), ("100+", 100, 10**9))


def content(rec):
    rr = (rec.get("output") or {}).get("raw_response", "")
    try:
        return json.loads(rr)["choices"][0]["message"]["content"]
    except Exception:
        m = re.search(r"<\|action_start\|>.*?<\|action_end\|>", rr, re.DOTALL)
        return m.group(0) if m else rr


def split_think(text: str) -> tuple[str, str]:
    """Return (reasoning, action_tail). Empty reasoning if no/empty think block."""
    if not text:
        return "", ""
    if "</think>" in text:
        pre, _, post = text.rpartition("</think>")
        reasoning = pre.replace("<think>", "").strip()
        return reasoning, post.strip()
    if "<think>" in text:
        # opened but never closed — treat remainder after open as reasoning attempt
        _, _, rest = text.partition("<think>")
        return rest.strip(), ""
    return "", text.strip()


def mouse_bucket(m: float) -> str:
    for name, lo, hi in MOUSE_BUCKETS:
        if lo <= m <= hi:
            return name
    return "100+"


def pct(n, d) -> str:
    return f"{100.0 * n / max(d, 1):.1f}%"


def collect(root: str):
    # ---- game performance ----
    byg = defaultdict(lambda: [0, 0, []])  # success, n, progresses
    runs_csvs = glob.glob(f"{root}/**/runs.csv", recursive=True)
    for cf in runs_csvs:
        for r in csv.DictReader(open(cf)):
            g = r.get("game_id") or "?"
            e = byg[g]
            e[1] += 1
            if r.get("final_status") == "success":
                e[0] += 1
            p = r.get("progress")
            if p not in (None, "", "None"):
                try:
                    e[2].append(float(p))
                except ValueError:
                    pass

    succ = sum(v[0] for v in byg.values())
    ntot = sum(v[1] for v in byg.values())
    allp = [p for v in byg.values() for p in v[2]]
    mean_pg = (sum(allp) / len(allp)) if allp else 0.0

    by_genre = {}
    for genre, games in GENRE.items():
        gs = [byg[g] for g in games if g in byg]
        s = sum(v[0] for v in gs)
        n = sum(v[1] for v in gs)
        ps = [p for v in gs for p in v[2]]
        by_genre[genre] = {
            "n": n,
            "sr": (s / n) if n else 0.0,
            "pg": (sum(ps) / len(ps)) if ps else 0.0,
            "succ": s,
        }

    # ---- step-level action / think stats ----
    steps = 0
    parse_ok = 0
    think_nonempty = 0
    think_closed = 0
    has_action_start = 0
    has_action_end = 0
    has_decision = 0
    finish_length = 0
    mouse_nz = 0
    key_active = 0
    idle = 0
    multi = 0
    keys = Counter()
    mag = Counter()
    think_lens: list[int] = []
    # brevity buckets (chars of non-empty reasoning)
    think_buckets = Counter()

    paths = set(glob.glob(f"{root}/**/agent_0/interactions.jsonl", recursive=True))
    for cf in runs_csvs:
        for row in csv.DictReader(open(cf)):
            run_dir_text = row.get("run_dir")
            if not run_dir_text:
                continue
            run_dir = Path(run_dir_text)
            if not run_dir.is_absolute():
                run_dir = GW_ROOT / run_dir
            interaction_path = run_dir / "agent_0" / "interactions.jsonl"
            if interaction_path.is_file():
                paths.add(str(interaction_path))
    for f in sorted(paths):
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            steps += 1
            output = rec.get("output") or {}
            parsed_action = output.get("parsed_action")
            if isinstance(parsed_action, dict) and parsed_action.get("tool_name"):
                arguments = parsed_action.get("arguments") or {}
                executed = output.get("executed_action") or {}
                reasoning = str(
                    arguments.get("reasoning") or executed.get("reasoning") or ""
                ).strip()
                if reasoning:
                    think_nonempty += 1
                    nch = len(reasoning)
                    think_lens.append(nch)
                    if nch <= 50:
                        think_buckets["1-50"] += 1
                    elif nch <= 200:
                        think_buckets["51-200"] += 1
                    elif nch <= 500:
                        think_buckets["201-500"] += 1
                    else:
                        think_buckets["501+"] += 1
                else:
                    think_buckets["0"] += 1

                validity = output.get("action_validity") or {}
                if validity.get("is_valid") is True:
                    parse_ok += 1
                action_kind = str(executed.get("action") or "")
                key = executed.get("key")
                if key:
                    key_active += 1
                    keys[str(key)] += 1
                mouse_values = [
                    executed.get(name)
                    for name in ("dx", "dy", "x", "y")
                    if isinstance(executed.get(name), int | float)
                ]
                mouse_magnitude = max((abs(float(value)) for value in mouse_values), default=0)
                mag[mouse_bucket(mouse_magnitude)] += 1
                if mouse_magnitude > 0:
                    mouse_nz += 1
                if action_kind == "wait":
                    idle += 1
                continue

            raw_response = str(output.get("raw_response") or "")
            validity = output.get("action_validity") or {}
            is_api_action_string = (
                raw_response.startswith("ChatCompletion(")
                and validity.get("agent_type") == "computer_use"
            )
            if is_api_action_string:
                if "<|action_start|>" in raw_response:
                    has_action_start += 1
                if "<|action_end|>" in raw_response:
                    has_action_end += 1
                if "finish_reason='length'" in raw_response:
                    finish_length += 1

                reasoning, _tail = split_think(raw_response)
                if reasoning:
                    think_nonempty += 1
                    nch = len(reasoning)
                    think_lens.append(nch)
                    if nch <= 50:
                        think_buckets["1-50"] += 1
                    elif nch <= 200:
                        think_buckets["51-200"] += 1
                    elif nch <= 500:
                        think_buckets["201-500"] += 1
                    else:
                        think_buckets["501+"] += 1
                else:
                    think_buckets["0"] += 1

                if not isinstance(parsed_action, dict) or validity.get("is_valid") is not True:
                    continue
                parse_ok += 1
                executed = output.get("executed_action") or parsed_action
                slots = executed.get("slots") or []
                tokens = [
                    str(item)
                    for slot in slots
                    for item in slot
                    if not str(item).startswith("MOUSE:")
                ]
                mouse_tokens = [
                    str(item)
                    for slot in slots
                    for item in slot
                    if str(item).startswith("MOUSE:")
                ]
                unique_keys = set(tokens)
                if tokens:
                    key_active += 1
                if len(unique_keys) >= 2:
                    multi += 1
                for key in unique_keys:
                    keys[key] += 1

                mouse = executed.get("mouse") or {}
                try:
                    norm_dx = (float(mouse["x"]) - float(mouse["from_x"])) / 1280.0 * 1000.0
                    norm_dy = (float(mouse["y"]) - float(mouse["from_y"])) / 720.0 * 1000.0
                    mouse_magnitude = max(abs(norm_dx), abs(norm_dy))
                except (KeyError, TypeError, ValueError):
                    mouse_magnitude = 0.0
                mag[mouse_bucket(mouse_magnitude)] += 1
                if mouse_magnitude > 0:
                    mouse_nz += 1
                if not tokens and not mouse_tokens and mouse_magnitude == 0:
                    idle += 1
                continue

            raw = content(rec)
            reasoning, _tail = split_think(raw)
            if reasoning:
                think_nonempty += 1
                nch = len(reasoning)
                think_lens.append(nch)
                if nch <= 50:
                    think_buckets["1-50"] += 1
                elif nch <= 200:
                    think_buckets["51-200"] += 1
                elif nch <= 500:
                    think_buckets["201-500"] += 1
                else:
                    think_buckets["501+"] += 1
            else:
                think_buckets["0"] += 1
            if "</think>" in (raw or ""):
                think_closed += 1
            if "<|action_start|>" in (raw or ""):
                has_action_start += 1
            if "<|action_end|>" in (raw or ""):
                has_action_end += 1
            if re.search(r"<Decision(?:Making)?>", raw or "", re.I):
                has_decision += 1
            # finish_reason=length from raw_response JSON if present
            rr = (rec.get("output") or {}).get("raw_response", "")
            try:
                fr = json.loads(rr)["choices"][0].get("finish_reason")
                if fr == "length":
                    finish_length += 1
            except Exception:
                pass
            a = P(raw)
            if a is None:
                continue
            parse_ok += 1
            dx, dy, dz = a["norm_dx"], a["norm_dy"], a["norm_dz"]
            toks = [t for s in a["substeps"] for t in s]
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

    def _pctile(vals: list[int], q: float) -> float:
        if not vals:
            return 0.0
        xs = sorted(vals)
        i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        return float(xs[i])

    return {
        "root": root,
        "byg": dict(byg),
        "succ": succ,
        "ntot": ntot,
        "mean_pg": mean_pg,
        "by_genre": by_genre,
        "steps": steps,
        "parse_ok": parse_ok,
        "think_nonempty": think_nonempty,
        "think_closed": think_closed,
        "has_action_start": has_action_start,
        "has_action_end": has_action_end,
        "has_decision": has_decision,
        "finish_length": finish_length,
        "think_lens": think_lens,
        "think_len_mean": (sum(think_lens) / len(think_lens)) if think_lens else 0.0,
        "think_len_p50": _pctile(think_lens, 0.50),
        "think_len_p90": _pctile(think_lens, 0.90),
        "think_buckets": think_buckets,
        "mouse_nz": mouse_nz,
        "key_active": key_active,
        "idle": idle,
        "multi": multi,
        "keys": keys,
        "mag": mag,
    }


def print_human(s):
    print(f"### GAME PERFORMANCE — {s['root']}")
    print(
        f"  success {s['succ']}/{s['ntot']} = {s['succ']/max(s['ntot'],1):.4f}   "
        f"mean_progress {s['mean_pg']:.3f}"
    )
    print("  by genre (SR / PG):")
    for g in ("runner", "arcade", "platformer", "puzzle", "simulation"):
        d = s["by_genre"][g]
        print(f"    {g:12} n={d['n']:3d}  SR={d['sr']:.3f}  PG={d['pg']:.3f}")

    print(f"\n### STEP RATES  (raw_steps={s['steps']} parse_ok={s['parse_ok']})")
    print(f"  P(think non-empty)  {pct(s['think_nonempty'], s['steps'])}")
    print(f"  P(action parsed)    {pct(s['parse_ok'], s['steps'])}")
    print(f"  P(mouse≠0)          {pct(s['mouse_nz'], s['parse_ok'])}")
    print(f"  P(key non-idle)     {pct(s['key_active'], s['parse_ok'])}")

    print("\n### THINK conciseness (chars; non-empty reasoning only)")
    print(f"  closed </think>  {pct(s['think_closed'], s['steps'])}")
    print(
        f"  len mean/p50/p90  {s['think_len_mean']:.0f} / "
        f"{s['think_len_p50']:.0f} / {s['think_len_p90']:.0f}"
    )
    for b in ("0", "1-50", "51-200", "201-500", "501+"):
        print(f"  bucket {b:8}: {pct(s['think_buckets'].get(b, 0), s['steps'])}")

    print("\n### Native format compliance (literal occurrence under noskip)")
    print(f"  P(<|action_start|>)  {pct(s['has_action_start'], s['steps'])}")
    print(f"  P(<|action_end|>)    {pct(s['has_action_end'], s['steps'])}")
    print(f"  P(<Decision*>)       {pct(s['has_decision'], s['steps'])}")
    print(f"  P(finish=length)     {pct(s['finish_length'], s['steps'])}")

    print("\n### Mouse displacement histogram max(|dx|,|dy|)")
    for name, _, _ in MOUSE_BUCKETS:
        print(f"  {name:8}: {pct(s['mag'].get(name, 0), s['parse_ok'])}")

    print("\n### Key distribution (incl. IDLE / multi-key)")
    print(f"  {'(IDLE)':12}: {pct(s['idle'], s['parse_ok'])}")
    print(f"  {'multi-key':12}: {pct(s['multi'], s['parse_ok'])}")
    for k, n in s["keys"].most_common(20):
        print(f"  {k:12}: {pct(n, s['parse_ok'])}")


def print_markdown(s, *, model="", date="", think_mode="", temp="1.0", idx="N", short=""):
    root = s["root"].rstrip("/")
    label = short or os.path.basename(root)
    model = model or f"(fill in full model name · results={label})"
    date = date or "YYYY-MM-DD"
    think_mode = think_mode or "none"
    sr = f"{100.0 * s['succ'] / max(s['ntot'], 1):.1f}%"
    pg = f"{s['mean_pg']:.3f}"
    row = (
        f"| {idx} | {date} | {model} | {root} | {sr} / {pg} | {temp} | {think_mode} | "
        f"{pct(s['think_nonempty'], s['steps'])} | {pct(s['parse_ok'], s['steps'])} | "
        f"{pct(s['mouse_nz'], s['parse_ok'])} | {pct(s['key_active'], s['parse_ok'])} |"
    )
    print("<!-- ===== paste into summary table ===== -->")
    print(row)
    print()
    print(f"### #{idx} · {date} · {label}")
    print()
    print(f"- **Model (full name)**: {model}")
    print(f"- **Results directory**: `{root}`")
    print(f"- **Config**: temp={temp} · think={think_mode}")
    print(f"- **Overall score**: SR={sr} / PG={pg}  ({s['succ']}/{s['ntot']})")
    print(
        f"- **think conciseness**: closed={pct(s['think_closed'], s['steps'])}; "
        f"len mean/p50/p90={s['think_len_mean']:.0f}/"
        f"{s['think_len_p50']:.0f}/{s['think_len_p90']:.0f} chars"
    )
    print(
        f"- **Native format**: action_start={pct(s['has_action_start'], s['steps'])}; "
        f"action_end={pct(s['has_action_end'], s['steps'])}; "
        f"Decision*={pct(s['has_decision'], s['steps'])}; "
        f"finish=length={pct(s['finish_length'], s['steps'])}"
    )
    print()
    print("#### Per-genre scores, 5 genres (SR / PG)")
    print()
    print("| genre | n | SR | PG |")
    print("|---|---:|---:|---:|")
    for g in ("runner", "arcade", "platformer", "puzzle", "simulation"):
        d = s["by_genre"][g]
        print(f"| {g} | {d['n']} | {100.0*d['sr']:.1f}% | {d['pg']:.3f} |")
    print()
    print("#### Mouse displacement histogram `max(|dx|,|dy|)`")
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
    ap.add_argument("root", nargs="?", default="results/global_step_800_merged")
    ap.add_argument("--markdown", action="store_true", help="emit Markdown snippets for a results table")
    ap.add_argument("--model", default="", help="full model name for markdown row")
    ap.add_argument("--date", default="", help="YYYY-MM-DD")
    ap.add_argument("--think-mode", default="",
                    help='e.g. \'open <think>\' / \'closed <think>\\n\\n</think>\' / none')
    ap.add_argument("--temp", default="1.0")
    ap.add_argument("--idx", default="N")
    ap.add_argument("--short", default="")
    args = ap.parse_args()

    s = collect(args.root)
    if args.markdown:
        print_markdown(
            s,
            model=args.model,
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
