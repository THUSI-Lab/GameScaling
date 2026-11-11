#!/usr/bin/env python3
"""Validity gate: was this run actually able to measure the model?

usage: oga_check_render.py RUNS_ROOT CKPT_ID

Two independent ways a run produces a full, plausible-looking score table while
measuring nothing: the engine renders too slowly for a held action to move the
character (below), or an episode never starts at all and reports the previous
episode's leftover score (EVAL_FLOW 7b-bis -- a map reload that did not finish,
visible as an episode with 1 step and zero model calls).

An action chunk is held for a fixed wall-clock 6x33ms, so if UE5's render thread
is starved the character barely moves while the keys are down. The agent is then
scored on a game it could not play, and the run says more about the GPU than the
model -- silently, because nothing crashes and every episode still reports a
score. This happened for real: 7 UE5 instances packed onto 2 GPUs produced
OVERALL 0.08 for a checkpoint that scores 0.17 with one instance per GPU.

The metric is p90 of the horizontal displacement over steps whose six sub-steps
all hold W. Median displacement is not usable: a weak model walks into walls, so
a low median can mean either a slow engine or a bad policy. The upper envelope is
close to the engine's character speed cap whenever the render keeps up, which
makes it a property of the environment rather than of the policy.

Thresholds below were measured on 2026-08-14 over 9 runs that had one UE5 per
GPU (all sharded runs plus the 8-GPU parallel run). Per-game p90 minima across
those runs were last_stand 35.0, scene_escape 31.5, monster_shoot 32.9,
obstacle_run_3d 30.6; the packed run scored 24.3 / 24.8 / 14.2 / 2.7. cue_chase
and solo_craft are excluded: movement is not their dominant action, so their p90
spreads too wide (cue_chase ranged 9.9-97.1 across healthy runs) to gate on.

Warns rather than fails: a false positive must not discard an hour of GPU time.
"""
import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

MIN_P90 = 28.0
GATE_GAMES = ("last_stand", "monster_shoot", "obstacle_run_3d", "scene_escape")
MIN_SAMPLES = 15


def sub_steps(chunk_text):
    """Return the per-sub-step key sets of an action chunk, or None."""
    m = re.search(r"<\|action_start\|>(.*?)<\|action_end\|>", chunk_text, re.S)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(";")]
    if len(parts) < 2:
        return None
    return [set(p.split()) for p in parts[1:]]      # parts[0] is the mouse delta


def action_keys(action):
    """Policy runs log ``{"chunk_text": ...}``; API (lumine) runs log the bare
    ``dx dy ; keys ; ...`` string, one group per ``chunk_steps`` sub-step."""
    try:
        text = json.loads(action or "{}").get("chunk_text", "")
    except (ValueError, AttributeError):
        text = f"<|action_start|>{action}<|action_end|>"
    return sub_steps(text)


def forward_displacements(steps_jsonl, n_sub=6, hold_s=0.2):
    """Horizontal distance moved on steps that hold W for every sub-step, rescaled
    to a 200 ms hold so the floor calibrated on 6x33 ms chunks applies to API
    runs too (lumine: 5 x 120 ms = 600 ms per chunk)."""
    out = []
    prev = None
    for line in open(steps_jsonl):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        pos = rec.get("position")
        if not pos:
            prev = None                             # gap: distance would be bogus
            continue
        here = (pos["x"], pos["y"])
        keys = action_keys(rec.get("action"))
        if prev is not None and keys and len(keys) == n_sub and all("W" in k for k in keys):
            out.append(math.dist(prev, here) * 0.2 / hold_s)
        prev = here
    return out


def p90(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(0.9 * len(values)))]


def main():
    root, ckpt = sys.argv[1], sys.argv[2]
    by_game = defaultdict(list)
    dead = []
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
            res = json.load(open(rj))
            if (res.get("agent") or {}).get("model") != ckpt:
                continue
        except Exception:
            continue
        game = os.path.relpath(rj, root).split(os.sep)[0]
        metrics = res.get("metrics") or {}
        n_steps = metrics.get("steps")
        n_calls = (metrics.get("latency") or {}).get("n")
        if n_steps is not None and n_steps <= 1 and not n_calls:
            dead.append((game, os.path.basename(os.path.dirname(rj)),
                         (metrics.get("game") or {}).get("score")))
        steps = os.path.join(os.path.dirname(rj), "reflection_trace", "steps.jsonl")
        if os.path.exists(steps):
            params = res.get("parameters") or {}
            n_sub = params.get("chunk_steps") or 6
            hold_s = n_sub * params["hold_duration"] if params.get("chunk_steps") else 0.2
            by_game[game] += forward_displacements(steps, n_sub, hold_s)

    if dead:
        print(f"\n  !! DEAD EPISODES ({len(dead)}): started and ended without one model call")
        for game, ts, score in dead:
            print(f"  !!   {game:18} {ts}  reported score={score}")
        print("  !! The map reload did not finish, so the first frame was the previous")
        print("  !! episode's and the score is leftover, not earned. Discard these and")
        print("  !! rerun the game with a longer MAP_WAIT (EVAL_FLOW 7b-bis).")

    if not by_game:
        print("[render-check] no position traces found; skipped")
        return

    suspect = []
    print(f"\n--------- RENDER HEALTH ({ckpt}) ---------")
    print("  forward-hold displacement (6x W), p90 vs engine-speed floor")
    for game in sorted(by_game):
        vals = by_game[game]
        gated = game in GATE_GAMES and len(vals) >= MIN_SAMPLES
        if len(vals) < MIN_SAMPLES:
            print(f"  {game:18} n={len(vals):4d} (too few full-hold steps to judge)")
            continue
        val = p90(vals)
        if gated and val < MIN_P90:
            suspect.append((game, val))
            mark = f"SUSPECT (< {MIN_P90:.0f})"
        else:
            mark = "ok" if gated else "not gated"
        print(f"  {game:18} n={len(vals):4d} med={st.median(vals):6.1f} p90={val:6.1f}  {mark}")

    if suspect:
        print("\n  !! RENDER-STARVED: " + ", ".join(f"{g} p90={v:.1f}" for g, v in suspect))
        print("  !! The engine ran too slowly for a held action to move the character,")
        print("  !! so these scores measure the GPU, not the checkpoint. Do not report")
        print("  !! them. Check that no two UE5 instances shared a GPU and that the")
        print("  !! node is not on the bad-render list, then rerun.")
    else:
        print("\n  render health OK -- scores are comparable to previous runs")
    print("------------------------------------------\n")


if __name__ == "__main__":
    main()
