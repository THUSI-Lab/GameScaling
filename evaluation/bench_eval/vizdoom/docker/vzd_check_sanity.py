#!/usr/bin/env python3
"""Validity gate: refuse to let a structurally void run be registered as a score.

usage: vzd_check_sanity.py RUNS_ROOT SERVED_NAME [--episodes N] [--maps "a b c"]

This is the ViZDoom counterpart of OmniGameArena's `oga_check_render.py`, and it
exists for the same reason: every failure mode below produces a complete,
plausible-looking results table. Nothing crashes, no episode errors out, and the
aggregator happily prints four map means -- of a run that measured the harness,
the serving stack or the prompt instead of the model.

  UNCLOSED-THINK >20% of turns ended while still inside `<think>`, so the model
                 never reached an action. Reported before UNPARSED because it is
                 the same turns seen one layer down, and the fix is a serving
                 knob rather than the policy: `finish_reason=length` means
                 `reasoning_max_tokens` is too small, anything else means a stop
                 sequence matched inside the reasoning (a thinking model quoting
                 `<|action_end|>` at itself while planning ends its own turn).
  UNPARSED       >30% of turns carried no `<|action_start|>`. Those turns became
                 no-op chunks, so the score is a floor imposed by the output
                 format, not by the policy. Usually the wrong reasoning style
                 for the checkpoint (a THINKING ckpt answering with <Decision>).
  NO-AIM         the mouse was zero on >95% of turns. Doom is aimed with the
                 mouse; an agent that only walks scores ~0 on every map and the
                 four means will look like a uniformly weak model.
  NO-FIRE        a map where no episode ever pressed fire. Score is 0 by
                 construction.
                 Both of the above are only blocking once an episode has enough
                 decisions to make "never" mean something (see
                 MIN_STEPS_FOR_BEHAVIOUR); on a short smoke they are notes.
  DEAD EPISODES  an episode with <=1 decision. It still writes a result.json and
                 still contributes a score to the mean. Only checked under `pdq`:
                 under `lcrt` the decision latency is charged to the game clock,
                 so a 1-decision episode is a real (very short) episode.
  DROPPED KEYS   >15% of steps named a key that is not bound, so the press was
                 silently discarded. Only counts BARE names (`RB`): a real key
                 with a stray special token glued to it (`LB<|endoftext|>`) is a
                 serialization artifact, reported as a note instead.
  SERVER ERRORS  >5% of turns lost to endpoint failures; those are no-ops too.
  MISSING        fewer episodes than requested -- a worker died and the mean is
                 over whatever survived, silently at a smaller n.

Exit code 0 = clean, 1 = at least one blocking finding.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

UNPARSED_MAX = 0.30
NOAIM_MIN_ZERO = 0.95
SERVER_ERROR_MAX = 0.05
UNCLOSED_THINK_MAX = 0.20
# Asking for a key the harness does not bind is a silent action loss: the press
# is dropped, no error is raised, and a full results table still comes out. A
# few percent is ordinary hallucination (every model does it); 27b_s1 asked for
# `RB` on 55% of its steps and scored below a blind spin script.
DROPPED_KEY_MAX = 0.15
# NO-AIM and NO-FIRE ask "did this never happen", which a 12-step smoke answers
# by chance: not firing for two seconds of game time is ordinary play. Below this
# many decisions they are printed as notes instead of blocking, so that a gate
# failure keeps meaning something. The rate-based findings stay blocking at any
# length -- an unparsed rate is just as real over 12 turns as over 350.
MIN_STEPS_FOR_BEHAVIOUR = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("served")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--maps", default=None)
    args = ap.parse_args()

    by: dict[str, list[dict]] = defaultdict(list)
    for rj in glob.glob(os.path.join(args.root, "**", "result.json"), recursive=True):
        try:
            d = json.load(open(rj))
        except (OSError, ValueError):
            continue
        if (d.get("agent") or {}).get("model") != args.served:
            continue
        by[d.get("game") or "?"].append(d)

    findings: list[str] = []
    notes: list[str] = []
    print(f"\n---------------- VALIDITY GATE ({args.served}) ----------------")

    if not by:
        print("  no results found for this model")
        print("  VERDICT: INVALID (nothing to check)")
        print("-------------------------------------------------------------\n")
        return 1

    expected_maps = args.maps.split() if args.maps else sorted(by)
    for game in expected_maps:
        ds = by.get(game) or []
        if not ds:
            findings.append(f"MISSING: {game} produced no episode at all")
            continue

        def mean(path: tuple[str, ...], data=ds) -> float:
            vals = []
            for d in data:
                cur = d.get("metrics") or {}
                for key in path:
                    cur = (cur or {}).get(key)
                if isinstance(cur, (int, float)):
                    vals.append(float(cur))
            return statistics.mean(vals) if vals else 0.0

        unparsed = mean(("action", "unparsed_rate"))
        unclosed = mean(("reasoning", "unclosed_think_rate"))
        why: dict[str, int] = defaultdict(int)
        for d in ds:
            reasons = ((d.get("metrics") or {}).get("reasoning") or {}).get(
                "unclosed_finish_reasons") or {}
            for reason, count in reasons.items():
                why[reason] += int(count)
        mouse_nz = mean(("action", "mouse_nonzero_rate"))
        fire = mean(("action", "attack_rate"))
        steps = mean(("steps",))
        errors = mean(("action", "server_errors"))
        # A 1-step episode means "the harness produced a score without the policy
        # ever really playing" only under pdq, where the world is frozen while the
        # model thinks and 350 decisions always fit. Under lcrt the thinking time
        # is charged back to the game, so at API latencies a whole 60s episode is
        # ~7 decisions and dying on the first one is a REAL outcome, not a bug.
        # Under rt the episode always spans the full 350 chunks but a slow model
        # gets fewer decisions inside them, so a low decision count is likewise
        # the measurement, not a failure.
        clocks = {(d.get("parameters") or {}).get("clock_mode", "pdq") for d in ds}
        realtime = clocks in ({"lcrt"}, {"rt"}, {"trt"})
        dead = ([] if realtime else
                [d for d in ds if (d.get("metrics") or {}).get("steps", 0) <= 1])
        dropped: dict[str, int] = defaultdict(int)
        for d in ds:
            for key, count in (((d.get("metrics") or {}).get("action") or {})
                               .get("dropped_tokens") or {}).items():
                dropped[key] += int(count)
        dropped_n = sum(dropped.values()) / max(len(ds), 1)

        rt_extra = ""
        if clocks in ({"rt"}, {"trt"}):
            # Under rt the score is only readable next to these two: how often
            # the policy got to change its mind, and how old the frame it acted
            # on was. A run at decision_rate 0.3 is not comparable to one at 1.0
            # even though both cover the same 60s of game time.
            rt_extra = (f" dec_rate={mean(('clock', 'decision_rate')):.2f}"
                        f" stale={mean(('clock', 'mean_staleness_ms')):.0f}ms")
        print(f"  {game:20} n={len(ds)} steps={steps:.0f} unparsed={unparsed:.2f} "
              f"unclosed={unclosed:.2f} mouse_nonzero={mouse_nz:.2f} "
              f"fire={fire:.2f} dead={len(dead)}{rt_extra}")

        if unclosed > UNCLOSED_THINK_MAX:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(why.items())) or "unknown"
            cure = ("raise --reasoning-max-tokens" if why.get("length")
                    else "a stop sequence fired inside <think>; unarm it")
            findings.append(
                f"UNCLOSED-THINK: {game} {unclosed:.0%} of turns ended mid-<think> "
                f"({reasons}) -- {cure}. The UNPARSED/NO-AIM/NO-FIRE findings "
                f"below are downstream of this")
        if unparsed > UNPARSED_MAX:
            findings.append(f"UNPARSED: {game} {unparsed:.0%} of turns had no action tag")
        # Two very different things end up in `dropped_tokens`, and only one of
        # them is a policy defect:
        #   `RB`                 -- the model wanted a button that does not exist
        #                           and fired into the void (gs27b_s1, 55% of
        #                           steps). Its intent is LOST, so the score is of
        #                           a crippled action space. Blocking.
        #   `LB<|endoftext|>`    -- a valid key with a stray special token glued
        #                           on, because the model emitted an EOS spelling
        #                           just before the correct `<|action_end|>`. The
        #                           intent is unambiguous and, measured on the
        #                           gemini-3.8-flash batch, 1619/1621 of these sit
        #                           in the LAST of the six 33ms groups, so the cost
        #                           is one key for the final tic of a 171ms chunk.
        #                           A note, with the number, not a verdict.
        glued = {k: v for k, v in dropped.items() if "<|" in k}
        bare = {k: v for k, v in dropped.items() if "<|" not in k}
        bare_n = sum(bare.values()) / max(len(ds), 1)
        glued_n = sum(glued.values()) / max(len(ds), 1)
        if steps and bare_n / steps > DROPPED_KEY_MAX:
            top = ", ".join(f"{k}={v}" for k, v in
                            sorted(bare.items(), key=lambda kv: -kv[1])[:3])
            findings.append(
                f"DROPPED KEYS: {game} asked for unbound keys on {bare_n/steps:.0%} "
                f"of steps ({top}). Those presses did nothing, so the score is of a "
                f"policy acting through a fraction of its intended action space. The "
                f"prompt's 'Key Naming Rules' block names LB/RB/MB and the whole "
                f"keyboard as SPELLINGS; only the per-game binding list is bound, so "
                f"this is the model misreading the prompt (it would do the same in "
                f"OGA) -- but the number belongs next to the score")
        if steps and glued_n / steps > DROPPED_KEY_MAX:
            top = ", ".join(f"{k}={v}" for k, v in
                            sorted(glued.items(), key=lambda kv: -kv[1])[:3])
            notes.append(
                f"{game} glued a stray special token onto a real key on "
                f"{glued_n/steps:.0%} of steps ({top}) -- an EOS spelling written "
                f"just before the correct <|action_end|>. Nearly all of these land "
                f"in the last of the six 33ms groups, so the cost is that one key "
                f"for the final tic; not blocking, but report it with the score")
        behaviour = findings if steps >= MIN_STEPS_FOR_BEHAVIOUR else notes
        if mouse_nz < (1.0 - NOAIM_MIN_ZERO):
            behaviour.append(f"NO-AIM: {game} moved the mouse on only {mouse_nz:.0%} of turns")
        if fire == 0.0:
            behaviour.append(f"NO-FIRE: {game} never pressed fire in any episode")
        if behaviour is notes and (mouse_nz < (1.0 - NOAIM_MIN_ZERO) or fire == 0.0):
            notes.append(f"{game} has only {steps:.0f} decisions, too few for "
                         f"NO-AIM/NO-FIRE to be evidence -- not blocking")
        if dead:
            findings.append(f"DEAD EPISODES: {game} has {len(dead)} episode(s) with <=1 step")
        if steps and errors / max(steps, 1.0) > SERVER_ERROR_MAX:
            findings.append(f"SERVER ERRORS: {game} lost {errors:.0f} turns/episode to the endpoint")
        if args.episodes and len(ds) < args.episodes:
            findings.append(f"MISSING: {game} has {len(ds)}/{args.episodes} episodes")

        # Not blocking, but worth printing: a map where every episode hit the
        # step budget behaves differently from one where the agent kept dying.
        terms = {d.get("termination") for d in ds}
        if terms:
            notes.append(f"{game} terminations={sorted(t for t in terms if t)}")

    for n in notes:
        print(f"  note: {n}")
    if findings:
        print("  " + "-" * 58)
        for f in findings:
            print(f"  !! {f}")
        print(f"  VERDICT: INVALID -- do not register these scores ({len(findings)} finding(s))")
    else:
        print("  VERDICT: OK")
    print("-------------------------------------------------------------\n")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
