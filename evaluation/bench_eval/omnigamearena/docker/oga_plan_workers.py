#!/usr/bin/env python3
"""Assign games to UE5 workers, longest job first.

usage: oga_plan_workers.py NWORK GAME [GAME ...]
out:   one line per worker: "<worker index>\\t<game> <game> ..."

When there are fewer UE5 instances than games (the 3 vLLM + 5 UE5 layout), a
worker runs its games one after another, so the wall clock is the slowest
worker rather than the slowest game. Round-robin by list index used to decide
this, which could put two 30-minute games on one worker while another worker
took a 30-second game and then idled. Greedy longest-processing-time-first
keeps the three heavy games on their own workers and packs the short ones
together; for the standard suite that is within a few percent of optimal.

Costs are seconds per episode, measured 2026-08-14 on the 8-GPU run (steps per
episode x per-step latency with an uncontended vLLM). They only need to be
right in relative terms, and only for ordering. An unlisted game is assumed to
be as expensive as the worst known one so that it lands on its own worker
instead of silently doubling someone's runtime.
"""
import sys

COST = {
    # cue_chase / scene_escape are 500 on the reference suite (official VLM
    # cap) and 300 on solo_suite. Planning with 500 is the conservative order:
    # a 300-step run then finishes early instead of stacking two long games.
    "cue_chase": 500 * 2.16,
    "monster_shoot": 300 * 2.15,
    "scene_escape": 500 * 2.01,
    "last_stand": 88 * 1.72,
    "obstacle_run_3d": 27 * 1.29,
    "solo_craft": 15 * 1.79,
    "obstacle_run_2d": 13 * 1.03,
}
UNKNOWN_COST = max(COST.values())


def plan(nwork, games):
    nwork = max(1, min(nwork, len(games)))
    buckets = [[] for _ in range(nwork)]
    load = [0.0] * nwork
    for game in sorted(games, key=lambda g: -COST.get(g, UNKNOWN_COST)):
        i = min(range(nwork), key=lambda k: load[k])
        buckets[i].append(game)
        load[i] += COST.get(game, UNKNOWN_COST)
    return buckets, load


def main():
    nwork = int(sys.argv[1])
    games = sys.argv[2:]
    if not games:
        raise SystemExit("no games given")
    buckets, load = plan(nwork, games)
    for i, (games_i, secs) in enumerate(zip(buckets, load)):
        # est is per episode and uncontended, so it under-reads a real run; it is
        # here to make an unbalanced plan visible in the log, not as a promise.
        print(f"{i}\t{' '.join(games_i)}\t{secs / 60:.1f}")


if __name__ == "__main__":
    main()
