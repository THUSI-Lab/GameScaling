#!/usr/bin/env python3
"""Balance (map, episode) units over workers by estimated length, longest first.

usage: vzd_plan_workers.py NWORKERS MAP[:EPISODES] ...
       vzd_plan_workers.py 6 battle1:3 battle2:3 defend_the_center:3 defend_the_line:3

Prints one TSV line per worker: ``worker_index<TAB>unit unit ...<TAB>est_minutes``.

Round-robin would be worse than it looks: a `defend_the_*` episode runs to the
2100-tic timeout (~12 min) while a `battle*` episode usually ends when the agent
dies (~6 min), so slicing the list in order can leave one worker holding three
timeouts (36 min) while another finishes in 18 and idles. LPT (longest
processing time first) bounds the makespan at 4/3 of optimal.
"""
from __future__ import annotations

import os
import sys

# The driver runs this from a /tmp snapshot of docker/ (so that editing the repo
# mid-job cannot make a live run resume mid-line), which puts the harness package
# out of reach of a path relative to __file__. The driver exports VZD_ROOT.
_ROOT = os.environ.get("VZD_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
try:
    from vizdoom_arena.games import EST_MINUTES, get_game  # noqa: E402
except ModuleNotFoundError as exc:                          # pragma: no cover
    raise SystemExit(
        f"cannot import vizdoom_arena from {_ROOT!r}: {exc}. "
        f"Set VZD_ROOT to the evaluation/vizdoom directory."
    ) from None


def plan(nworkers: int, units: list[tuple[str, int]]) -> tuple[list[list[str]], list[float]]:
    nworkers = max(1, int(nworkers))
    buckets: list[list[str]] = [[] for _ in range(nworkers)]
    loads = [0.0] * nworkers
    ordered = sorted(units, key=lambda u: -EST_MINUTES.get(u[0], 8.0))
    for game, ep in ordered:
        w = min(range(nworkers), key=lambda i: loads[i])
        buckets[w].append(f"{game}:{ep}")
        loads[w] += EST_MINUTES.get(game, 8.0)
    return buckets, loads


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    nworkers = int(sys.argv[1])
    units: list[tuple[str, int]] = []
    for token in sys.argv[2:]:
        name, _, eps = token.partition(":")
        get_game(name)                       # fail fast on a typo
        for ep in range(int(eps or 1)):
            units.append((name, ep))
    buckets, loads = plan(nworkers, units)
    for i, (bucket, load) in enumerate(zip(buckets, loads)):
        print(f"{i}\t{' '.join(bucket)}\t{load:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
