#!/usr/bin/env python3
"""Run a SINGLE rollout in its own process, driven by a JSON args payload on stdin.

run_eval.py launches this in a fresh session (its own process group) so that a
hung Minecraft instance can be killed with os.killpg — something a ThreadPool
worker thread can never do. The rollout writes success.json / loss.json itself;
this wrapper only reconstructs the args Namespace and calls rollout().
"""
import json
import os
import sys
from types import SimpleNamespace

# Force THIS checkout onto sys.path so the editable-installed openagents from an
# older checkout can't shadow it (see openha-editable-install-shadowing).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.rollout_openha import rollout  # noqa: E402


def main():
    payload = json.load(sys.stdin)
    args = SimpleNamespace(**payload)
    rollout(args)


if __name__ == "__main__":
    main()
