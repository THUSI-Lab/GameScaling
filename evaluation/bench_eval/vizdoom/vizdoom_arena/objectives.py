"""Objective sets for B4-human runs (`--objective-set`), read from configs/objectives.json."""

from __future__ import annotations

import json
import os

DEFAULT_OBJECTIVES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "objectives.json")


def load_objectives(path: str, name: str, games: list[str]) -> dict[str, list[str]]:
    """{map: [text, ...]}; a unit uses texts[episode % len(texts)]."""
    with open(path, encoding="utf-8") as f:
        sets = json.load(f)
    if name not in sets or name.startswith("_"):
        known = ", ".join(k for k in sets if not k.startswith("_"))
        raise SystemExit(f"unknown objective set {name!r} in {path}; known: {known}")
    out: dict[str, list[str]] = {}
    for game in games:
        text = sets[name].get(game)
        if not text:
            raise SystemExit(f"objective set {name!r} has no entry for {game}")
        out[game] = [text] if isinstance(text, str) else list(text)
    return out
