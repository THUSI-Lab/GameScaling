"""Translate GameScaling keyboard/mouse chunks into PS5 controller (DualSense) input.

The released checkpoints act in the 74-token keyboard/mouse vocabulary (``gamescaling.action``). A
:class:`ControllerProfile` maps that vocabulary onto a controller so the same models can play games that only
accept a gamepad:

- movement keys (default ``W A S D``) -> left stick, evaluated per 33 ms step;
- mouse movement ``X Y`` (per-mille of the screen over 200 ms) -> right stick, held for the whole chunk;
- other tokens -> controller buttons (``LB`` -> ``R2`` fire, ``RB`` -> ``L2`` aim, ``Space`` -> ``Cross``, ...);
- scroll ``Z`` -> optional button taps.

Profiles are plain data, so a game-specific layout is a small JSON file (see :meth:`ControllerProfile.from_json`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..action import CHUNK_STEPS, KEY_VOCAB, Action

# DualSense button names, shared by every backend.
BUTTONS: Tuple[str, ...] = (
    "Cross", "Circle", "Square", "Triangle",
    "L1", "R1", "L2", "R2", "L3", "R3",
    "Up", "Down", "Left", "Right",
    "Create", "Options", "PS", "Touchpad",
)

# A typical console shooter / third-person layout. Override per game with a JSON profile.
DEFAULT_BUTTONS: Dict[str, str] = {
    "LB": "R2", "RB": "L2", "MB": "R3",          # fire / aim / melee
    "Space": "Cross", "Ctrl": "Circle", "C": "Circle",
    "R": "Square", "E": "Triangle", "F": "Triangle",
    "Q": "L1", "G": "R1",
    "Shift": "L3", "V": "R3",
    "Tab": "Touchpad", "M": "Touchpad", "Esc": "Options", "Enter": "Cross",
    "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
    "one": "Up", "two": "Right", "three": "Down", "four": "Left",
}
DEFAULT_MOVE_KEYS: Dict[str, Tuple[float, float]] = {"W": (0.0, -1.0), "S": (0.0, 1.0),
                                                     "A": (-1.0, 0.0), "D": (1.0, 0.0)}


@dataclass
class ControllerStep:
    """Controller state for one 33 ms step. Sticks are -1..1 with +x right and +y down; triggers 0..1."""

    left: Tuple[float, float] = (0.0, 0.0)
    right: Tuple[float, float] = (0.0, 0.0)
    l2: float = 0.0
    r2: float = 0.0
    buttons: frozenset = frozenset()


@dataclass
class ControllerProfile:
    name: str = "default"
    buttons: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BUTTONS))
    move_keys: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_MOVE_KEYS))
    # Mouse movement (per-mille of the screen per 200 ms chunk) that maps to full right-stick deflection.
    # Smaller = more sensitive camera. Tune per game together with its in-game stick sensitivity.
    mouse_full_scale: float = 150.0
    mouse_deadzone: float = 3.0       # per-mille; movements this small or smaller leave the right stick centred
    invert_y: bool = False
    scroll_up: Optional[str] = None   # button tapped in step 1 when Z > 0 (e.g. "Triangle" to switch weapon)
    scroll_down: Optional[str] = None

    def __post_init__(self) -> None:
        bad = {k: v for k, v in self.buttons.items() if v not in BUTTONS}
        if bad:
            raise ValueError(f"profile {self.name!r}: unknown controller buttons {bad}; choose from {BUTTONS}")
        unknown = [k for k in list(self.buttons) + list(self.move_keys) if k not in KEY_VOCAB]
        if unknown:
            raise ValueError(f"profile {self.name!r}: {unknown} are not GameScaling key tokens")
        for b in (self.scroll_up, self.scroll_down):
            if b is not None and b not in BUTTONS:
                raise ValueError(f"profile {self.name!r}: unknown scroll button {b!r}")
        self.move_keys = {k: (float(v[0]), float(v[1])) for k, v in self.move_keys.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ControllerProfile":
        """Keys missing from ``d`` keep their defaults; ``"buttons"`` entries are merged into the defaults
        unless ``"replace_buttons": true``. Map a token to ``null`` to unbind it."""
        d = dict(d)
        buttons = {} if d.pop("replace_buttons", False) else dict(DEFAULT_BUTTONS)
        for k, v in (d.pop("buttons", None) or {}).items():
            if v is None:
                buttons.pop(k, None)
            else:
                buttons[k] = v
        move = d.pop("move_keys", None)
        return cls(buttons=buttons, move_keys=move if move is not None else dict(DEFAULT_MOVE_KEYS), **d)

    @classmethod
    def from_json(cls, path: str) -> "ControllerProfile":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _stick_from_mouse(x: float, y: float, p: ControllerProfile) -> Tuple[float, float]:
    def axis(v: float) -> float:
        if abs(v) <= p.mouse_deadzone:
            return 0.0
        return max(-1.0, min(1.0, v / p.mouse_full_scale))
    ry = axis(y)
    return axis(x), (-ry if p.invert_y else ry)


def _stick_from_keys(keys, p: ControllerProfile) -> Tuple[float, float]:
    x = sum(p.move_keys[k][0] for k in keys if k in p.move_keys)
    y = sum(p.move_keys[k][1] for k in keys if k in p.move_keys)
    n = math.hypot(x, y)
    if n > 1.0:  # diagonals: unit length, like a real stick pushed to the rim
        x, y = x / n, y / n
    return (x, y)


def translate(action: Action, profile: ControllerProfile) -> Tuple[List[ControllerStep], List[str]]:
    """``Action`` -> 6 :class:`ControllerStep` plus the tokens this profile does not bind.

    An unparsed action (``action.parsed is False``) becomes six neutral steps, i.e. a no-op.
    """
    if not action.parsed:
        return [ControllerStep() for _ in range(CHUNK_STEPS)], []
    x, y, z = action.mouse
    right = _stick_from_mouse(x, y, profile)
    unmapped: List[str] = []
    steps: List[ControllerStep] = []
    for i, group in enumerate(action.steps[:CHUNK_STEPS]):
        buttons = set()
        for tok in group:
            if tok in profile.move_keys:
                continue
            b = profile.buttons.get(tok)
            if b is None:
                if tok not in unmapped:
                    unmapped.append(tok)
            else:
                buttons.add(b)
        if i == 0 and z > 0 and profile.scroll_up:
            buttons.add(profile.scroll_up)
        if i == 0 and z < 0 and profile.scroll_down:
            buttons.add(profile.scroll_down)
        steps.append(ControllerStep(
            left=_stick_from_keys(group, profile), right=right,
            l2=1.0 if "L2" in buttons else 0.0, r2=1.0 if "R2" in buttons else 0.0,
            buttons=frozenset(buttons)))
    while len(steps) < CHUNK_STEPS:
        steps.append(ControllerStep(right=right))
    return steps, unmapped
