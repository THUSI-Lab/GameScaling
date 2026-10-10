"""The GameScaling action format: parse model text into one 200 ms keyboard/mouse chunk.

Every turn the model emits exactly one chunk:

    <|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>

- ``X Y`` is the mouse displacement over the whole 200 ms, in per-mille of the
  screen (``X = sum(dx) / width * 1000``, clamped to +-1000; X>0 right, Y>0 down).
- ``Z`` is the scroll amount in wheel clicks (clamped to +-5; Z>0 up).
- ``g1..g6`` are six 33 ms substeps. Each group lists the keys/buttons that are
  held during that substep (at most 4). A key present in consecutive groups is
  held continuously; an empty group releases everything.

Reasoning models may prefix the chunk with ``<think>...</think>``. The action is
always read from the text *after* the last ``</think>``: thinking models quote
the format back at themselves ("the format is <|action_start|>X Y Z ; ..."), and
reading the whole reply would execute the quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ACTION_START = "<|action_start|>"
ACTION_END = "<|action_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

CHUNK_MS = 200.0
CHUNK_STEPS = 6
STEP_MS = CHUNK_MS / CHUNK_STEPS
MAX_KEYS_PER_STEP = 4
MOUSE_LIMIT = 1000.0
SCROLL_LIMIT = 5.0

NOOP_ACTION_TEXT = f"{ACTION_START}0 0 0 ; ; ; ; ; ; {ACTION_END}"

# The 74-token vocabulary the model was trained on. Case matters:
# `one`..`nine` are number-row keys, `One`..`Twelve` are F1..F12.
LETTER_KEYS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGIT_KEYS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
FUNCTION_KEYS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                 "Nine", "Ten", "Eleven", "Twelve")
MOUSE_BUTTON_KEYS = ("LB", "RB", "MB")
ARROW_KEYS = ("Up", "Down", "Left", "Right")
SPECIAL_KEYS = ("Shift", "Ctrl", "Alt", "Tab", "Caps", "Esc", "Space", "Enter", "Back",
                "Delete", "Insert", "Home", "End", "Pause")
PUNCTUATION_KEYS = ("Equal", "Minus", "Period", "Slash", "Quote")
KEY_VOCAB: frozenset[str] = frozenset(
    LETTER_KEYS + DIGIT_KEYS + FUNCTION_KEYS + MOUSE_BUTTON_KEYS + ARROW_KEYS
    + SPECIAL_KEYS + PUNCTUATION_KEYS
)

# `}` instead of `>` is a closing-bracket typo some models make.
_TAG_RE = re.compile(r"<\|action_start\|[>}](.*?)<\|action_end\|[>}]", re.DOTALL)
_BARE_RE = re.compile(r"^\s*-?\d")
# Lenient fallbacks (see `parse_action(lenient=True)`).
_WRAPPED_INT_RE = re.compile(
    r"<\|action(?:_start)?\|>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)(.*?)<\|action(?:_end)?\|>",
    re.DOTALL,
)
_SKELETON_RE = re.compile(r"[^\n;<]*(?:;[^\n;<]*){5,}")
_BARE_INT_RE = re.compile(r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*(;[^\n<]*)", re.DOTALL)
_INT_RE = re.compile(r"^-?\d+$")


def split_reasoning(text: str) -> tuple[str, str, bool]:
    """``text`` -> (reasoning, text after the reasoning, reasoning was closed).

    An unclosed ``<think>`` means generation ended before any action existed
    (the token budget ran out, or a stop string matched inside the reasoning),
    so the tail is empty.
    """
    text = text or ""
    if THINK_CLOSE in text:
        head, _, tail = text.rpartition(THINK_CLOSE)
        return head.split(THINK_OPEN)[-1].strip(), tail, True
    if THINK_OPEN in text:
        return text.split(THINK_OPEN)[-1].strip(), "", False
    return "", text, True


@dataclass
class Action:
    """One parsed 200 ms chunk. Unparseable text is a no-op with ``parsed=False``."""

    mouse: tuple[float, float, float] = (0.0, 0.0, 0.0)
    steps: list[list[str]] = field(default_factory=lambda: [[] for _ in range(CHUNK_STEPS)])
    parsed: bool = False
    truncated: bool = False
    body: str = ""
    dropped: list[str] = field(default_factory=list)

    @property
    def history_text(self) -> str:
        """What this turn looks like as an assistant message in the next prompt.

        Always a well-formed chunk: an unparseable turn is stored as an explicit
        no-op, because putting the model's own failure into context teaches it
        to repeat the failure.
        """
        if not self.parsed:
            return NOOP_ACTION_TEXT
        return f"{ACTION_START}{self.body}{ACTION_END}"

    @property
    def is_noop(self) -> bool:
        return not any(self.steps) and self.mouse == (0.0, 0.0, 0.0)

    @property
    def pressed(self) -> set[str]:
        return {k for group in self.steps for k in group}

    def to_text(self) -> str:
        x, y, z = (int(round(v)) for v in self.mouse)
        groups = " ; ".join(" ".join(g) for g in self.steps)
        return f"{ACTION_START}{x} {y} {z} ; {groups}{ACTION_END}"


def _build(nums: list[float], groups: list[str], body: str) -> Action:
    x = max(-MOUSE_LIMIT, min(MOUSE_LIMIT, nums[0]))
    y = max(-MOUSE_LIMIT, min(MOUSE_LIMIT, nums[1]))
    z = max(-SCROLL_LIMIT, min(SCROLL_LIMIT, nums[2]))
    steps: list[list[str]] = []
    dropped: list[str] = []
    for group in groups[:CHUNK_STEPS]:
        keys = []
        for tok in group.split():
            (keys if tok in KEY_VOCAB else dropped).append(tok)
        steps.append(keys[:MAX_KEYS_PER_STEP])
    while len(steps) < CHUNK_STEPS:
        steps.append([])
    return Action(mouse=(x, y, z), steps=steps, parsed=True, body=body, dropped=dropped)


def _parse_strict(tail: str) -> Action:
    m = _TAG_RE.search(tail)
    if m:
        body = m.group(1)
    elif _BARE_RE.match(tail) and ";" in tail:
        # Tags are special tokens; a server that strips them returns a bare body.
        body = tail.strip()
    else:
        return Action()
    parts = [seg.strip() for seg in body.split(";")]
    if not parts or not parts[0]:
        return Action()
    nums: list[float] = []
    for tok in parts[0].split()[:3]:
        try:
            nums.append(float(tok))
        except ValueError:
            return Action()
    if len(nums) < 3:
        return Action()
    return _build(nums, parts[1:], body)


def _parse_lenient(text: str) -> Action:
    """Recovery for malformed replies (see `parse_action`)."""
    text = text.replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")
    m = _WRAPPED_INT_RE.search(text)
    if m:
        nums = [float(m.group(i)) for i in (1, 2, 3)]
        key_part = m.group(4).strip()
        body = text[m.start():m.end()]
        body = body[body.find(">") + 1:body.rfind("<")]
    else:
        skel = None
        for sm in _SKELETON_RE.finditer(text):
            skel = sm.group(0)
        if skel is None:
            bm = _BARE_INT_RE.search(text)
            if not bm:
                return Action()
            nums = [float(bm.group(i)) for i in (1, 2, 3)]
            key_part = bm.group(4).strip()
            body = bm.group(0)
        else:
            head = skel.partition(";")[0]
            head_tokens = head.split()
            if head_tokens and len(head_tokens) <= 3 and all(_INT_RE.match(t) for t in head_tokens):
                nums = [float(t) for t in head_tokens] + [0.0, 0.0, 0.0]
                key_part = skel[len(head):]
            else:
                nums = [0.0, 0.0, 0.0]
                key_part = skel
            body = skel
    segments = key_part.split(";") if key_part else []
    if segments and segments[0].strip() == "":
        segments = segments[1:]
    return _build(nums[:3], segments, body.strip())


def parse_action(text: str, *, lenient: bool = False) -> Action:
    """Raw model reply -> :class:`Action`. Never raises; failures are no-ops.

    ``lenient=False`` (default): only a chunk after the
    last ``</think>`` counts, either tagged or as a bare ``X Y Z ; ...`` body.

    ``lenient=True``: additionally recover shapes a
    well-trained checkpoint rarely emits -- an action inside an unclosed
    ``<think>``, a 2-number mouse prefix, a keys-only skeleton -- by locating the
    ``;``-separated six-group skeleton anywhere in the reply.
    """
    _, tail, closed = split_reasoning(text)
    if closed:
        action = _parse_strict(tail)
        if action.parsed or not lenient:
            return action
    elif not lenient:
        return Action(truncated=True)
    action = _parse_lenient(text or "")
    if not closed and not action.parsed:
        action.truncated = True
    return action
