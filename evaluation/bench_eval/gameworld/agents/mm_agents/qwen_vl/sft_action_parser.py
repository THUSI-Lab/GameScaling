"""Parse training-pipeline SFT action format into GameWorld canonical actions.

SFT format: <|action_start|>dx dy dz ; k1 k2 ; k3 ; ... ; k6<|action_end|>
  - dx, dy: relative mouse movement in 0-1000 normalized scale
  - dz: scroll amount (-5 to 5)
  - 6 substep groups separated by ';', each with space-separated key tokens

Converts to GameWorld canonical actions: click, press_key, press_keys,
mouse_move, scroll, wait, etc.
"""

from __future__ import annotations

import re
from typing import Optional

# Adaptive parsing of the action string across tokenizer variants:
#   1) WRAPPED  — older checkpoints emit the markers as plain text, so both
#      <|action_start|> ... <|action_end|> appear in the decoded content.
#   2) BARE     — newer checkpoints register the markers as *special tokens*, so
#      vLLM strips them on decode (skip_special_tokens=True) and the content
#      arrives as "dx dy dz ; g1 ; ... ; g6" with NO markers.
# We try WRAPPED first (anchored on the markers → safest, won't grab stray
# numbers from any prose), then fall back to BARE (anchored on "3 ints + ';'" so
# it only fires on a real action string). The BARE pattern also recovers the
# mixed case (start marker present but end stripped, or vice-versa).
_WRAPPED_RE = re.compile(
    r"<\|action(?:_start)?\|>\s*"
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    r"(.*?)"
    r"<\|action(?:_end)?\|>",
    re.DOTALL,
)
# A native /step stop can omit only the closing marker. Recognize that exact
# shape before the broad skeleton fallback, whose first group would otherwise
# contain the start-marker remnant and shift every key group by one slot. This
# requires all six groups and never supplies a marker, mouse value, or key.
_OPEN_WRAPPED_RE = re.compile(
    r"<\|action_start\|>\s*"
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*"
    r"((?:;[^;\n<]*){6})\s*\Z",
    re.DOTALL,
)
_BARE_RE = re.compile(
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*"
    r"(;[^\n<]*)",  # three ints immediately followed by the ';'-separated groups
    re.DOTALL,
)
# SKELETON: the action always ends in the 6-group key skeleton. Match a run
# carrying >=5 ';' with NO newline and NO '<'. Excluding '<' bounds the match
# against think tags (<think>/</think>) and any "<|action..." remnants; the
# newline bound separates it from prose reasoning on other lines. This one
# pattern captures every observed shape — with 3-int mouse prefix
# ("0 0 0 ; ; Right ; ..."), 2-int prefix ("-5 4 ; A ; ..."), keys-only
# ("LeftArrow ; LeftArrow ; ..."), and the inverted "action</think>" order —
# so prose can't match (reasoning almost never packs >=5 ';' on one line).
_SKELETON_RE = re.compile(r"[^\n;<]*(?:;[^\n;<]*){5,}")
_INT_RE = re.compile(r"^-?\d+$")

# training-pipeline key token → GameWorld Playwright key name
SFT_KEY_TO_GW: dict[str, str] = {
    "Left": "ArrowLeft",
    "Right": "ArrowRight",
    "Up": "ArrowUp",
    "Down": "ArrowDown",
    "LeftArrow": "ArrowLeft",
    "RightArrow": "ArrowRight",
    "UpArrow": "ArrowUp",
    "DownArrow": "ArrowDown",
    "Esc": "Escape",
    "Caps": "CapsLock",
    "Enter": "Enter",
    "Back": "Backspace",
    "Backspace": "Backspace",
    "Space": "Space",
    "Shift": "Shift",
    "Ctrl": "Control",
    "Alt": "Alt",
    "Tab": "Tab",
    # Number words → digit characters
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # F-key words → F1-F12
    "One": "F1", "Two": "F2", "Three": "F3", "Four": "F4",
    "Five": "F5", "Six": "F6", "Seven": "F7", "Eight": "F8",
    "Nine": "F9", "Ten": "F10", "Eleven": "F11", "Twelve": "F12",
}

# Mouse button tokens
MOUSE_BUTTONS = {"LB": "left", "RB": "right", "MB": "middle"}

SUBSTEP_DURATION_SEC = 0.033


def _map_key(token: str) -> str:
    """Map a single SFT key token to GameWorld key name."""
    mapped = SFT_KEY_TO_GW.get(token)
    if mapped:
        return mapped
    if len(token) == 1 and token.isalpha():
        return token.lower()
    return token


def parse_sft_action(raw_text: str) -> Optional[dict]:
    """Parse an action string into a structured dict, adaptively across tokenizer
    and prompt-format variants. Returns None if no action is found.

    Handled shapes:
      - WRAPPED: "<|action_start|>dx dy dz ; g1 ; ...<|action_end|>" (markers as
        text, old checkpoints).
      - BARE special-token: "dx dy dz ; g1 ; ..." (markers stripped by vLLM).
      - THINK models: "<think> reasoning </think> <action>", and the messy
        variants they actually emit — action never closing </think>, a 2-int
        (dx dy) mouse prefix, keys-only with no mouse prefix, or "action</think>"
        inverted order. All are recovered by locating the ';'-skeleton.
    """
    # Restrict the new open-window path to the decision after the last closed
    # thought. Keep all existing fallback behavior for unclosed thoughts.
    if "</think>" in raw_text:
        open_tail = raw_text.rsplit("</think>", 1)[1]
    elif "<think>" in raw_text:
        open_tail = ""
    else:
        open_tail = raw_text

    # Scrub the think TAGS (not their content) so the ';'-skeleton search is not
    # broken by the '<' in "</think>" gluing onto a mouse prefix, while still
    # keeping an action that was emitted *before* </think> (inverted order).
    # Replace with a space so adjacent tokens never fuse.
    raw_text = raw_text.replace("<think>", " ").replace("</think>", " ")

    # 1) WRAPPED text markers — safest, anchored on the literal markers.
    wm = _WRAPPED_RE.search(raw_text)
    if wm:
        dx, dy, dz = int(wm.group(1)), int(wm.group(2)), int(wm.group(3))
        key_part = wm.group(4).strip()
    elif (om := _OPEN_WRAPPED_RE.search(open_tail)) is not None:
        dx, dy, dz = int(om.group(1)), int(om.group(2)), int(om.group(3))
        key_part = om.group(4).strip()
    else:
        # 2) Locate the action by its ';'-skeleton (see _SKELETON_RE). Take the
        #    LAST match: the action follows any reasoning.
        skel = None
        for sm in _SKELETON_RE.finditer(raw_text):
            skel = sm.group(0)
        if skel is None:
            # 3) Last resort: a triple-int action with fewer than 5 ';' groups.
            bm = _BARE_RE.search(raw_text)
            if not bm:
                return None
            dx, dy, dz = int(bm.group(1)), int(bm.group(2)), int(bm.group(3))
            key_part = bm.group(4).strip()
        else:
            head, sep, _ = skel.partition(";")
            head_tokens = head.split()
            if head_tokens and len(head_tokens) <= 3 and all(_INT_RE.match(t) for t in head_tokens):
                # numeric mouse prefix: dx [dy [dz]]  (missing components -> 0)
                nums = [int(t) for t in head_tokens] + [0, 0, 0]
                dx, dy, dz = nums[0], nums[1], nums[2]
                key_part = skel[len(head):]  # keep leading ';' for group alignment
            else:
                # keys-only: the head is the first key group, mouse is zero
                dx = dy = dz = 0
                key_part = skel

    segments = key_part.split(";") if key_part else []
    # The action string is "x y z ; g1 ; g2 ; ... ; g6" (a ';' precedes EVERY key
    # group, per training action_trans), so splitting leaves a leading empty
    # segment (the mouse/key boundary). Drop it so g1..g6 align to substeps 0..5
    # (otherwise slot 0 is a phantom empty and g6 is truncated by [:6]).
    if segments and segments[0].strip() == "":
        segments = segments[1:]
    substeps: list[list[str]] = []
    for segment in segments[:6]:
        tokens = segment.strip().split()
        substeps.append([t for t in tokens if t.strip()])

    while len(substeps) < 6:
        substeps.append([])

    return {
        "norm_dx": dx,
        "norm_dy": dy,
        "norm_dz": dz,
        "substeps": substeps[:6],
    }


def sft_to_gameworld_action(
    sft_action: dict,
    cursor_x: float,
    cursor_y: float,
    image_w: int = 1280,
    image_h: int = 720,
) -> tuple[dict, float, float]:
    """Convert parsed SFT action to a GameWorld canonical action dict.

    Args:
        sft_action: Output of parse_sft_action().
        cursor_x: Current virtual cursor X in normalized 0-1000 space.
        cursor_y: Current virtual cursor Y in normalized 0-1000 space.
        image_w: Screenshot width in pixels.
        image_h: Screenshot height in pixels.

    Returns:
        (action_dict, new_cursor_x, new_cursor_y)
    """
    dx = sft_action["norm_dx"]
    dy = sft_action["norm_dy"]
    dz = sft_action["norm_dz"]
    substeps = sft_action["substeps"]

    new_cx = max(0.0, min(1000.0, cursor_x + dx))
    new_cy = max(0.0, min(1000.0, cursor_y + dy))

    # Collect mouse buttons and keyboard keys across all substeps
    mouse_btn: str | None = None
    mouse_hold_steps = 0
    keyboard_keys: dict[str, int] = {}  # key → count of substeps it appears

    for step_keys in substeps:
        step_mouse = None
        for token in step_keys:
            if token in MOUSE_BUTTONS:
                step_mouse = MOUSE_BUTTONS[token]
            else:
                gw_key = _map_key(token)
                keyboard_keys[gw_key] = keyboard_keys.get(gw_key, 0) + 1

        if step_mouse:
            if mouse_btn is None:
                mouse_btn = step_mouse
            if step_mouse == mouse_btn:
                mouse_hold_steps += 1

    # Priority: mouse click > scroll > keyboard > mouse_move > wait

    # 1. Mouse click
    if mouse_btn is not None:
        pixel_x = new_cx / 1000.0 * image_w
        pixel_y = new_cy / 1000.0 * image_h
        if mouse_hold_steps >= 4:
            return (
                {
                    "action": "click_hold",
                    "x": pixel_x,
                    "y": pixel_y,
                    "button": mouse_btn,
                    "duration": mouse_hold_steps * SUBSTEP_DURATION_SEC,
                },
                new_cx,
                new_cy,
            )
        else:
            return (
                {"action": "click", "x": pixel_x, "y": pixel_y, "button": mouse_btn},
                new_cx,
                new_cy,
            )

    # 2. Scroll
    if dz != 0:
        return (
            {"action": "scroll", "delta_x": 0, "delta_y": dz * 120},
            new_cx,
            new_cy,
        )

    # 3. Keyboard
    if keyboard_keys:
        # Find dominant key combination (the substep with most keyboard keys)
        best_substep: list[str] = []
        for step_keys in substeps:
            mapped = [_map_key(t) for t in step_keys if t not in MOUSE_BUTTONS]
            if len(mapped) > len(best_substep):
                best_substep = mapped

        if len(best_substep) == 1:
            key = best_substep[0]
            hold_count = keyboard_keys.get(key, 1)
            duration = hold_count * SUBSTEP_DURATION_SEC
            return (
                {"action": "press_key", "key": key, "duration": duration},
                new_cx,
                new_cy,
            )
        elif len(best_substep) > 1:
            max_hold = max(keyboard_keys.get(k, 1) for k in best_substep)
            duration = max_hold * SUBSTEP_DURATION_SEC
            return (
                {"action": "press_keys", "keys": best_substep, "duration": duration},
                new_cx,
                new_cy,
            )

    # 4. Mouse movement only (significant delta, no keys or clicks)
    if abs(dx) > 5 or abs(dy) > 5:
        pixel_x = new_cx / 1000.0 * image_w
        pixel_y = new_cy / 1000.0 * image_h
        return (
            {
                "action": "mouse_move",
                "from_x": cursor_x / 1000.0 * image_w,
                "from_y": cursor_y / 1000.0 * image_h,
                "x": pixel_x,
                "y": pixel_y,
            },
            new_cx,
            new_cy,
        )

    # 5. Nothing meaningful → wait
    return ({"action": "wait", "duration": 0.2}, new_cx, new_cy)


def parse_sft_to_gameworld(
    raw_text: str,
    cursor_x: float,
    cursor_y: float,
    image_w: int = 1280,
    image_h: int = 720,
) -> tuple[dict | None, float, float]:
    """End-to-end: raw model text → GameWorld action dict + updated cursor.

    Returns (None, cursor_x, cursor_y) if parsing fails.
    """
    sft_action = parse_sft_action(raw_text)
    if sft_action is None:
        return None, cursor_x, cursor_y
    return sft_to_gameworld_action(sft_action, cursor_x, cursor_y, image_w, image_h)


def sft_to_key_timeline(
    sft_action: dict,
    cursor_x: float,
    cursor_y: float,
    image_w: int = 1280,
    image_h: int = 720,
) -> tuple[dict, float, float]:
    """Lossless variant: keep the 6 substeps as a per-33ms held-key timeline
    (no collapse). Keys are mapped to GameWorld names; mouse buttons become
    "MOUSE:<button>" items (so they can be held across slots/windows). The net
    mouse displacement + scroll are emitted once for the whole window."""
    dx = sft_action["norm_dx"]
    dy = sft_action["norm_dy"]
    dz = sft_action["norm_dz"]
    substeps = sft_action["substeps"]

    slots: list[list[str]] = []
    for step_keys in substeps:
        items: list[str] = []
        for tok in step_keys:
            if tok in MOUSE_BUTTONS:
                items.append("MOUSE:" + MOUSE_BUTTONS[tok])
            else:
                items.append(_map_key(tok))
        slots.append(items)

    new_cx = max(0.0, min(1000.0, cursor_x + dx))
    new_cy = max(0.0, min(1000.0, cursor_y + dy))

    action: dict = {
        "action": "key_timeline",
        "slots": slots,
        "slot_ms": SUBSTEP_DURATION_SEC * 1000.0,
    }
    if abs(dx) > 5 or abs(dy) > 5:
        action["mouse"] = {
            "from_x": cursor_x / 1000.0 * image_w,
            "from_y": cursor_y / 1000.0 * image_h,
            "x": new_cx / 1000.0 * image_w,
            "y": new_cy / 1000.0 * image_h,
        }
    if dz != 0:
        action["scroll"] = {"delta_x": 0, "delta_y": dz * 120}
    return action, new_cx, new_cy


def parse_sft_to_key_timeline(
    raw_text: str,
    cursor_x: float,
    cursor_y: float,
    image_w: int = 1280,
    image_h: int = 720,
) -> tuple[dict | None, float, float]:
    """End-to-end: raw model text → key_timeline action + updated cursor."""
    sft_action = parse_sft_action(raw_text)
    if sft_action is None:
        return None, cursor_x, cursor_y
    return sft_to_key_timeline(sft_action, cursor_x, cursor_y, image_w, image_h)


__all__ = [
    "parse_sft_action",
    "sft_to_gameworld_action",
    "parse_sft_to_gameworld",
    "sft_to_key_timeline",
    "parse_sft_to_key_timeline",
    "SFT_KEY_TO_GW",
    "MOUSE_BUTTONS",
]
