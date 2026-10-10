"""System prompt for GameScaling checkpoints.

One skeleton, in this order:

1. opening: gaming expert, the game's name, the 200 ms / 6 x 33 ms contract;
2. ``**Game Rules**``: what the game is and how it is scored;
3. ``**Common Keys in <game>**``: the keys this game uses, in the training vocabulary;
4. ``**Output Format**``: empty-think action, or think + action;
5. ``**Explanation**`` + ``**Key Naming Rules**``.

The ``**...**`` markers are literal plain-text section headers that the model
saw in training, not Markdown. Copy the strings, do not paraphrase them: the
checkpoints key their output tags off the exact Output Format section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .keymaps import PLAYWRIGHT_TO_TOKEN

OPENING = (
    "You are a gaming expert, and you are currently playing {name}. You are "
    "proficient in keyboard and mouse operations. You understand game mechanics "
    "and combat pacing, can quickly extract key information from the current "
    "screen, and can think and make precise decisions at critical moments. Based "
    "on the current screen, plan the next 200ms of actions, consisting of 6 steps "
    "spaced 33ms apart. Each step lasts 33ms, until the next step begins. If the "
    "current situation continues the previous strategy, output the action "
    "directly. Think only when the situation changes significantly, the previous "
    "analysis is no longer valid, or a new objective appears."
)

OUTPUT_FORMAT = (
    "**Action only (usual)**\n"
    "<think></think><|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10"
    "<|action_end|>\n"
    "**Thinking + action (when necessary)**\n"
    "<think>reasoning</think><|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10"
    "<|action_end|>"
)

TAIL = (
    "**Explanation**\n"
    "1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means "
    "move right, Y>0 means move down) and scroll amount Z (Z>0 means scroll up).\n"
    "2. **Key Sequence**: Then list 6 groups of keys; within each group, keys are "
    "separated by spaces, and groups are separated by semicolons.\n"
    "- Each group can contain up to 4 keys.\n"
    "- If a group has no keys, leave it empty but keep the `;`.\n"
    "3. Only output a plain string that conforms to the above format — no line breaks "
    "and no quotation marks.\n"
    "**Key Naming Rules**\n"
    "- Number keys `0-9`: use lowercase English words, e.g., `zero` for `0`, `one` for `1`, "
    "... `nine` for `9`.\n"
    "- Function keys `F1-F12`: use capitalized English words, e.g., `One` represents `F1`, "
    "`Two` represents `F2`, and so on.\n"
    "- Mouse buttons: use `LB`, `RB`, `MB` for the left, right, and middle mouse button.\n"
    "- Letters: use the uppercase letter, e.g., `A`, `D`, `W`, `S`.\n"
    "- Arrow keys: use `Up`, `Down`, `Left`, `Right`.\n"
    "- Modifier/special keys: `Shift`, `Ctrl`, `Alt`, `Tab`, `Caps`, `Esc`, `Space`, "
    "`Enter`, `Back` (backspace), `Delete`, `Insert`, `Home`, `End`, `Pause`."
)

MOUSE_AIM_LINE = "- Mouse X Y -> aim / rotate the camera (X>0 right, Y>0 down)"
_MOUSE_BUTTON_ALIASES = {"LMB": "LB", "RMB": "RB", "MMB": "MB"}

USER_TURN_TEXT = "current game screen"


def build_system_prompt(
    game_name: str,
    *,
    rules: str | None = None,
    key_bindings: Iterable[tuple[str, str]] | dict[str, str] | None = None,
    mouse_aim: bool = True,
) -> str:
    """The standard system prompt for any game.

    ``key_bindings`` maps training tokens to what they do, e.g.
    ``{"W": "Move forward", "LB": "Fire"}``; ``LMB``/``RMB``/``MMB`` are accepted
    as aliases of ``LB``/``RB``/``MB``. ``mouse_aim`` adds the camera line for
    games that are steered with the mouse. The same prompt serves both
    ``PolicyConfig(reasoning=True)`` and ``reasoning=False``.
    """
    parts = [OPENING.format(name=game_name)]
    if rules and rules.strip():
        parts.append("**Game Rules**\n" + rules.strip())
    items = key_bindings.items() if isinstance(key_bindings, dict) else (key_bindings or ())
    common = [f"- {_MOUSE_BUTTON_ALIASES.get(k, k)} -> {v}" for k, v in items]
    if mouse_aim:
        common.append(MOUSE_AIM_LINE)
    if common:
        parts.append(f"**Common Keys in {game_name}**\n" + "\n".join(common))
    parts.append("**Output Format**\n" + OUTPUT_FORMAT)
    parts.append(TAIL)
    return "\n".join(parts)


_PW_TO_TOKEN = PLAYWRIGHT_TO_TOKEN
_PW_KEY_RE = re.compile(r"\b(" + "|".join(map(re.escape, _PW_TO_TOKEN)) + r")\b")
_DIGIT_WORD = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
               "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
_COMBO_SLASH_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\s*/\s*([A-Za-z])(?![A-Za-z0-9])")
_COMBO_PLUS_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\s*\+\s*([A-Za-z])(?![A-Za-z0-9])")
_LIST_LETTER_RE = re.compile(r"(?m)^(\s*-\s*)([a-z])(?=\s*[:/+])")
_COLON_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([a-z])(?=\s*:)")
_DIGIT_RANGE_RE = re.compile(r"(?<![A-Za-z0-9])(\d)\s*-\s*(\d)(?=\s*:)")
_LIST_DIGIT_RE = re.compile(r"(?m)^(\s*-\s*)(\d)(?=\s*[:/])")


def translate_controls(text: str | None) -> str:
    """Rewrite key names written for a browser / keyboard API into training tokens.

    ``ArrowLeft`` -> ``Left``, ``Escape`` -> ``Esc``, ...; single letters become
    upper case and digits become words, but only where they stand as keys
    (``a/d``, ``w + a``, ``- r:``, ``1-9:``) so prose is left alone.
    """
    if not text:
        return ""
    out = _PW_KEY_RE.sub(lambda m: _PW_TO_TOKEN[m.group(1)], str(text))
    out = _DIGIT_RANGE_RE.sub(lambda m: f"{_DIGIT_WORD[m.group(1)]}-{_DIGIT_WORD[m.group(2)]}", out)
    out = _LIST_DIGIT_RE.sub(lambda m: m.group(1) + _DIGIT_WORD[m.group(2)], out)
    out = _COMBO_SLASH_RE.sub(lambda m: f"{m.group(1).upper()}/{m.group(2).upper()}", out)
    out = _COMBO_PLUS_RE.sub(lambda m: f"{m.group(1).upper()} + {m.group(2).upper()}", out)
    out = _LIST_LETTER_RE.sub(lambda m: m.group(1) + m.group(2).upper(), out)
    out = _COLON_LETTER_RE.sub(lambda m: m.group(1).upper(), out)
    return out
