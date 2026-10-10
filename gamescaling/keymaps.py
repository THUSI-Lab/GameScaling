"""Training key vocabulary -> the key names of common input backends.

Mouse buttons (`LB` / `RB` / `MB`) are not keyboard keys; every table below
leaves them out and backends handle them through `MOUSE_BUTTONS`.
"""

from __future__ import annotations

from .action import LETTER_KEYS

MOUSE_BUTTONS: dict[str, str] = {"LB": "left", "RB": "right", "MB": "middle"}

# Unreal Engine 5 FKey names. Left variants are
# used for the L/R-merged modifier tokens.
TO_UE5: dict[str, str] = {
    **{c: c for c in LETTER_KEYS},
    "zero": "Zero", "one": "One", "two": "Two", "three": "Three", "four": "Four",
    "five": "Five", "six": "Six", "seven": "Seven", "eight": "Eight", "nine": "Nine",
    "One": "F1", "Two": "F2", "Three": "F3", "Four": "F4", "Five": "F5", "Six": "F6",
    "Seven": "F7", "Eight": "F8", "Nine": "F9", "Ten": "F10", "Eleven": "F11",
    "Twelve": "F12",
    "Shift": "LeftShift", "Ctrl": "LeftControl", "Alt": "LeftAlt", "Tab": "Tab",
    "Caps": "CapsLock", "Esc": "Escape", "Space": "SpaceBar", "Enter": "Enter",
    "Back": "BackSpace", "Delete": "Delete", "Insert": "Insert", "Home": "Home",
    "End": "End", "Pause": "Pause",
    "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
    "Equal": "Equals", "Minus": "Hyphen", "Period": "Period", "Slash": "Slash",
    "Quote": "Apostrophe",
}

# Playwright `page.keyboard` names (browser games). Letters are lower case.
TO_PLAYWRIGHT: dict[str, str] = {
    **{c: c.lower() for c in LETTER_KEYS},
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "One": "F1", "Two": "F2", "Three": "F3", "Four": "F4", "Five": "F5", "Six": "F6",
    "Seven": "F7", "Eight": "F8", "Nine": "F9", "Ten": "F10", "Eleven": "F11",
    "Twelve": "F12",
    "Shift": "Shift", "Ctrl": "Control", "Alt": "Alt", "Tab": "Tab", "Caps": "CapsLock",
    "Esc": "Escape", "Space": "Space", "Enter": "Enter", "Back": "Backspace",
    "Delete": "Delete", "Insert": "Insert", "Home": "Home", "End": "End", "Pause": "Pause",
    "Up": "ArrowUp", "Down": "ArrowDown", "Left": "ArrowLeft", "Right": "ArrowRight",
    "Equal": "Equal", "Minus": "Minus", "Period": "Period", "Slash": "Slash",
    "Quote": "Quote",
}

# Playwright name -> training token, for rewriting a game's documented controls
# into the vocabulary the model emits.
PLAYWRIGHT_TO_TOKEN: dict[str, str] = {
    "ArrowLeft": "Left", "ArrowRight": "Right", "ArrowUp": "Up", "ArrowDown": "Down",
    "Escape": "Esc", "CapsLock": "Caps", "Control": "Ctrl", "Backspace": "Back",
}
