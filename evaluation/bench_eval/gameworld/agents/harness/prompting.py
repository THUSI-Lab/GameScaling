"""Shared prompt-rendering helpers for model clients."""

from __future__ import annotations

import logging
import os
import re
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined

LOGGER = logging.getLogger(__name__)

CATALOG_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "catalog" / "prompt_templates"


class SemanticControlLike(Protocol):
    """Minimal shape used by prompt and semantic-control helpers."""

    action_id: str
    description: str
    binding: dict[str, Any]
    required: list[str]


@cache
def _get_env(templates_dir: str | Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(templates_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_catalog_template(
    template_name: str,
    context: dict[str, Any],
    templates_dir: Path = CATALOG_PROMPTS_DIR,
) -> str:
    """Render one catalog prompt template with strict Jinja variables."""
    env = _get_env(templates_dir)
    return env.get_template(template_name).render(**context).strip()


def join_prompt_sections(*sections: str | None) -> str:
    """Join non-empty prompt blocks with blank lines."""
    return "\n\n".join(
        str(section).strip()
        for section in sections
        if section and str(section).strip()
    )


# GameWorld/Playwright key names -> training-pipeline SFT data-naming tokens.
# Only keys whose naming differs need translating; letters/Shift/Space/Tab/Enter
# are identical in both conventions. Used to rewrite a game's computer-use
# controls into the naming the SFT model was trained to emit.
# GameWorld canonical key name -> training-pipeline STRICT training token
# (training key vocabulary: arrows are Up/Down/Left/Right, NOT
# *Arrow; Backspace->Back; Escape->Esc; CapsLock->Caps; Control->Ctrl). This is
# the vocabulary the model was trained on, so the prompt must present controls
# in these exact names.
_GW_TO_SFT_KEY = {
    "ArrowLeft": "Left",
    "ArrowRight": "Right",
    "ArrowUp": "Up",
    "ArrowDown": "Down",
    "Escape": "Esc",
    "CapsLock": "Caps",
    "Control": "Ctrl",
    "Backspace": "Back",
}
_GW_KEY_RE = re.compile(r"\b(" + "|".join(map(re.escape, _GW_TO_SFT_KEY)) + r")\b")

_DIGIT_WORD = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
               "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
# Single-letter/digit key references only appear in ACTION SPACE listings in a few
# fixed shapes; normalize ONLY those so prose ("mine a block", "level 5") is safe:
#   slash combo  "a/d" -> "A/D"     plus combo "w + a" -> "W + A"
#   list item    "- r:" -> "- R:"   colon key  "w:" -> "W:"
#   digit range  "1-9:" -> "one-nine:"   list digit "- 1:" -> "- one:"
_COMBO_SLASH_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\s*/\s*([A-Za-z])(?![A-Za-z0-9])")
_COMBO_PLUS_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\s*\+\s*([A-Za-z])(?![A-Za-z0-9])")
_LIST_LETTER_RE = re.compile(r"(?m)^(\s*-\s*)([a-z])(?=\s*[:/+])")
_COLON_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([a-z])(?=\s*:)")
_DIGIT_RANGE_RE = re.compile(r"(?<![A-Za-z0-9])(\d)\s*-\s*(\d)(?=\s*:)")
_LIST_DIGIT_RE = re.compile(r"(?m)^(\s*-\s*)(\d)(?=\s*[:/])")


def translate_controls_to_sft_naming(text: str | None) -> str:
    """Rewrite a game's controls block into the SFT STRICT key vocabulary.

    Multi-char GW names via _GW_TO_SFT_KEY (arrows, Esc, ...); then single-letter
    keys -> uppercase and digit keys -> words, but ONLY in ACTION-SPACE key-listing
    positions (combos, list items, colon-suffixed) so prose is left intact."""
    if not text:
        return ""
    out = _GW_KEY_RE.sub(lambda m: _GW_TO_SFT_KEY[m.group(1)], str(text))
    out = _DIGIT_RANGE_RE.sub(lambda m: f"{_DIGIT_WORD[m.group(1)]}-{_DIGIT_WORD[m.group(2)]}", out)
    out = _LIST_DIGIT_RE.sub(lambda m: m.group(1) + _DIGIT_WORD[m.group(2)], out)
    out = _COMBO_SLASH_RE.sub(lambda m: f"{m.group(1).upper()}/{m.group(2).upper()}", out)
    out = _COMBO_PLUS_RE.sub(lambda m: f"{m.group(1).upper()} + {m.group(2).upper()}", out)
    out = _LIST_LETTER_RE.sub(lambda m: m.group(1) + m.group(2).upper(), out)
    out = _COLON_LETTER_RE.sub(lambda m: m.group(1).upper(), out)
    return out


def build_semantic_controls_map(
    semantic_controls: list[SemanticControlLike] | None,
) -> dict[str, dict[str, Any]]:
    """Build a semantic-control lookup keyed by canonical action id."""
    mapping: dict[str, dict[str, Any]] = {}
    for action in semantic_controls or []:
        action_id = str(getattr(action, "action_id", "") or "").strip()
        binding = getattr(action, "binding", None)
        if not action_id or not isinstance(binding, dict) or not binding:
            continue

        binding_copy = dict(binding)
        mapping[action_id] = dict(binding_copy)
    return mapping


def render_semantic_action_space(semantic_controls: list[SemanticControlLike] | None) -> str:
    """Render the semantic action block injected into general-model prompts."""
    lines = [
        "REGISTERED ACTIONS (Semantic Controls).",
        "Choose exactly ONE action per step:",
        "",
    ]

    for action in semantic_controls or []:
        action_id = str(getattr(action, "action_id", "") or "").strip()
        if not action_id:
            continue

        description = str(getattr(action, "description", "") or "").strip()
        required = [
            str(item).strip()
            for item in (getattr(action, "required", []) or [])
            if str(item).strip()
        ]

        line = f"- `{action_id}`"
        if description:
            line += f": {description}"
        if required:
            line += f" (required: {', '.join(required)})"
        lines.append(line)

    return "\n".join(lines)


def render_system_prompt(
    template_name: str,
    game_rules: str | None,
    task_prompt: str | None,
    role_section: str | None,
    computer_use_controls_section: str | None,
    semantic_action_space: str | None,
    output_format: str | None,
    game_name: str | None = None,
) -> str:
    """Render the final system prompt for one model-role pair."""
    display_name = ""
    if game_name:
        # "01_2048" -> "2048", "02_another-gentlemans-adventure" -> "another gentlemans adventure"
        display_name = re.sub(r"^\d+_", "", str(game_name)).replace("-", " ").replace("_", " ").strip()
    system_prompt = render_catalog_template(
        template_name,
        {
            "game_name_block": display_name,
            "game_rules_block": game_rules or "",
            "task_instruction_block": task_prompt or "",
            "role_control_block_semantic": join_prompt_sections(
                role_section,
                semantic_action_space,
            ),
            "role_control_block_computer_use": join_prompt_sections(
                role_section,
                computer_use_controls_section,
            ),
            # SFT variant: controls rewritten into the model's data key naming.
            "role_control_block_computer_use_sft": join_prompt_sections(
                role_section,
                translate_controls_to_sft_naming(computer_use_controls_section),
            ),
            "model_output_format_block": output_format or "",
            # Think-trained action models: expose optional <think>...</think>
            # before the action in the output-format section (gated by SFT_THINK).
            "think_enabled": bool(os.environ.get("SFT_THINK")),
            # No-think arm (SFT_THINK=closed_action): state the adaptation after
            # Key Naming Rules; the official skeleton above stays unchanged.
            "no_think_note": (os.environ.get("SFT_THINK") or "").strip().lower()
            in ("closed_action", "closed-action", "forced"),
        },
    )

    LOGGER.debug(
        "Rendered system prompt from template '%s' (%d chars)",
        template_name,
        len(system_prompt),
    )
    return system_prompt


__all__ = [
    "CATALOG_PROMPTS_DIR",
    "build_semantic_controls_map",
    "join_prompt_sections",
    "render_catalog_template",
    "render_semantic_action_space",
    "render_system_prompt",
    "translate_controls_to_sft_naming",
]
