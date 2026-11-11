"""System prompts.

Three styles, and they are NOT comparable with each other.

`oga`     THE DEFAULT AND THE ONLY STYLE FOR CROSS-BENCHMARK NUMBERS. The exact
          document OmniGameArena assembles for every `<|action_start|>` model:
          gaming-expert opening naming the game, `**Game Rules**`,
          `**Common Keys in <name>**`, Output Format, Explanation, Key Naming
          Rules -- in that order, from the skeleton strings below, which are
          byte-copies of `_YOURVLA_OPENING` / `_OUTPUT_FORMAT_*` /
          `_YOURVLA_TAIL` in
          `../omnigamearena/.../benchmark/factory.py`. That file's
          `_build_yourvla_system` is the eval standard there ("every
          <|action_start|>-format model MUST use the same assembled system
          prompt"), so a ViZDoom score is only readable next to an OGA score if
          it came from the same skeleton. The two builders must stay
          byte-for-byte identical.

`native`  The CPT training system prompt, verbatim and alone. The model is told
          the output format and nothing about Doom, so this measures whether a
          policy can guess the objective of an unseen FPS from pixels. No OGA
          batch was ever measured this way.

`guided`  Historical. The training prompt plus the map's goal, the Doom key
          subset and a numbered tutorial -- structurally its own thing, matching
          neither training nor OGA. Kept only to re-read the batches in
          previously recorded results that predate the `oga` style; do not start new runs
          with it.

Whatever the style, the wording is copied, never paraphrased: the CPT model keys
its output tag off the exact Output Format section, and rephrasing it is enough
to make the model emit prose instead of `<|action_start|>`.
"""

from __future__ import annotations

from .games import GameSpec

# EXACT training system prompt (training-pipeline/merge_all_sources.py
# SYSTEM_PROMPT), byte-identical to omnigamearena's models/yourvla.py.
TRAINING_SYSTEM_PROMPT = '''You are an experienced PC game player, proficient in keyboard and mouse operations. Based on the current game screen and the corresponding historical trajectory, plan the next 200ms of actions, consisting of 6 steps. Each step is spaced 33ms apart. Every step lasts 33ms from its start time until the next step begins.
**Output Format**
<|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>
**Explanation**
1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means move right, Y>0
means move down) and scroll amount Z (Z>0 means scroll up).
2. **Key Sequence**: Then list 6 groups of keys; within each group, keys are separated by spaces,
and groups are separated by semicolons.
- Each group can contain up to 4 keys.
- If a group has no keys, leave it empty but keep the `;`.
3. Only output a plain string that conforms to the above format — no line breaks and no quotation marks.
**Key Naming Rules**
- Number keys `1-9`: use lowercase English words, e.g., `one` represents the `1` key on the keyboard.
- Function keys `F1-F12`: use capitalized English words, e.g., `One` represents `F1`, `Two` represents `F2`, and so on.
- Other keys (letters, Shift, Tab, Space, etc.): use the real keyboard name with an initial capital letter, e.g., `A`, `D`, `Shift`, `Space`, etc.'''

USER_TURN_TEXT = "current game screen"

_ACTION_SPACE = """**Keys this game uses**
- `W` / `S` — move forward / backward
- `A` / `D` — strafe left / right
- `LB` — fire the current weapon (`Ctrl` does the same)
- `Shift` — run
- `Space` — open doors / use
{weapons}
- Mouse `X` — turn (X>0 turns right); mouse `Y` — look (Y>0 looks down)
Any other key does nothing in this game."""


def _weapon_line(spec: GameSpec) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four",
             5: "five", 6: "six", 7: "seven"}
    slots = ", ".join(f"`{words[i]}`" for i in spec.weapon_slots if i in words)
    return f"- {slots} — select weapon slot {'/'.join(str(i) for i in spec.weapon_slots)}"


# -- `oga` skeleton -------------------------------------------------------
# Byte-copies of factory.py. Copied rather than imported because a worker runs
# from a /tmp snapshot of this directory with no OGA checkout on sys.path; the
# equality test is what keeps the copy honest.
_OGA_OPENING = (
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
_OGA_TAIL = (
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
_MOUSE_BTN_FIX = {"LMB": "LB", "RMB": "RB", "MMB": "MB"}

_OUTPUT_FORMAT_PLAIN = (
    "**Output Format**\n"
    "<|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>"
)
_OUTPUT_FORMAT_THINK = (
    "**Output Format**\n"
    "**Action only (usual)**\n"
    "<think></think><|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10"
    "<|action_end|>\n"
    "**Thinking + action (when necessary)**\n"
    "<think>reasoning</think><|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10"
    "<|action_end|>"
)


OBJECTIVE_POSITIONS = ("rules", "end", "section")

# Appended after Key Naming Rules for the no-think arm (empty think + forced
# action tag). The official skeleton above stays byte-identical; this section only
# states the inference-side adaptation: no reasoning, one chunk, closing tag.
NO_THINK_NOTE = (
    "**No-thinking Mode**\n"
    "Thinking is disabled for this session: do not write any reasoning and do not output "
    "`<think>` or `</think>`. Output exactly one action chunk per turn, from `<|action_start|>` "
    "to `<|action_end|>`, and always end it with `<|action_end|>`. Do not output a second chunk."
)


def build_oga_system_prompt(spec: GameSpec, reasoning: bool = False,
                            objective: str = "", objective_pos: str = "rules",
                            no_think: bool = False) -> str:
    """OGA's `_build_yourvla_system`, with a ViZDoom GameSpec as the game.

    The Output Format section depends on `reasoning` exactly as it does there:
    with reasoning on, it advertises empty-think and think+action, because the
    CPT mix trained the tag off the think block's length rather than off the
    system prompt.

    `objective` adds one `Current objective: ...` line. `objective_pos="rules"`
    makes it the last line of Game Rules; `"end"` makes it the last line of the
    whole system prompt, after Key Naming Rules; `"section"` closes the prompt
    with a `**Current Objective**` section, as the Minecraft S1
    (GAME-JEV/mc/mc_s1.py) does. Empty means the OGA prompt byte for byte; a
    non-empty one is a different experiment and must run under its own served
    name.
    """
    if objective_pos not in OBJECTIVE_POSITIONS:
        raise ValueError(f"objective_pos must be one of {OBJECTIVE_POSITIONS}, "
                         f"got {objective_pos!r}")
    if objective and objective_pos == "section":
        line = "**Current Objective**\n" + objective.strip()
    else:
        line = f"Current objective: {objective.strip()}" if objective else ""
    in_rules = bool(line) and objective_pos == "rules"
    name = spec.prompt_key or spec.name
    parts = [_OGA_OPENING.format(name=name)]
    if spec.rules or in_rules:
        rules = spec.rules.strip()
        if in_rules:
            rules = f"{rules}\n{line}" if rules else line
        parts.append("**Game Rules**\n" + rules)
    common = [f"- {_MOUSE_BTN_FIX.get(k, k)} -> {v}" for k, v in spec.key_bindings]
    common.append("- Mouse X Y -> aim / rotate the camera (X>0 right, Y>0 down)")
    parts.append(f"**Common Keys in {name}**\n" + "\n".join(common))
    parts.append(_OUTPUT_FORMAT_THINK if reasoning else _OUTPUT_FORMAT_PLAIN)
    parts.append(_OGA_TAIL)
    if no_think:
        parts.append(NO_THINK_NOTE)
    if line and objective_pos in ("end", "section"):
        parts.append(line)
    return "\n".join(parts)


def build_system_prompt(spec: GameSpec, style: str = "oga",
                        reasoning: bool = False, objective: str = "",
                        objective_pos: str = "rules", no_think: bool = False) -> str:
    if style == "oga":
        return build_oga_system_prompt(spec, reasoning=reasoning, objective=objective,
                                       objective_pos=objective_pos, no_think=no_think)
    if objective:
        raise ValueError(f"an objective is only defined for the oga style, not {style!r}")
    if style == "native":
        return TRAINING_SYSTEM_PROMPT
    if style != "guided":
        raise ValueError(
            f"prompt_style must be 'oga', 'guided' or 'native', got {style!r}")

    tutorial = "\n".join(f"{i}. {line}" for i, line in enumerate(spec.tutorial, 1))
    return (
        f"{TRAINING_SYSTEM_PROMPT}\n\n"
        f"## Game\n"
        f"You are playing Doom, a first-person shooter, on the map `{spec.name}`.\n"
        f"{spec.goal}\n\n"
        f"{_ACTION_SPACE.format(weapons=_weapon_line(spec))}\n\n"
        f"## Game Tutorials\n{tutorial}\n\n"
        f"## User Instruction\n{spec.goal}"
    )
