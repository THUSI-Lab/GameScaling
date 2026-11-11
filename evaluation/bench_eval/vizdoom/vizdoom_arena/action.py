"""Unified keyboard/mouse action chunk -> ViZDoom button vectors.

The policy emits one 200ms chunk per turn, in the STRICT format the CPT model was
trained on (identical to OmniGameArena's `YourVLAChunkedAdapter`):

    <|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>

  - `X Y Z` is the mouse displacement over the whole 200ms window, normalised
    per-mille of the screen (X>0 = right, Y>0 = down, Z>0 = wheel up).
  - the six groups are the window as 6 substeps of ~33ms, each holding <= 4 keys.

Doom runs at a fixed 35 tics/s, i.e. 28.6ms per tic, so **one substep == one
tic** and a chunk is 6 tics = 171ms. That is the closest integer match to the
training cadence (5Hz decisions) and is why the harness never calls
`make_action(a, 6)`: passing 6 tics at once would apply group 1's keys for the
whole window and throw away groups 2-6.

Measured engine conventions (probed 2026-08-31, ViZDoom 1.2.4), both of which
are the opposite of what the button names suggest:

  - `TURN_LEFT_RIGHT_DELTA = +10` moves ANGLE from 0 to 350, i.e. clockwise.
    Doom's ANGLE grows counter-clockwise (moving forward at ANGLE 50 travels
    towards +x/+y at 50 degrees), so **a positive delta turns RIGHT** and mouse
    X maps to it unchanged.
  - `LOOK_UP_DOWN_DELTA = +10` drives PITCH negative, and ZDoom stores positive
    pitch as looking down, so **a positive delta looks UP**. Training's Y>0 is
    "mouse moved down" = look down, so the look delta is `-Y`.

Get either sign wrong and the agent still plays, still scores, and still
produces a full results table -- it just aims away from whatever it was looking
at. There is no error to catch, hence the explicit probe recorded above.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import vizdoom as vzd

ACTION_START = "<|action_start|>"
ACTION_END = "<|action_end|>"
CHUNK_STEPS = 6

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Tolerate the `<|action_start|}` typo variant the OGA runner also accepts.
_ACTION_TAG_RE = re.compile(r"<\|action_start\|[>}](.*?)<\|action_end\|[>}]", re.DOTALL)


def split_reasoning(text: str) -> tuple[str, str, bool]:
    """`text` -> (reasoning, text after the reasoning, reasoning was closed).

    Thinking models routinely quote the action format back to themselves while
    deliberating ("so the format is <|action_start|>X Y Z ; ..."). Reading the
    action out of the whole response therefore parses the *spec* as if it were a
    decision -- and since the placeholders are not numbers, a lenient parser
    turns them into `0 0 0` with no keys, so the turn is silently scored as a
    deliberate no-op instead of a failure. Everything downstream must look only
    at what follows `</think>`.

    An unclosed `<think>` means generation ended before any action existed, so
    the tail is empty -- either the reasoning budget ran out or a stop sequence
    matched inside the trace (see `DoomPolicyAgent.stop`).
    """
    text = text or ""
    if THINK_CLOSE in text:
        head, _, tail = text.rpartition(THINK_CLOSE)
        return head.split(THINK_OPEN)[-1].strip(), tail, True
    if THINK_OPEN in text:
        return text.split(THINK_OPEN)[-1].strip(), "", False
    return "", text, True

# Fixed button order. Every map uses the same layout so a trace's action vector
# is comparable across maps and the prompt's key list never changes.
DOOM_BUTTONS: list[vzd.Button] = [
    vzd.Button.ATTACK,
    vzd.Button.USE,
    vzd.Button.SPEED,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.SELECT_WEAPON1,
    vzd.Button.SELECT_WEAPON2,
    vzd.Button.SELECT_WEAPON3,
    vzd.Button.SELECT_WEAPON4,
    vzd.Button.SELECT_WEAPON5,
    vzd.Button.SELECT_WEAPON6,
    vzd.Button.SELECT_WEAPON7,
    vzd.Button.TURN_LEFT_RIGHT_DELTA,
    vzd.Button.LOOK_UP_DOWN_DELTA,
]
_IDX = {b: i for i, b in enumerate(DOOM_BUTTONS)}
TURN_IDX = _IDX[vzd.Button.TURN_LEFT_RIGHT_DELTA]
LOOK_IDX = _IDX[vzd.Button.LOOK_UP_DOWN_DELTA]
ATTACK_IDX = _IDX[vzd.Button.ATTACK]

# STRICT vocabulary -> Doom button. Case matters in the source format:
# "one".."nine" are number-row keys, "One".."Twelve" are F1-F12 (which Doom does
# not use, so they are deliberately absent and get dropped).
KEY_TO_BUTTON: dict[str, vzd.Button] = {
    # Movement (WASD and the arrow keys Doom shipped with)
    "W": vzd.Button.MOVE_FORWARD,
    "S": vzd.Button.MOVE_BACKWARD,
    "A": vzd.Button.MOVE_LEFT,
    "D": vzd.Button.MOVE_RIGHT,
    "Up": vzd.Button.MOVE_FORWARD,
    "Down": vzd.Button.MOVE_BACKWARD,
    "Left": vzd.Button.TURN_LEFT,
    "Right": vzd.Button.TURN_RIGHT,
    # Fire: LB is the mouse token, Ctrl is vanilla Doom's keyboard fire.
    "LB": vzd.Button.ATTACK,
    "Ctrl": vzd.Button.ATTACK,
    # Interact: Space is vanilla Doom's use key, E the modern convention.
    "Space": vzd.Button.USE,
    "E": vzd.Button.USE,
    "Shift": vzd.Button.SPEED,
    # Weapon slots
    "one": vzd.Button.SELECT_WEAPON1,
    "two": vzd.Button.SELECT_WEAPON2,
    "three": vzd.Button.SELECT_WEAPON3,
    "four": vzd.Button.SELECT_WEAPON4,
    "five": vzd.Button.SELECT_WEAPON5,
    "six": vzd.Button.SELECT_WEAPON6,
    "seven": vzd.Button.SELECT_WEAPON7,
}


@dataclass
class ActionChunk:
    """One parsed 200ms window."""

    mouse: tuple[float, float, float] = (0.0, 0.0, 0.0)   # normalised X Y Z
    steps: list[list[str]] = field(default_factory=lambda: [[] for _ in range(CHUNK_STEPS)])
    parsed: bool = False              # the text really contained an action body
    dropped_tokens: list[str] = field(default_factory=list)
    # Generation ended inside <think>, so no action was ever reached. Either the
    # reasoning budget ran out or a stop sequence matched mid-thought.
    truncated: bool = False
    raw_body: str = ""                # the accepted body, verbatim

    @property
    def history_text(self) -> str:
        """What this turn should look like in the next prompt's history.

        Always a well-formed tag: an unparseable turn is recorded as an explicit
        no-op, because storing the raw text would put the model's own failure
        (empty output, or the format spec it quoted while thinking) into context
        as an example to imitate.
        """
        body = self.raw_body.strip() if self.parsed else "0 0 0 ; ; ; ; ; ;"
        return f"{ACTION_START}{body}{ACTION_END}"

    @property
    def is_noop(self) -> bool:
        return not any(self.steps) and self.mouse == (0.0, 0.0, 0.0)

    @property
    def pressed_keys(self) -> set[str]:
        return {k for grp in self.steps for k in grp}


def parse_chunk(text: str) -> ActionChunk:
    """Raw model output -> ActionChunk. Unparseable text degrades to a no-op.

    A no-op is returned rather than raised so one bad generation cannot end an
    episode; `parsed=False` is what the runner counts so the run can be rejected
    afterwards instead of quietly scoring ~0 (see `vzd_check_sanity.py`).
    """
    _, tail, closed = split_reasoning(text)
    if not closed:
        return ActionChunk(truncated=True)

    # First tag in the tail, which is what a `<|action_end|>` stop sequence would
    # have returned. Keeping that equivalence is why the tag can be left unarmed
    # for free-form reasoning without changing a well-behaved model's action.
    m = _ACTION_TAG_RE.search(tail)
    if m:
        body = m.group(1)
    elif re.match(r"^\s*-?\d", tail) and ";" in tail:
        # Tags can be stripped when the tokenizer treats them as special tokens.
        # Only accept a body that actually looks like an action, so a captioning
        # failure still registers as unparsed.
        body = tail
    else:
        return ActionChunk()

    parts = [seg.strip() for seg in body.split(";")]
    if not parts or not parts[0]:
        return ActionChunk()

    xyz = parts[0].split()
    vals: list[float] = []
    for tok in xyz[:3]:
        try:
            vals.append(float(tok))
        except ValueError:
            # Placeholders ("X Y Z") and prose are not a decision to hold still.
            return ActionChunk()
    if len(vals) < 3:
        return ActionChunk()
    # Training clamps X/Y to +-1000 per-mille and Z to +-5 wheel clicks.
    nx = max(-1000.0, min(1000.0, vals[0]))
    ny = max(-1000.0, min(1000.0, vals[1]))
    nz = max(-5.0, min(5.0, vals[2]))

    steps: list[list[str]] = []
    dropped: list[str] = []
    for group in parts[1:CHUNK_STEPS + 1]:
        keys = []
        for tok in group.split():
            if tok in KEY_TO_BUTTON:
                keys.append(tok)
            else:
                dropped.append(tok)
        steps.append(keys[:4])          # the format allows at most 4 keys per group
    while len(steps) < CHUNK_STEPS:
        steps.append([])

    return ActionChunk(mouse=(nx, ny, nz), steps=steps, parsed=True,
                       dropped_tokens=dropped, raw_body=body)


# Doom's projection, measured by `docker/vzd_calibrate_turn.py` at 1280x720:
# turn by a known angle, cross-correlate the frames, get the pixel shift.
#   horizontal  8.36 px/deg at the centre -> focal 479.1 px -> FOV 106.4 deg
#   vertical   10.09 px/deg at the centre -> focal 577.9 px -> FOV  63.8 deg
# Stored as a fraction of the frame dimension so they do not depend on the
# render resolution (the FOV does not either, at a fixed 16:9 aspect). Two
# independent checks say these are right: 106.4 deg is exactly ZDoom's
# widescreen expansion of the classic 4:3 90 deg FOV (106.26), and the
# vertical/horizontal focal ratio is 1.206 = Doom's 1.2 aspect correction.
FOCAL_X_WIDTHS = 479.1 / 1280.0     # 0.3743
FOCAL_Y_HEIGHTS = 577.9 / 720.0     # 0.8026


def chunk_to_vectors(
    chunk: ActionChunk,
    turn_deg_per_full_x: float = 90.0,
    look_deg_per_full_y: float = 45.0,
    mouse_mode: str = "projective",
) -> list[list[float]]:
    """ActionChunk -> six ViZDoom action vectors, one per tic.

    The mouse displacement is for the whole window, so it is spread evenly over
    the six tics (delta buttons are per-tic degrees). OmniGameArena instead
    sends the whole delta once at the start of the chunk, which it can do
    because UE keeps running; here the game only advances during those tics, so
    an unsplit delta would turn the entire amount in the first 28ms.

    `mouse_mode` picks how per-mille becomes degrees, and the two are NOT
    comparable:

    `linear` -- `X/1000 * turn_deg_per_full_x`. THE CORRECT FORM, because the
        training per-mille is proportional to an ANGLE, not to a screen offset.
        The data pipeline stores the raw mouse delta the OS reported, divided by
        the recording's capture width (`convert_p2p_sessions.py:247-254`,
        `convert_vpt_sessions.py:20`: "mouse dx/dy are raw mouse pixels ...
        normalize by 1280x720"), with no FOV or sensitivity step anywhere. A
        human bringing a target that sits `dx` px off centre onto the crosshair
        moves the mouse by `atan(dx/focal) / sensitivity`, so the tangent is
        already spent inside the recorded number and the inverse is a plain
        gain.

        The gain itself is underdetermined and has to be measured: every source
        normalised by its own capture width and every source game had its own
        sensitivity, so X=100 means different angles in different recordings.
        The one anchor with a documented sensitivity is the VPT/Minecraft share
        -- normalised by 1280, and MineStudio's `CAMERA_SCALER = 360/2400`
        (`evaluation/MCU/minestudio/.../hero/mc.py:676`) -- giving
        `1280 * 0.15 / 1000 = 0.192 deg` per unit, i.e. 192 deg full scale, with
        `look_deg_per_full_y = turn * 720/1280` since the sensitivity is shared
        between the axes. Treat that as the centre of a sweep, not as truth.

    `projective` -- `atan((X/1000) / FOCAL_X_WIDTHS)`, ignoring the gain
        arguments. Kept ONLY to reproduce earlier recorded runs, which were
        run with it. Its premise -- that per-mille was a fraction of a frame of
        *screen-space travel* -- is contradicted by the pipeline above, and
        applying a tangent here projects a second time. It matches the linear
        VPT anchor to within 1.25x below |X|~100 (75% of emitted turns) but
        saturates badly above that: 2.8x short at full scale, on the 7.5% of
        turns above |X|=250 that swing the gun onto a new target.

    Why not match OGA numerically: OGA resolves per-mille to *pixels*
    (`X/1000 * 1920`) and lets UE's own mouse sensitivity finish the job, and
    that sensitivity is neither documented nor calibrated (`mouse_gain` has sat
    at 1.0 since the adapter was written). There is no number to copy.
    """
    nx, ny, _ = chunk.mouse
    if mouse_mode == "projective":
        turn_total = math.degrees(math.atan((nx / 1000.0) / FOCAL_X_WIDTHS))
        look_total = -math.degrees(math.atan((ny / 1000.0) / FOCAL_Y_HEIGHTS))
    elif mouse_mode == "linear":
        turn_total = (nx / 1000.0) * turn_deg_per_full_x
        look_total = -(ny / 1000.0) * look_deg_per_full_y
    else:
        raise ValueError(
            f"mouse_mode must be 'projective' or 'linear', got {mouse_mode!r}")
    turn_per_tic = turn_total / CHUNK_STEPS
    # Y>0 is "mouse moved down" = look down; a positive LOOK delta looks up.
    look_per_tic = look_total / CHUNK_STEPS

    vectors: list[list[float]] = []
    for group in chunk.steps:
        vec = [0.0] * len(DOOM_BUTTONS)
        for key in group:
            btn = KEY_TO_BUTTON.get(key)
            if btn is not None:
                vec[_IDX[btn]] = 1.0
        vec[TURN_IDX] = turn_per_tic
        vec[LOOK_IDX] = look_per_tic
        vectors.append(vec)
    return vectors


def noop_chunk_text() -> str:
    """The well-formed empty action stored in history for unparseable turns."""
    return f"{ACTION_START}0 0 0 ; ; ; ; ; ; {ACTION_END}"
