"""Action adapter for our Qwen3-VL CPT policy (STRICT Lumine-token format).

Our model emits ONE 200ms action chunk per turn, in the STRICT token form
(the format defined by the game-action dataset cleaning standard used in training):

    <|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>

  - ``X Y Z``  = mouse RELATIVE displacement over the whole 200ms window,
                 NORMALISED (not pixels):
                   X = round(Sum dx / screen_w * 1000)  clamped +/-1000  (X>0 = right)
                   Y = round(Sum dy / screen_h * 1000)  clamped +/-1000  (Y>0 = down)
                   Z = round(Sum scroll / 120)          clamped +/-5     (Z>0 = wheel up)
  - 6 groups   = the 200ms window as 6 x ~33ms substeps; each group holds
                 <= 4 keys simultaneously, empty group = ";" (no key that
                 substep). Keyboard tokens and mouse-button tokens LB/RB/MB
                 are mixed inside a group.

This differs from OGA's native ``KeyboardMouseChunkedAdapter`` in three ways,
which is why we subclass rather than reuse directly:
  1. Timing: 6 substeps x ~33ms = 200ms (OGA default is 8 x 120ms = 960ms).
  2. Mouse is normalised per-mille, NOT degrees/pixels -> must de-normalise.
  3. Key vocabulary is our 74-token STRICT set. In particular our F-keys are
     ``One..Twelve`` (capitalised) while number keys are ``zero..nine``
     (lowercase) -- reusing OGA's map would silently send F1 as the "1" key.

We keep OGA's ``TimedChunkedAction`` execution engine (so LCRT interleaving
and press/release-diff behaviour are inherited); we only override key/mouse
translation and add the STRICT-format parser.
"""

from __future__ import annotations

import logging
import re

from .keyboard_mouse_chunked import KeyboardMouseChunkedAdapter

logger = logging.getLogger(__name__)

# 200ms chunk = 6 substeps of ~33.3ms (matches training: 6x33ms).
CHUNK_STEPS = 6
STEP_DURATION = 0.2 / CHUNK_STEPS  # ~0.0333s

# Tolerate a Claude-style ``<|action_start|}`` typo like the runner regex does.
_ACTION_TAG_RE = re.compile(
    r"<\|action_start\|[>}](.*?)<\|action_end\|[>}]", re.DOTALL,
)

# -- Our 74-token STRICT vocab -> UE5 FKey names --------------------------
# CASE MATTERS: "one".."nine" are number-row keys; "One".."Twelve" are F1-F12.
# NOTE: the UE5 FKey names below (BackSpace / SpaceBar / Hyphen / Equals /
# Apostrophe / arrow keys) must be verified against the actual OmniGameArena
# build's FKey names once it is running; they follow standard UE5 naming.
OUR_KEY_TO_UE5: dict[str, str] = {
    # Letters A-Z (identical)
    **{c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    # Number row 0-9 (lowercase words -> UE5 "Zero".."Nine")
    "zero": "Zero", "one": "One", "two": "Two", "three": "Three",
    "four": "Four", "five": "Five", "six": "Six", "seven": "Seven",
    "eight": "Eight", "nine": "Nine",
    # Function keys F1-F12 (capitalised words -> UE5 "F1".."F12")
    "One": "F1", "Two": "F2", "Three": "F3", "Four": "F4", "Five": "F5",
    "Six": "F6", "Seven": "F7", "Eight": "F8", "Nine": "F9", "Ten": "F10",
    "Eleven": "F11", "Twelve": "F12",
    # Modifiers / special (left variant chosen for L/R-merged tokens)
    "Shift": "LeftShift", "Ctrl": "LeftControl", "Alt": "LeftAlt",
    "Tab": "Tab", "Caps": "CapsLock", "Esc": "Escape",
    "Space": "SpaceBar", "Enter": "Enter", "Back": "BackSpace",
    # Navigation / editing / arrows
    "Delete": "Delete", "Insert": "Insert", "Home": "Home", "End": "End",
    "Pause": "Pause", "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
    # Punctuation
    "Equal": "Equals", "Minus": "Hyphen", "Period": "Period",
    "Slash": "Slash", "Quote": "Apostrophe",
}

# Mouse-button tokens -> UE5Client.send_mouse_button() names.
OUR_MOUSE_BUTTONS: dict[str, str] = {"LB": "left", "RB": "right", "MB": "middle"}


class YourVLAChunkedAdapter(KeyboardMouseChunkedAdapter):
    """Execute a STRICT-format 200ms chunk against the UE5 RemoteInput channel.

    The agent hands us ``{"chunk_text": "<|action_start|>...<|action_end|>"}``
    (raw model output). We parse it here so mouse de-normalisation stays with
    the adapter (it owns the screen geometry + calibration gain).

    Mouse calibration
    -----------------
    Training normalised dx/dy by the *source* capture resolution (default
    1920x1080). UE5 RemoteInput ``send_mouse_move(dx, dy)`` consumes raw mouse
    delta units, which are NOT guaranteed 1:1 with screen pixels. So the
    pixel estimate is scaled by ``mouse_gain_{x,y}`` -- start at 1.0 and
    calibrate with ``utils/mouse_calibration.py`` against the live build.
    ``invert_y`` defaults to **True**: training/YourVLA ``Y>0 = move down``,
    while UE RemoteInput ``dy>0 = look up``. Set ``invert_y=False`` only for
    an intentional A/B calibration arm.
    """

    def __init__(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
        mouse_gain_x: float = 1.0,
        mouse_gain_y: float = 1.0,
        invert_y: bool = True,
        scroll_per_click: float = 120.0,
        step_duration: float = STEP_DURATION,
        chunk_steps: int = CHUNK_STEPS,
    ):
        # tap_keys=() -> every key is a plain hold for its substep (our format
        # has no tap concept). mouse_axes drives only the prompt schema text.
        super().__init__(
            key_bindings=None,
            mouse_axes=("X", "Y", "Z"),
            chunk_steps=chunk_steps,
            tap_keys=(),
            step_duration=step_duration,
        )
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.mouse_gain_x = float(mouse_gain_x)
        self.mouse_gain_y = float(mouse_gain_y)
        self.invert_y = bool(invert_y)
        self.scroll_per_click = float(scroll_per_click)

    # -- Translation overrides (STRICT vocab, not OGA's) ------------------

    def _map_key(self, key: str) -> str:
        mapped = OUR_KEY_TO_UE5.get(key)
        if mapped is None:
            logger.warning("Dropping unmapped action token %r", key)
            return ""  # _press skips empty
        return mapped

    def _press(self, client, key: str, pressed: bool) -> None:
        if key in OUR_MOUSE_BUTTONS:
            client.send_mouse_button(OUR_MOUSE_BUTTONS[key], pressed)
            return
        ue5 = self._map_key(key)
        if ue5:
            client.send_key(ue5, pressed=pressed)

    # -- STRICT-format parsing -------------------------------------------

    def parse_chunk(self, text: str) -> dict:
        """Parse raw model output -> base-adapter action dict.

        Returns ``{"mouse": (dx_px, dy_px, scroll), "steps": [[...], ...]}``
        consumable by the inherited ``TimedChunkedAction`` engine. Malformed
        output degrades to a no-op chunk (all-empty steps, zero mouse) so a
        single bad generation never crashes the episode.
        """
        # Compatibility with two tokenizer schemes:
        #  (a) tags as PLAIN text -> ``<|action_start|>BODY<|action_end|>`` present
        #      in the decoded content (regex extracts BODY).
        #  (b) tags as SPECIAL tokens -> stripped by skip_special_tokens, so only a
        #      bare ``X Y Z ; k ; ...`` BODY comes back. Accept that too, as long as
        #      it looks like an action (leading signed int + at least one ';'), so a
        #      genuine captioning failure still degrades to a no-op.
        m = _ACTION_TAG_RE.search(text or "")
        if m:
            body = m.group(1)
        elif text and re.match(r"^\s*-?\d", text) and ";" in text:
            body = text
        else:
            body = ""
        parts = [seg.strip() for seg in body.split(";")]
        if not parts or parts[0] == "":
            logger.warning("No action payload parsed from %r", text[:120])
            return {"mouse": (0.0, 0.0, 0.0), "steps": [[] for _ in range(self.chunk_steps)]}

        # First segment = "X Y Z" normalised mouse; the rest = key groups.
        xyz = parts[0].split()
        try:
            nx = float(xyz[0]) if len(xyz) > 0 else 0.0
            ny = float(xyz[1]) if len(xyz) > 1 else 0.0
            nz = float(xyz[2]) if len(xyz) > 2 else 0.0
        except ValueError:
            nx = ny = nz = 0.0

        dx_px = nx / 1000.0 * self.screen_w * self.mouse_gain_x
        dy_px = ny / 1000.0 * self.screen_h * self.mouse_gain_y
        if self.invert_y:
            dy_px = -dy_px
        scroll = nz * self.scroll_per_click  # currently informational (no wheel API)

        groups = parts[1:]
        steps: list[list[str]] = []
        for g in groups:
            keys = g.split() if g else []
            steps.append(keys)
        # Pad / truncate to chunk_steps so timing stays exactly 200ms.
        if len(steps) < self.chunk_steps:
            steps.extend([] for _ in range(self.chunk_steps - len(steps)))
        elif len(steps) > self.chunk_steps:
            steps = steps[: self.chunk_steps]

        return {"mouse": (dx_px, dy_px, scroll), "steps": steps}

    # -- Execution (accept our chunk_text, delegate to OGA engine) --------

    def _to_base_action(self, action: dict) -> dict:
        if "steps" in action and "mouse" in action:
            return action  # already parsed (e.g. replayed traces)
        return self.parse_chunk(action.get("chunk_text", ""))

    def execute(self, client, action: dict, hold_after_s: float = 0.0) -> None:
        base = self._to_base_action(action)
        if base["mouse"][2]:
            logger.debug("scroll=%.1f requested but RemoteInput has no wheel API; dropping",
                         base["mouse"][2])
        super().execute(client, base, hold_after_s=hold_after_s)

    def execute_rt(self, client, action: dict, decision_ready) -> None:
        super().execute_rt(client, self._to_base_action(action), decision_ready)

    def start_timed_action(self, client, action: dict):
        # Used by the two-player interleaved (LCRT) runner.
        return super().start_timed_action(client, self._to_base_action(action))
