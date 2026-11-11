"""ViZDoom environment wrapper: one action chunk in, one frame + score out.

Doom only advances when we ask it to, so the world is frozen while the model
thinks. That is exactly OmniGameArena's `pdq` clock, and it is the default here
too. Two clocks charge for latency instead, and they charge for different
things -- never compare scores across clocks.

`lcrt` fast-forwards the measured decision latency into the game after the chunk
has been applied, *applying no-ops for those tics*. So the policy alternates
between acting and standing still with every key released. No real-time agent
works that way: AlphaStar's units keep executing their last order, OpenAI Five's
hero keeps walking, Lumine's character keeps holding W. `lcrt` therefore
measures something harsher than latency -- it measures latency plus forced
paralysis, and on `defend_the_center` it drove survival down to the `idle`
script's (349 vs 368 tics).

`rt` is the real-time model those systems actually use: a one-deep pipeline with
action persistence. The game advances in 6-tic chunks without pause; a decision
issued from the frame at tic T becomes available at T + latency and is applied
from the next chunk boundary onwards, and until it arrives the PREVIOUS action
keeps running. Latency then costs what it costs in reality -- decision rate and
observation staleness -- rather than idle time:

    latency <= one chunk (171ms)  ->  a fresh action every chunk, one chunk stale
                                      (this is OpenAI Five's "action offset":
                                      the action at T+1 is computed from the
                                      observation at T)
    latency == k chunks           ->  a fresh action every k chunks, the action
                                      held in between, k chunks stale

so `decision_rate = min(1, chunk / latency)` and nothing is ever a no-op except
the first few chunks before the very first decision lands.

`trt` (10-27 diagnostic) is `rt` with the thinking paused: the world is frozen
while the model decodes its <think> block and runs in real time for everything
else (ViT, prefill, action tokens, transport). The think share of decode time is
decode_ms * think_tokens / gen_tokens as reported by the /step engine. No real
agent gets this; it only splits rt's think penalty into its latency part.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import vizdoom as vzd
from PIL import Image

from .action import (
    ATTACK_IDX,
    CHUNK_STEPS,
    DOOM_BUTTONS,
    LOOK_IDX,
    TURN_IDX,
    ActionChunk,
    chunk_to_vectors,
)
from .games import TICS_PER_SECOND, GameSpec

logger = logging.getLogger(__name__)

# Native render size. 1280x720 is what the CPT model's training data was resized
# to, so rendering there avoids an upscale before the policy ever sees a frame.
DEFAULT_RESOLUTION = (1280, 720)

_RESOLUTIONS = {
    (640, 480): vzd.ScreenResolution.RES_640X480,
    (800, 600): vzd.ScreenResolution.RES_800X600,
    (1024, 768): vzd.ScreenResolution.RES_1024X768,
    (1280, 720): vzd.ScreenResolution.RES_1280X720,
    (1920, 1080): vzd.ScreenResolution.RES_1920X1080,
}

_GAME_VARIABLES = (
    "KILLCOUNT", "DEATHCOUNT", "HITCOUNT", "HEALTH", "ARMOR",
    "AMMO2", "SELECTED_WEAPON", "POSITION_X", "POSITION_Y", "ANGLE", "PITCH",
    "USER2",
)


class VizDoomEnv:
    def __init__(
        self,
        spec: GameSpec,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        render_hud: bool = True,
        render_crosshair: bool = True,
        turn_deg_per_full_x: float = 90.0,
        look_deg_per_full_y: float = 45.0,
        mouse_mode: str = "projective",
        clock_mode: str = "pdq",
    ):
        if clock_mode not in ("pdq", "lcrt", "rt", "trt"):
            raise ValueError(
                f"clock_mode must be pdq, lcrt, rt or trt, got {clock_mode!r}")
        self.spec = spec
        self.resolution = resolution
        self.render_hud = render_hud
        self.render_crosshair = render_crosshair
        self.mouse_mode = str(mouse_mode)
        self.turn_deg_per_full_x = float(turn_deg_per_full_x)
        self.look_deg_per_full_y = float(look_deg_per_full_y)
        self.clock_mode = clock_mode

        self.game = vzd.DoomGame()
        self._configure()
        self.game.init()

        self._last_frame: np.ndarray | None = None
        self._max_killcount = 0.0
        self._tics_elapsed = 0
        self._fastforward_tics = 0
        self._held_tics = 0

    # -- setup -------------------------------------------------------------

    def _configure(self) -> None:
        g, spec = self.game, self.spec
        if not os.path.isfile(spec.wad_path):
            raise FileNotFoundError(f"scenario WAD missing: {spec.wad_path}")
        g.set_doom_scenario_path(spec.wad_path)
        g.set_doom_map(spec.doom_map)
        g.set_doom_skill(spec.doom_skill)

        res = _RESOLUTIONS.get(tuple(self.resolution))
        if res is None:
            raise ValueError(
                f"resolution {self.resolution} is not a ViZDoom preset; "
                f"pick one of {sorted(_RESOLUTIONS)}"
            )
        g.set_screen_resolution(res)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        # Headless: no X server or GPU is involved, Doom renders in software.
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_render_hud(self.render_hud)
        g.set_render_crosshair(self.render_crosshair)
        g.set_render_weapon(True)
        g.set_render_decals(False)
        g.set_render_particles(False)
        g.set_render_messages(False)

        g.set_living_reward(spec.living_reward)
        g.set_death_penalty(spec.death_penalty)
        g.set_episode_timeout(spec.episode_timeout)
        g.set_episode_start_time(spec.episode_start_time)
        g.set_mode(vzd.Mode.PLAYER)

        g.set_available_buttons(list(DOOM_BUTTONS))
        g.set_available_game_variables(
            [getattr(vzd.GameVariable, v) for v in _GAME_VARIABLES]
        )

    # -- episode lifecycle -------------------------------------------------

    def reset(self, seed: int | None = None) -> Image.Image:
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self._max_killcount = 0.0
        self._tics_elapsed = 0
        self._fastforward_tics = 0
        self._held_tics = 0
        self._last_frame = None
        return self.observe()

    def observe(self) -> Image.Image:
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            self._last_frame = state.screen_buffer
        if self._last_frame is None:
            # Only reachable if the very first get_state() is empty.
            return Image.new("RGB", tuple(self.resolution), (0, 0, 0))
        return Image.fromarray(self._last_frame)

    @property
    def done(self) -> bool:
        return bool(self.game.is_episode_finished())

    def step(self, chunk: ActionChunk, latency_s: float = 0.0,
             hold_tics: int = 0) -> dict[str, Any]:
        """Apply one 200ms chunk as 6 single-tic actions.

        Returns a dict of per-step telemetry; the frame is fetched separately via
        `observe()` so the caller controls when the policy's view is taken.
        """
        vectors = chunk_to_vectors(
            chunk,
            mouse_mode=self.mouse_mode,
            turn_deg_per_full_x=self.turn_deg_per_full_x,
            look_deg_per_full_y=self.look_deg_per_full_y,
        )
        reward = 0.0
        attack_tics = 0
        pos0 = self._position()
        for vec in vectors:
            if self.game.is_episode_finished():
                break
            # One tic per substep: `make_action(vec, 6)` would hold substep 1's
            # keys for the whole window and silently discard substeps 2-6.
            reward += self.game.make_action(vec, 1)
            self._tics_elapsed += 1
            if vec[ATTACK_IDX] > 0:
                attack_tics += 1

        if self.clock_mode in ("rt", "trt") and hold_tics > 0:
            # Action persistence: the next decision is still in flight, so the
            # keys stay where the policy left them -- W stays down, the trigger
            # stays pulled. The MOUSE deltas are zeroed, because those are a
            # finite intended rotation that the 6 tics above already delivered;
            # carrying `turn_per_tic` into the hold would spin the player by
            # max(6, latency)/6 times the angle it asked for.
            hold = list(vectors[-1])
            hold[TURN_IDX] = 0.0
            hold[LOOK_IDX] = 0.0
            for _ in range(hold_tics):
                if self.game.is_episode_finished():
                    break
                reward += self.game.make_action(hold, 1)
                self._tics_elapsed += 1
                self._held_tics += 1
                if hold[ATTACK_IDX] > 0:
                    attack_tics += 1

        if self.clock_mode == "lcrt" and latency_s > 0 and not self.game.is_episode_finished():
            # Charge the thinking time back to the game: the model does not get
            # to reason for free, so a slow stack loses game time it controls.
            extra = int(round(latency_s * TICS_PER_SECOND))
            if extra > 0:
                noop = [0.0] * len(DOOM_BUTTONS)
                for _ in range(extra):
                    if self.game.is_episode_finished():
                        break
                    reward += self.game.make_action(noop, 1)
                    self._tics_elapsed += 1
                    self._fastforward_tics += 1

        gvars = self.game_variables()
        self._max_killcount = max(self._max_killcount, gvars.get("KILLCOUNT", 0.0) or 0.0)
        pos1 = self._position()
        return {
            "reward": reward,
            "total_reward": self.game.get_total_reward(),
            "attack_tics": attack_tics,
            "displacement": float(np.hypot(pos1[0] - pos0[0], pos1[1] - pos0[1])),
            "tics_elapsed": self._tics_elapsed,
            "game_variables": gvars,
            "done": self.done,
        }

    # -- telemetry ---------------------------------------------------------

    def _position(self) -> tuple[float, float]:
        try:
            return (self.game.get_game_variable(vzd.GameVariable.POSITION_X),
                    self.game.get_game_variable(vzd.GameVariable.POSITION_Y))
        except Exception:                      # noqa: BLE001 - episode may be over
            return (0.0, 0.0)

    def game_variables(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in _GAME_VARIABLES:
            try:
                out[name] = float(self.game.get_game_variable(getattr(vzd.GameVariable, name)))
            except Exception:                  # noqa: BLE001
                pass
        return out

    def episode_score(self) -> float:
        """The number Figure 8 calls the episode reward, per `GameSpec.score`."""
        if self.spec.score == "killcount":
            # D3/D4 do carry a reward script worth +1 per kill, so this equals
            # get_total_reward() on every episode measured so far -- both are the
            # paper's "cumulative episode reward". KILLCOUNT is read as a running
            # max anyway because the variable is not guaranteed readable once the
            # player is dead, which get_total_reward() would not protect against.
            return float(max(self._max_killcount,
                             self.game_variables().get("KILLCOUNT", 0.0) or 0.0))
        if self.spec.score == "total_reward":
            return float(self.game.get_total_reward())
        raise ValueError(f"unknown score mode {self.spec.score!r}")

    @property
    def fastforward_tics(self) -> int:
        return self._fastforward_tics

    @property
    def tics_elapsed(self) -> int:
        return self._tics_elapsed

    @property
    def held_tics(self) -> int:
        """`rt` only: tics spent replaying the previous chunk's key state.

        Nonzero exactly when a decision took longer than one 171 ms chunk, so it
        is 0 for policies that keep up with the action rate and grows with
        latency for those that do not.
        """
        return self._held_tics

    def close(self) -> None:
        try:
            self.game.close()
        except Exception:                      # noqa: BLE001
            pass
