"""Agent / Adapter factory.

Composes one ``(agent, adapter)`` pair per experiment cell from three
clean axes:

  - ``profile.kind``    - agent class:   ``vlm`` | ``openp2p`` | ``nitrogen``.
  - ``profile.method``  - VLM output style (only meaningful when
    ``kind == "vlm"``):  ``lumine`` (main table).
  - ``profile.extra``   - per-agent kwargs forwarded to the constructor.

The matching adapter is chosen automatically from ``method`` so YAMLs
never need to mention adapters by name.
"""

from __future__ import annotations

import logging
from typing import Any

from omni_game_arena.adapters.base import BaseActionAdapter
from omni_game_arena.adapters.keyboard_mouse_chunked import KeyboardMouseChunkedAdapter
from omni_game_arena.adapters.nitrogen_adapter import NitroGenAdapter
from omni_game_arena.adapters.p2p_adapter import P2PAdapter
from omni_game_arena.adapters.yourvla_adapter import YourVLAChunkedAdapter
from omni_game_arena.models import NitroGenAgent, OpenP2PAgent, VLMAgent, YourVLAAgent
from omni_game_arena.prompts.skill_injection import load_prompt_skills

from .config import ParamsPoint, AgentProfile
from .frame_pack import FramePackConfig
from .frame_pack_wrapper import attach_frame_pack
from .games.base import GameSpec

logger = logging.getLogger(__name__)


# -- Public API ----------------------------------------------------------

def build_agent_and_adapter(
    profile: AgentProfile,
    params: ParamsPoint,
    game: GameSpec,
    player_index: int | None = None,
) -> tuple[Any, BaseActionAdapter]:
    """Construct an (agent, adapter) pair for one experiment cell."""
    kind = profile.kind.lower()

    if kind == "vlm":
        return _build_vlm(profile, params, game, player_index=player_index)
    if kind == "openp2p":
        return _build_openp2p(profile)
    if kind == "nitrogen":
        return _build_nitrogen(profile)
    if kind == "yourvla":
        return _build_yourvla(profile, params, game, player_index=player_index)
    raise ValueError(
        "Unknown agent kind: "
        f"{profile.kind!r} (expected vlm | openp2p | nitrogen | yourvla)"
    )


# -- Our Qwen3-VL CPT policy (SGLang/vLLM OpenAI endpoint) ---------------

# Aligned with GameWorld `catalog/prompt_templates/sft_computer_use_template.j2`:
#   opening (gaming-expert + adaptive think) → Game Rules / Common Keys →
#   Output Format (plain, or action-only vs think+action) → Explanation → Key Naming.
# No separate "**Reasoning (think first)**" appendix.
_YOURVLA_OPENING = (
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
_YOURVLA_TAIL = (
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


# Appended after Key Naming Rules for the no-think arm (empty think + forced
# action tag). The official skeleton stays byte-identical; this section only
# states the inference-side adaptation: no reasoning, one chunk, closing tag.
NO_THINK_NOTE = (
    "**No-thinking Mode**\n"
    "Thinking is disabled for this session: do not write any reasoning and do not output "
    "`<think>` or `</think>`. Output exactly one action chunk per turn, from `<|action_start|>` "
    "to `<|action_end|>`, and always end it with `<|action_end|>`. Do not output a second chunk."
)


def _build_yourvla_system(game, reasoning: bool = False,
                          player_index: int | None = None, no_think: bool = False) -> str:
    """Per-game system prompt in GameWorld SFT-template order.

    When ``reasoning`` is set, Output Format lists both empty-think (usual) and
    think+action (when necessary). Inference still prefills opening ``<think>``
    (model may close immediately when no reasoning is needed).

    PvP/Coop games describe each side separately (``MidlineClashPlayer1`` vs
    ``...Player2``: own goal, own spawn, own opponent), so the player index has to
    reach the prompt key. Falling back to ``prompt_key`` would hand both agents
    player 1's rules and the match would silently measure the wrong thing.
    """
    key_for_player = getattr(game, "prompt_key_for_player", None)
    name = key_for_player(player_index) if callable(key_for_player) else None
    name = name or getattr(game, "prompt_key", None) or getattr(game, "name", "this game")
    parts = [_YOURVLA_OPENING.format(name=name)]

    # Game Rules (from prompts/games/<name>.txt) — before controls, like GameWorld.
    try:
        from omni_game_arena.prompts import load_game_prompt
        rules = load_game_prompt(name)
        if rules:
            parts.append("**Game Rules**\n" + rules.strip())
    except Exception:  # pragma: no cover
        pass

    # Common Keys ≈ GameWorld "Role and Controls".
    kb = dict(getattr(game, "key_bindings", None) or {})
    if not kb:
        kb = {"W": "Move forward", "A": "Move left", "S": "Move backward",
              "D": "Move right", "Space": "Jump"}
    common = [f"- {_MOUSE_BTN_FIX.get(k, k)} -> {v}" for k, v in kb.items()]
    if getattr(game, "mouse_axes", None):
        common.append("- Mouse X Y -> aim / rotate the camera (X>0 right, Y>0 down)")
    if common:
        parts.append(f"**Common Keys in {name}**\n" + "\n".join(common))

    parts.append(_OUTPUT_FORMAT_THINK if reasoning else _OUTPUT_FORMAT_PLAIN)
    parts.append(_YOURVLA_TAIL)
    if no_think:
        parts.append(NO_THINK_NOTE)
    return "\n".join(parts)


def _yourvla_resize_to(extra: dict) -> tuple[int, int] | None:
    """Observation resize sent to the policy. Independent of mouse denorm.

    Missing key -> agent default (1280x720). ``resize_to`` may be ``[W, H]``,
    ``"WxH"``, or ``null`` (leave the screenshot as captured).
    """
    if "resize_to" not in extra:
        from omni_game_arena.models.yourvla import TRAIN_RESIZE
        return TRAIN_RESIZE
    raw = extra.get("resize_to")
    if raw in (None, False, 0, "native"):
        return None
    if isinstance(raw, str):
        w, h = raw.lower().replace(",", "x").split("x")
        return (int(w), int(h))
    return (int(raw[0]), int(raw[1]))


def _build_yourvla(profile: AgentProfile, params: ParamsPoint, game=None,
                   player_index: int | None = None):
    e = profile.extra
    adapter = YourVLAChunkedAdapter(
        screen_w=e.get("screen_w", 1920),
        screen_h=e.get("screen_h", 1080),
        mouse_gain_x=e.get("mouse_gain_x", 1.0),
        mouse_gain_y=e.get("mouse_gain_y", 1.0),
        # UE RemoteInput: dy>0 = look up; training Y>0 = move down → default invert.
        invert_y=e.get("invert_y", True),
    )
    # EVAL STANDARD: every <|action_start|>-format model (CPT / SFT / RL) MUST use
    # the same assembled system prompt = skeleton + Common Keys + Game Rules. This
    # is authoritative whenever the game is known (always true in benchmark runs);
    # a stray config ``system_prompt`` cannot silently diverge from the standard.
    reasoning = bool(e.get("reasoning", getattr(params, "with_reasoning", False)))
    if game is not None:
        sys_prompt = _build_yourvla_system(
            game, reasoning=reasoning, player_index=player_index,
            no_think=e.get("reasoning_style", "two_stage") == "closed")
        if e.get("system_prompt"):
            logger.warning(
                "Ignoring config system_prompt override for yourvla: the eval "
                "standard requires the assembled skeleton+keys+rules prompt."
            )
    else:
        sys_prompt = e.get("system_prompt", "")
    agent = YourVLAAgent(
        model=profile.model,
        base_url=e.get("base_url"),
        api_key=e.get("api_key"),
        system_prompt=sys_prompt,
        history_len=(params.history_len if params.history_len is not None else 20),
        history_evict_chunk=e.get("history_evict_chunk", 1),
        # params carries it for sweeps; extra wins so a run config can set it alone.
        history_reasoning_len=e.get("history_reasoning_len",
                                    params.history_reasoning_len or 0),
        temperature=params.temperature,
        max_tokens=e.get("max_tokens", 96),
        resize_to=_yourvla_resize_to(e),
        reasoning=reasoning,
        reasoning_style=e.get("reasoning_style", "two_stage"),
        reasoning_max_tokens=e.get("reasoning_max_tokens", 2048),
        stop=e.get("stop"),
    )
    if hasattr(agent, "health_check") and not agent.health_check():
        logger.warning("YourVLA endpoint at %s is unreachable", agent.base_url)
    return agent, adapter


# -- VLM path (Claude / Gemini / GPT / Qwen-VL / ...) --------------------

def _build_vlm(
    profile: AgentProfile,
    params: ParamsPoint,
    game: GameSpec,
    player_index: int | None = None,
) -> tuple[VLMAgent, BaseActionAdapter]:
    method = (profile.method or "lumine").lower()
    adapter = _adapter_for_method(method, game, params)
    method_style = _method_for_params(method, params)

    # ``extra`` carries endpoint kwargs (``base_url``, ``api_key``) and any
    # agent-side override (``include_history_actions``).
    # ``key_bindings`` lives there too for legacy YAMLs but is consumed by
    # the adapter, not the agent - strip it before forwarding.
    agent_extra = {k: v for k, v in profile.extra.items() if k != "key_bindings"}

    agent = VLMAgent(
        model=profile.model,
        method=method_style,
        resize_size=params.resize_size,
        temperature=params.temperature,
        history_len=params.history_len,
        history_reasoning_len=params.history_reasoning_len,
        game=(
            game.prompt_key_for_player(player_index)
            if params.with_game_prompt
            else None
        ),
        **agent_extra,
    )
    prompt_skill_text = load_prompt_skills(profile.prompt_skills)
    agent.system_experience = prompt_skill_text
    if prompt_skill_text:
        logger.info(
            "Loaded %d prompt skill file(s) into VLM system prompt (%d chars)",
            len(profile.prompt_skills),
            len(prompt_skill_text),
        )
    _apply_frame_pack(agent, params)
    return agent, adapter


def _method_for_params(method: str, params: ParamsPoint):
    if method == "lumine":
        from omni_game_arena.prompts.methods.lumine import LumineStyle

        return LumineStyle(thinking=params.with_reasoning)
    return method


def _adapter_for_method(
    method: str,
    game: GameSpec,
    params: ParamsPoint,
) -> BaseActionAdapter:
    """Return the adapter paired with ``method``.

    Lumine uses the chunked keyboard+mouse adapter parameterised by the
    game's mouse axes / chunk steps.
    """
    if method == "lumine":
        chunk_steps = (
            params.chunk_steps
            if params.chunk_steps is not None
            else game.chunk_steps
        )
        return KeyboardMouseChunkedAdapter(
            key_bindings=game.key_bindings,
            mouse_axes=game.mouse_axes,
            chunk_steps=chunk_steps,
            tap_keys=game.tap_keys,
            step_duration=params.hold_duration,
        )
    raise ValueError(f"Unknown method: {method!r} (expected lumine)")


# -- Specialized policy paths --------------------------------------------

def _build_openp2p(profile: AgentProfile) -> tuple[OpenP2PAgent, P2PAdapter]:
    url = profile.extra.get("url", "http://127.0.0.1:8081")
    adapter = P2PAdapter()
    agent = OpenP2PAgent(
        url=url,
        text=profile.extra.get("text"),
        timeout=profile.extra.get("timeout", 180.0),
        health_timeout=profile.extra.get("health_timeout", 5.0),
        skip_health=profile.extra.get("skip_health", False),
        skip_reset=profile.extra.get("skip_reset", False),
        image_format=profile.extra.get("image_format", "JPEG"),
        image_quality=profile.extra.get("image_quality", 85),
    )
    if hasattr(agent, "health_check") and not agent.health_check():
        logger.warning("OpenP2P server at %s is unreachable", url)
    return agent, adapter


def _build_nitrogen(profile: AgentProfile) -> tuple[NitroGenAgent, NitroGenAdapter]:
    url = profile.extra.get("url", "http://127.0.0.1:8081")
    adapter = NitroGenAdapter(
        stick_scale=profile.extra.get("stick_scale", 80.0),
        invert_y=profile.extra.get("invert_y", True),
        axis_deadzone=profile.extra.get("axis_deadzone", 0.0),
    )
    fps = profile.extra.get("fps", 20)
    step_interval = 1.0 / fps if fps > 0 else 0.0
    agent = NitroGenAgent(
        url=url,
        text=profile.extra.get("text"),
        action_downsample_ratio=profile.extra.get("action_repeat", 1),
        step_interval=step_interval,
        allow_menu=profile.extra.get("allow_menu", False),
        timeout=profile.extra.get("timeout", 180.0),
        health_timeout=profile.extra.get("health_timeout", 5.0),
        image_format=profile.extra.get("image_format", "JPEG"),
        image_quality=profile.extra.get("image_quality", 85),
    )
    if hasattr(agent, "health_check") and not agent.health_check():
        logger.warning("NitroGen server at %s is unreachable", url)
    return agent, adapter


# -- Helpers -------------------------------------------------------------

def _apply_frame_pack(agent, params: ParamsPoint) -> None:
    """Attach FramePack compression; no-op when kernel == 'none'."""
    if params.frame_pack == "none":
        return
    cfg = FramePackConfig(
        kernel=params.frame_pack,
        base_size=params.resize_size,
        min_size=params.frame_pack_min_size,
    )
    attach_frame_pack(agent, cfg)
