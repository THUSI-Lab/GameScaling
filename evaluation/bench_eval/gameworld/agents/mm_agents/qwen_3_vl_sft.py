"""Qwen 3 VL SFT agent for training-pipeline fine-tuned models.

This agent adapts models trained with the training-pipeline action format
(<|action_start|>dx dy dz ; keys...<|action_end|>) to work with GameWorld's
canonical action system.
"""

from __future__ import annotations

import base64
import io
import os
import re
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image

from ..harness.memory import MemoryEntry
from .base.base_client import BaseClientConfig
from .base.computer_use_agent import ComputerUseAgent
from .qwen_vl.sft_action_parser import parse_sft_to_gameworld, parse_sft_to_key_timeline

# External key remap: the SFT model overfits to WASD movement, but ~24 GameWorld
# games use arrow keys. When SFT_WASD_REMAP is set, remap w/a/s/d -> arrows, but
# ONLY per-game when the WASD key is illegal for that game AND the arrow
# counterpart is legal (so WASD-native games like rocket-league / wordle typing
# are untouched). This measures the model's true intent without editing the bench.
_WASD_TO_ARROW = {"w": "ArrowUp", "a": "ArrowLeft", "s": "ArrowDown", "d": "ArrowRight"}

# Stop generation after the first action window. <|action_end|> is token 151670
# in the special-token checkpoints (vpt/cpt-v2/rl); overridable via env.
_ACTION_END = "<|action_end|>"
_ACTION_END_TOKEN_ID = int(os.environ.get("SFT_ACTION_END_TOKEN_ID", "151670"))
_VLLM_PHASE_METRICS = {
    "request_queue_time_seconds": "vllm_queue_ms",
    "request_prefill_time_seconds": "vllm_prefill_ms",
    "request_decode_time_seconds": "vllm_decode_ms",
    "time_to_first_token_seconds": "vllm_ttft_ms",
    "e2e_request_latency_seconds": "vllm_e2e_ms",
}


def _stop_config() -> tuple[list[str], list[int]]:
    """Request stop strings / token ids. Override with SFT_STOP / SFT_STOP_TOKEN_IDS.

    SFT_STOP is a comma-separated list of stop strings (e.g. <|im_end|>).
    SFT_STOP_TOKEN_IDS is a comma-separated list of token ids (e.g. 151645).
    If either is set, both default action_end stops are replaced (unset list → []).
    """
    stop_str = (os.environ.get("SFT_STOP") or "").strip()
    stop_ids = (os.environ.get("SFT_STOP_TOKEN_IDS") or "").strip()
    if not stop_str and not stop_ids:
        return [_ACTION_END], [_ACTION_END_TOKEN_ID]
    stops = [s for s in stop_str.split(",") if s] if stop_str else []
    ids = [int(x) for x in stop_ids.split(",") if x.strip()] if stop_ids else []
    return stops, ids

# Think-trained checkpoints (e.g. 8s_*): <think>/</think> are NORMAL tokens.
# When SFT_THINK is set we (a) advertise the optional think-first format in the
# prompt (see prompting.think_enabled) and (b) prefill the assistant turn via
# continue_final_message:
#   SFT_THINK=1|open  → "<think>"          (model must reason then close)
#   SFT_THINK=closed  → "<think>\n\n</think>\n\n"  (skip reasoning; emit action)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_CLOSED_EMPTY = "<think>\n\n</think>\n\n"


def _think_prefill() -> str | None:
    """Return assistant prefill text for SFT_THINK mode, or None if disabled."""
    mode = (os.environ.get("SFT_THINK") or "").strip().lower()
    if not mode or mode in ("0", "false", "off", "no"):
        return None
    if mode in ("closed", "close", "empty", "2"):
        return _THINK_CLOSED_EMPTY
    if mode in ("closed_action", "closed-action", "forced"):
        # Same as ViZDoom/OGA `reasoning_style=closed`: empty think AND a forced
        # <|action_start|>, so the reply can only be the action body.
        return _THINK_CLOSED_EMPTY + "<|action_start|>"
    # default: open <think> (1 / open / true / any other truthy)
    return _THINK_OPEN


def _keep_think_in_memory() -> bool:
    """If set, store the full <think>…</think> + action string in multi-turn history.

    Default strips the think block (19 rounds of reasoning would bloat the prompt).
    Ablations that want CoT in context set SFT_MEMORY_KEEP_THINK=1.
    """
    v = (os.environ.get("SFT_MEMORY_KEEP_THINK") or "").strip().lower()
    return v in ("1", "true", "on", "yes")


def _memory_rounds_override(config: BaseClientConfig) -> BaseClientConfig:
    """SFT_MEMORY_ROUNDS overrides catalog memory_rounds (e.g. 19→5)."""
    raw = (os.environ.get("SFT_MEMORY_ROUNDS") or "").strip()
    if not raw:
        return config
    return config.with_overrides(memory_rounds=int(raw))

_SFT_SYSTEM_PROMPT = (
    "You are an experienced PC game player, proficient in keyboard and mouse "
    "operations. Based on the current game screen and the corresponding "
    "historical trajectory, plan the "
    "next 200ms of actions, consisting of 6 steps. Each step is spaced 33ms "
    "apart. Every step lasts 33ms from its start time until the next step begins.\n"
    "**Output Format**\n"
    "<|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>\n"
    "**Explanation**\n"
    "1. **Mouse Movement**: First, specify the relative displacement X, Y "
    "(X>0 means move right, Y>0 means move down) and scroll amount Z "
    "(Z>0 means scroll up).\n"
    "2. **Key Sequence**: Then list 6 groups of keys; within each group, "
    "keys are separated by spaces, and groups are separated by semicolons.\n"
    "- Each group can contain up to 4 keys.\n"
    "- If a group has no keys, leave it empty but keep the `;`.\n"
    "3. Only output a plain string that conforms to the above format — "
    "no line breaks and no quotation marks.\n"
    "**Key Naming Rules**\n"
    "- Number keys `1-9`: use lowercase English words, e.g., `one` represents "
    "the `1` key on the keyboard.\n"
    "- Function keys `F1-F12`: use capitalized English words, e.g., `One` "
    "represents `F1`, `Two` represents `F2`, and so on.\n"
    "- Other keys (letters, Shift, Tab, Space, etc.): use the real keyboard "
    "name with an initial capital letter, e.g., `A`, `D`, `Shift`, `Space`, etc."
)


@dataclass
class Qwen3VLSFTConfig(BaseClientConfig):
    """Configuration for Qwen 3 VL SFT game agent."""

    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_type: str = "computer_use"
    endpoint: str = "http://127.0.0.1:8088/v1/chat/completions"
    max_tokens: int = 100
    cursor_x: float = 500.0
    cursor_y: float = 500.0


class Qwen3VLSFTAgent(ComputerUseAgent):
    """Agent adapter for training-pipeline SFT models on GameWorld."""

    def __init__(self, config: BaseClientConfig, **shared_tools):
        config = _memory_rounds_override(config)
        super().__init__(config, **shared_tools)
        endpoint = (os.environ.get("SFT_ENDPOINT") or "").strip() or config.endpoint
        self._endpoint = self._require_endpoint(endpoint, "Qwen SFT client")
        self._external_step_session = None
        if os.environ.get('BENCH_EXTERNAL_RUNTIME') == '1':
            from rt_external.step_transport import enabled, begin_episode
            if enabled():
                self._external_step_session = requests.Session()
                begin_episode(self._external_step_session, self._endpoint)
        self._vllm_metric_snapshot = self._read_vllm_phase_metrics()
        self._model = config.model or "/model"
        self._resize_height = max(
            0,
            int(
                (os.environ.get("SFT_RESIZE_HEIGHT") or "").strip()
                or getattr(config, "resize_height", 0)
                or 0
            ),
        )
        self._cursor_x: float = getattr(config, "cursor_x", 500.0)
        self._cursor_y: float = getattr(config, "cursor_y", 500.0)
        self._last_raw_sft_output: str | None = None
        self._logger.info(
            "SFT endpoint=%s memory_rounds=%s resize_height=%s keep_think=%s",
            self._endpoint,
            self.config.memory_rounds,
            self._resize_height or "native",
            _keep_think_in_memory(),
        )

    def _phase_metrics_enabled(self) -> bool:
        value = (os.environ.get("GW_VLLM_PHASE_METRICS") or "").strip().lower()
        return value in {"1", "true", "on", "yes"}

    def _read_vllm_phase_metrics(self) -> dict[str, float] | None:
        if not self._phase_metrics_enabled():
            return None
        parsed = urlsplit(self._endpoint)
        metrics_url = urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))
        try:
            response = requests.get(metrics_url, timeout=5.0)
            response.raise_for_status()
        except Exception as exc:
            self._logger.warning("Cannot scrape vLLM phase metrics: %s", exc)
            return None

        values = {name: 0.0 for name in _VLLM_PHASE_METRICS}
        found: set[str] = set()
        for line in response.text.splitlines():
            if not line or line.startswith("#"):
                continue
            for name in _VLLM_PHASE_METRICS:
                metric = f"vllm:{name}_sum"
                if not line.startswith(metric):
                    continue
                try:
                    values[name] += float(line.rsplit(None, 1)[1])
                    found.add(name)
                except (IndexError, ValueError):
                    pass
                break
        return {name: values[name] for name in found}

    def _build_data_url(self, image_path: Path, mime_type: str = "image/png") -> str:
        if not self._resize_height:
            return super()._build_data_url(image_path, mime_type)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width = max(1, round(image.width * self._resize_height / image.height))
            image = image.resize(
                (width, self._resize_height),
                resample=Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def prepare_prompt(
        self,
        *,
        screenshot_path: Path,
        screen_width: int,
        screen_height: int,
    ) -> tuple[str | None, str, list[MemoryEntry]]:
        del screenshot_path, screen_width, screen_height
        # Match training turn format exactly: "current game screen" + <image>,
        # no mouse-position line.
        user_prompt = "current game screen"
        # STANDARD (2026-07): all <|action_start|>-format models (CPT/SFT/RL) MUST
        # use the merged prompt = the runtime-rendered sft_computer_use_template
        # (our SFT skeleton + GameWorld game rules + role/controls in our key
        # naming + task instruction). This is now the DEFAULT. Set SFT_LEGACY_PROMPT=1
        # to force the old generic prompt (no game info) for ablations.
        if os.environ.get("SFT_LEGACY_PROMPT") or not self.config.system_prompt:
            system_prompt = _SFT_SYSTEM_PROMPT
        else:
            system_prompt = self.config.system_prompt
        return system_prompt, user_prompt, self._collect_memory_context()

    def build_request_payload(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[object],
        screenshot_path: Path,
        screen_width: int,
        screen_height: int,
    ) -> dict[str, object]:
        del screen_width, screen_height

        messages: list[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Memory: past screenshot+action pairs
        for entry in (memory_entries or []):
            if hasattr(entry, "type"):
                if entry.type == "image":
                    image_file = entry.image_file()
                    if image_file and image_file.exists():
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": entry.text or "current game screen"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self._build_data_url(image_file)},
                                },
                            ],
                        })
                elif entry.type == "text" and entry.text:
                    role = "assistant" if (entry.field or "").lower() == "action" else "user"
                    messages.append({"role": role, "content": entry.text})

        # Current turn: user prompt parts then screenshot
        user_lines = user_prompt.split("\n")
        # "current game screen" on first line, then mouse position
        user_content: list[dict] = [
            {"type": "text", "text": user_lines[0] if user_lines else "current game screen"},
            {"type": "image_url", "image_url": {"url": self._build_data_url(screenshot_path)}},
        ]
        if len(user_lines) > 1:
            user_content.append({"type": "text", "text": "\n".join(user_lines[1:])})

        messages.append({"role": "user", "content": user_content})

        # STANDARD temperature is 1.0 (set in catalog). SFT_TEMPERATURE lets an
        # ablation override it (e.g. =0 for greedy) without editing the catalog.
        temperature = float(os.environ.get("SFT_TEMPERATURE", self.config.temperature))
        stop_strs, stop_ids = _stop_config()

        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(os.environ.get("SFT_MAX_TOKENS") or self.config.max_tokens),
            # Training trajectories include <|action_end|> in assistant turns that
            # later become multi-turn history. Default vLLM omits the stop token
            # from content; keep it so memory matches the training distribution.
            "include_stop_str_in_output": True,
        }
        # STANDARD: stop after the FIRST action window (one <|action_end|>).
        # Probe override: SFT_STOP / SFT_STOP_TOKEN_IDS (e.g. <|im_end|> / 151645).
        if stop_strs:
            payload["stop"] = stop_strs
        if stop_ids:
            payload["stop_token_ids"] = stop_ids

        # Decode: default keep special tokens so <|action_start|>/<|action_end|>
        # appear in content (and multi-turn memory). Set SFT_SKIP_SPECIAL_TOKENS=1
        # to restore vLLM's strip-on-decode behavior.
        _skip = (os.environ.get("SFT_SKIP_SPECIAL_TOKENS") or "0").strip().lower()
        if _skip not in ("1", "true", "on", "yes"):
            payload["skip_special_tokens"] = False

        # THINK mode: prefill assistant and continue (vLLM needs trailing
        # role=assistant for continue_final_message). See _think_prefill().
        prefill = _think_prefill()
        if prefill is not None:
            messages.append({"role": "assistant", "content": prefill})
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        return payload

    def send_request(self, request_payload: dict[str, object]) -> object:
        request_started = perf_counter()
        try:
            client = self._external_step_session or requests
            response = client.post(self._endpoint, json=request_payload, timeout=300.0)
        except Exception as exc:
            raise RuntimeError(f"SFT vLLM request failed: {exc}") from exc
        response_received = perf_counter()
        if response.status_code != 200:
            raise RuntimeError(f"SFT vLLM HTTP {response.status_code}: {response.text}")

        timing = {"http_roundtrip_ms": (response_received - request_started) * 1000.0}
        if self._phase_metrics_enabled():
            scrape_started = perf_counter()
            current = self._read_vllm_phase_metrics()
            timing["metrics_scrape_ms"] = (perf_counter() - scrape_started) * 1000.0
            previous = self._vllm_metric_snapshot
            if current is not None:
                self._vllm_metric_snapshot = current
            if previous is not None and current is not None:
                for metric_name, output_name in _VLLM_PHASE_METRICS.items():
                    if metric_name in previous and metric_name in current:
                        timing[output_name] = max(
                            0.0,
                            (current[metric_name] - previous[metric_name]) * 1000.0,
                        )
                server_e2e = timing.get("vllm_e2e_ms")
                if server_e2e is not None:
                    timing["http_transport_and_api_ms"] = max(
                        0.0,
                        timing["http_roundtrip_ms"] - server_e2e,
                    )
                accounted = sum(
                    timing.get(name, 0.0)
                    for name in ("vllm_queue_ms", "vllm_prefill_ms", "vllm_decode_ms")
                )
                if server_e2e is not None:
                    timing["vllm_other_ms"] = max(0.0, server_e2e - accounted)
        setattr(response, "_gw_timing_ms", timing)
        return response

    def _stringify_raw_response(self, response_obj: object) -> str:
        return (
            response_obj.text
            if isinstance(response_obj, requests.Response)
            else super()._stringify_raw_response(response_obj)
        )

    def close(self) -> None:
        if self._external_step_session is not None:
            self._external_step_session.close()
            self._external_step_session = None

    def _serialize_action_for_memory(self, action: dict[str, object] | None) -> str | None:
        """Return the raw SFT token string for memory instead of canonical JSON.

        This preserves the <|action_start|>...<|action_end|> format that the
        model was trained on, preventing distribution shift in multi-turn
        conversations where the model would otherwise see (and start mimicking)
        the canonical JSON action format.
        """
        if self._last_raw_sft_output is not None:
            raw = self._last_raw_sft_output
            self._last_raw_sft_output = None
            # Default: do not carry per-step reasoning into history (19 rounds of
            # <think> bloats the prompt). SFT_MEMORY_KEEP_THINK=1 keeps the full
            # block; prepend <think> because generation continues after the prefill.
            if raw and _THINK_CLOSE in raw:
                if _keep_think_in_memory():
                    if _THINK_OPEN not in raw:
                        raw = _THINK_OPEN + raw
                else:
                    raw = raw.rsplit(_THINK_CLOSE, 1)[1].strip()
            return raw
        if action is None:
            return None
        return super()._serialize_action_for_memory(action)

    def _remap_wasd(self, action: dict) -> dict:
        """Game-aware WASD->arrow remap (only when SFT_WASD_REMAP env is set)."""
        if not os.environ.get("SFT_WASD_REMAP"):
            return action
        allowed = getattr(self.config, "allowed_keys", None) or frozenset()
        if not allowed:
            return action

        def fix(k: str) -> str:
            arrow = _WASD_TO_ARROW.get(k)
            if arrow and k not in allowed and arrow in allowed:
                return arrow
            return k

        if action.get("action") == "press_key" and action.get("key"):
            action["key"] = fix(action["key"])
        elif action.get("action") == "press_keys" and action.get("keys"):
            action["keys"] = [fix(k) for k in action["keys"]]
        elif action.get("action") == "key_timeline" and action.get("slots"):
            action["slots"] = [
                [it if str(it).startswith("MOUSE:") else fix(it) for it in slot]
                for slot in action["slots"]
            ]
        return action

    @staticmethod
    def _ensure_action_end(text: str, *, finish_reason: str | None) -> str:
        """Re-attach <|action_end|> when stop truncated it out of content.

        Training always stores a closed action window in history. Even with
        include_stop_str_in_output, some vLLM builds still omit the stop token
        (especially stop_token_ids). If we stopped on action_end and the window
        is open, append the literal marker so multi-turn memory matches train.
        """
        if not text or _ACTION_END in text:
            return text
        # include_stop_str_in_output keeps a literal EOS when the model ends the
        # window with <|im_end|> instead of <|action_end|>; drop it before closing
        # so it is not parsed as a key of the last group.
        if (finish_reason or "").lower() == "stop":
            stripped = text.rstrip()
            for eos in ("<|im_end|>", "<|endoftext|>"):
                if stripped.endswith(eos):
                    stripped = stripped[: -len(eos)].rstrip()
            text = stripped
        # Probe: if stop was overridden away from action_end, do not fabricate it.
        stops, ids = _stop_config()
        if _ACTION_END not in stops and _ACTION_END_TOKEN_ID not in ids:
            return text
        # Only repair stop-truncated windows (not length-truncated mid-think).
        if (finish_reason or "").lower() not in ("stop",):
            return text
        if "<|action_start|>" not in text and "action_start" not in text:
            # Bare triple still counts as an action window for older markers.
            if not re.search(r"(?:^|\n)\s*-?\d+\s+-?\d+\s+-?\d+\s*;", text):
                return text
        return text.rstrip() + _ACTION_END

    def parse_response(
        self,
        response: object,
        *,
        raw_response: str,
        screen_width: int,
        screen_height: int,
    ) -> tuple[list[dict[str, object]] | None, str | None]:
        del raw_response
        data = response.json() if isinstance(response, requests.Response) else response
        message = self._require_choice_message(data, "Qwen SFT")
        response_text = self._extract_message_text(message)
        finish_reason = None
        try:
            finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
        except Exception:
            finish_reason = None
        # A forced <|action_start|> prefill is not echoed back by vLLM; restore it
        # so the reply parses (and enters history) as the full action window.
        prefill = _think_prefill() or ""
        if (prefill.endswith("<|action_start|>")
                and not response_text.lstrip().startswith("<|action_start|>")):
            response_text = "<|action_start|>" + response_text.lstrip()
        response_text = self._ensure_action_end(response_text, finish_reason=finish_reason)
        self._logger.debug("Raw SFT output: %s", response_text)

        # Try SFT format first: <|action_start|>dx dy dz ; ...
        # SFT_KEY_TIMELINE: emit the 6 substeps as a per-33ms held-key timeline
        # (lossless, cross-window continuity) instead of collapsing to 1 action.
        if os.environ.get("SFT_KEY_TIMELINE"):
            action, new_cx, new_cy = parse_sft_to_key_timeline(
                response_text,
                cursor_x=self._cursor_x,
                cursor_y=self._cursor_y,
                image_w=screen_width,
                image_h=screen_height,
            )
        else:
            action, new_cx, new_cy = parse_sft_to_gameworld(
                response_text,
                cursor_x=self._cursor_x,
                cursor_y=self._cursor_y,
                image_w=screen_width,
                image_h=screen_height,
            )

        if action is not None:
            action = self._remap_wasd(action)
            self._cursor_x = new_cx
            self._cursor_y = new_cy
            self._last_raw_sft_output = response_text
            return [action], None

        # Fallback: LoRA models sometimes emit bare JSON actions from the
        # base Qwen-VL instruct format.  Try to parse those too.
        action = self._try_parse_json_action(
            response_text, screen_width, screen_height
        )
        if action is not None:
            self._last_raw_sft_output = None
            return [action], None

        self._last_raw_sft_output = None
        return None, None

    def _try_parse_json_action(
        self, text: str, screen_width: int, screen_height: int
    ) -> dict[str, object] | None:
        """Parse bare JSON action dicts that base Qwen-VL may produce."""
        import json as _json

        text = text.strip()
        if not text.startswith("{"):
            return None
        try:
            obj = _json.loads(text)
        except _json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "action" not in obj:
            return None

        action_name = str(obj["action"]).strip().lower()

        # Normalize deprecated names
        _ALIAS = {
            "left_click": "click",
            "right_click": "click",
            "keypress": "press_key",
            "key_press": "press_key",
            "press": "press_key",
            "input_text": "type",
            "type_text": "type",
            "release_key": "wait",
        }
        action_name = _ALIAS.get(action_name, action_name)

        def _coord(val: object) -> tuple[float, float] | None:
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    return float(val[0]), float(val[1])
                except (TypeError, ValueError):
                    return None
            return None

        if action_name == "click":
            pos = _coord(obj.get("position") or obj.get("coordinate"))
            if pos:
                x, y = pos[0] / 1000.0 * screen_width, pos[1] / 1000.0 * screen_height
            else:
                x = self._cursor_x / 1000.0 * screen_width
                y = self._cursor_y / 1000.0 * screen_height
            btn = str(obj.get("button", "left")).lower()
            return {"action": "click", "x": x, "y": y, "button": btn}

        if action_name in ("press_key", "press_keys"):
            key = obj.get("key") or obj.get("keys")
            if isinstance(key, str):
                return {"action": "press_key", "key": key}
            if isinstance(key, list) and key:
                return {"action": "press_keys", "keys": key}
            return None

        if action_name == "type":
            txt = obj.get("text") or obj.get("value")
            if txt is not None:
                return {"action": "type", "text": str(txt)}

        if action_name == "wait":
            dur = obj.get("duration", 0.2)
            return {"action": "wait", "duration": dur}

        if action_name == "scroll":
            dy = obj.get("delta_y") or obj.get("amount", 0)
            return {"action": "scroll", "delta_x": 0, "delta_y": dy}

        if action_name == "mouse_move":
            pos = _coord(obj.get("position") or obj.get("coordinate"))
            if pos:
                x = pos[0] / 1000.0 * screen_width
                y = pos[1] / 1000.0 * screen_height
                return {
                    "action": "mouse_move",
                    "from_x": self._cursor_x / 1000.0 * screen_width,
                    "from_y": self._cursor_y / 1000.0 * screen_height,
                    "x": x, "y": y,
                }

        return None


__all__ = [
    "Qwen3VLSFTAgent",
    "Qwen3VLSFTConfig",
]
