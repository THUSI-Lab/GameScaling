"""Shared Qwen-aligned action-string protocol for API-backed models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...harness.memory import MemoryEntry
from ..qwen_vl.sft_action_parser import parse_sft_to_key_timeline
from .computer_use_agent import ComputerUseAgent

ACTION_START = "<|action_start|>"
ACTION_END = "<|action_end|>"
_ACTION_WINDOW_RE = re.compile(
    re.escape(ACTION_START) + r".*?" + re.escape(ACTION_END),
    re.DOTALL,
)
_WASD_TO_ARROW = {
    "w": "ArrowUp",
    "a": "ArrowLeft",
    "s": "ArrowDown",
    "d": "ArrowRight",
}


class ActionStringAgent(ComputerUseAgent):
    """Computer-use flow using literal Qwen action markers and six substeps."""

    def __init__(self, config, **shared_tools):
        super().__init__(config, **shared_tools)
        self._cursor_x = float(getattr(config, "cursor_x", 500.0))
        self._cursor_y = float(getattr(config, "cursor_y", 500.0))
        self._last_action_text: str | None = None

    def prepare_prompt(
        self,
        *,
        screenshot_path: Path,
        screen_width: int,
        screen_height: int,
    ) -> tuple[str | None, str, list[MemoryEntry]]:
        del screenshot_path, screen_width, screen_height
        return self.config.system_prompt, "current game screen", self._collect_memory_context()

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        choice = choices[0]
        value = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
        return str(value) if value is not None else None

    @classmethod
    def _response_text(cls, response: Any) -> str:
        message = cls._require_choice_message(response, "API action-string model")
        return cls._extract_message_text(message)

    @staticmethod
    def _close_stopped_action(text: str, finish_reason: str | None) -> str:
        if ACTION_START not in text or ACTION_END in text:
            return text
        if (finish_reason or "").lower() == "stop":
            return text.rstrip() + ACTION_END
        return text

    @staticmethod
    def _extract_reasoning(text: str) -> str | None:
        if "</think>" not in text:
            return None
        before, _, _ = text.rpartition("</think>")
        reasoning = before.replace("<think>", "").strip()
        return reasoning or None

    def _remap_wasd(self, action: dict[str, object]) -> dict[str, object]:
        allowed = getattr(self.config, "allowed_keys", None) or frozenset()

        def fix(key: str) -> str:
            arrow = _WASD_TO_ARROW.get(key)
            if arrow and key not in allowed and arrow in allowed:
                return arrow
            return key

        if action.get("action") == "press_key" and action.get("key"):
            action["key"] = fix(str(action["key"]))
        elif action.get("action") == "press_keys" and action.get("keys"):
            action["keys"] = [fix(str(key)) for key in action["keys"]]
        elif action.get("action") == "key_timeline" and action.get("slots"):
            action["slots"] = [
                [
                    str(item) if str(item).startswith("MOUSE:") else fix(str(item))
                    for item in slot
                ]
                for slot in action["slots"]
            ]
        return action

    def _uses_only_legal_keys(self, action: dict[str, object]) -> bool:
        allowed = getattr(self.config, "allowed_keys", None) or frozenset()
        if action.get("action") != "key_timeline":
            return True
        for slot in action.get("slots") or []:
            for item in slot:
                text = str(item)
                if text.startswith("MOUSE:"):
                    continue
                if text not in allowed:
                    self._logger.warning(
                        "Rejecting action-string window with illegal key token: %r",
                        text,
                    )
                    return False
        return True

    def parse_response(
        self,
        response: Any,
        *,
        raw_response: str,
        screen_width: int,
        screen_height: int,
    ) -> tuple[list[dict[str, object]] | None, str | None]:
        del raw_response
        response_text = self._close_stopped_action(
            self._response_text(response),
            self._finish_reason(response),
        )
        action_window = _ACTION_WINDOW_RE.search(response_text)
        if action_window is None:
            self._last_action_text = None
            return None, self._extract_reasoning(response_text)

        action_text = action_window.group(0)
        action, new_cursor_x, new_cursor_y = parse_sft_to_key_timeline(
            action_text,
            cursor_x=self._cursor_x,
            cursor_y=self._cursor_y,
            image_w=screen_width,
            image_h=screen_height,
        )
        if action is None:
            self._last_action_text = None
            return None, self._extract_reasoning(response_text)

        action = self._remap_wasd(action)
        if not self._uses_only_legal_keys(action):
            self._last_action_text = None
            return None, self._extract_reasoning(response_text)

        self._cursor_x = new_cursor_x
        self._cursor_y = new_cursor_y
        self._last_action_text = action_text
        return [action], self._extract_reasoning(response_text)

    def _serialize_action_for_memory(
        self,
        action: dict[str, object] | None,
    ) -> str | None:
        if self._last_action_text is not None:
            text = self._last_action_text
            self._last_action_text = None
            return text
        return super()._serialize_action_for_memory(action)


__all__ = ["ACTION_END", "ACTION_START", "ActionStringAgent"]
