"""Template flow for computer-use agents."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any

from .base_client import BaseClient


class ComputerUseAgent(BaseClient):
    """Shared request/response flow for low-level computer-use agents."""

    def prepare_prompt(
        self,
        *,
        screenshot_path: Path,
        screen_width: int,
        screen_height: int,
    ) -> tuple[str | None, str, list[Any]]:
        del screenshot_path, screen_width, screen_height
        return self._prepare_multimodal_prompt_and_memory()

    @abstractmethod
    def build_request_payload(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        memory_entries: list[Any],
        screenshot_path: Path,
        screen_width: int,
        screen_height: int,
    ) -> dict[str, Any]:
        """Build the provider-specific request payload."""

    @abstractmethod
    def send_request(self, request_payload: dict[str, Any]) -> Any:
        """Send the request payload to the provider."""

    @abstractmethod
    def parse_response(
        self,
        response: Any,
        *,
        raw_response: str,
        screen_width: int,
        screen_height: int,
    ) -> tuple[list[dict[str, object]] | None, str | None]:
        """Parse a provider response into candidate actions and optional reasoning."""

    def get_action(self, screenshot_path: Path) -> dict[str, object] | None:
        total_started = perf_counter()
        stage_started = total_started
        screen_width, screen_height = self._get_image_size(screenshot_path)
        timing = {"image_size_ms": (perf_counter() - stage_started) * 1000.0}

        stage_started = perf_counter()
        system_prompt, user_prompt, memory_entries = self.prepare_prompt(
            screenshot_path=screenshot_path,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        timing["prompt_prepare_ms"] = (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        request_payload = self.build_request_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory_entries=memory_entries,
            screenshot_path=screenshot_path,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        timing["model_input_prepare_ms"] = (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        raw_message_sent = self._stringify_raw_message_sent(request_payload)
        timing["payload_log_serialize_ms"] = (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        response = self.send_request(request_payload)
        timing["send_request_call_ms"] = (perf_counter() - stage_started) * 1000.0
        response_timing = getattr(response, "_gw_timing_ms", None)
        if isinstance(response_timing, dict):
            timing.update(
                {
                    str(key): float(value)
                    for key, value in response_timing.items()
                    if isinstance(value, (int, float))
                }
            )

        stage_started = perf_counter()
        raw_response = self._stringify_raw_response(response)
        timing["response_materialize_ms"] = (perf_counter() - stage_started) * 1000.0

        reasoning: str | None = None
        error: str | None = None
        stage_started = perf_counter()
        try:
            actions, reasoning = self.parse_response(
                response,
                raw_response=raw_response,
                screen_width=screen_width,
                screen_height=screen_height,
            )
            action, error = self._select_first_action(
                actions,
                raw_response=raw_response,
                error_prefix="No actions parsed",
                debug_label=self.__class__.__name__,
            )
        except Exception as exc:
            error = f"Failed to parse action: {exc}"
            self._logger.warning(error)
            action = None
        timing["response_parse_ms"] = (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        result = self._complete_action(
            screenshot_path=screenshot_path,
            raw_message_sent=raw_message_sent,
            raw_response=raw_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            memory_entries=memory_entries,
            action=action,
            reasoning=reasoning,
            error=error,
            timing=timing,
        )
        timing["client_finalize_ms"] = (perf_counter() - stage_started) * 1000.0
        timing["client_total_ms"] = (perf_counter() - total_started) * 1000.0
        return result
