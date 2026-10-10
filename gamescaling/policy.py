"""Multi-turn policy client for a GameScaling checkpoint served by vLLM.

One call to :meth:`GameScalingPolicy.act` is one 200 ms decision:

    system            the game's system prompt (see `gamescaling.prompt`)
    user, assistant   one pair per past decision: "current game screen" + frame,
    ...               then the action the model took (its reasoning is dropped
                      unless `history_think` asks to keep it)
    user              "current game screen" + the current frame
    assistant         prefilled with "<think>" (reasoning=True) or
                      "<think>\\n\\n</think>" (reasoning=False); the model continues

The checkpoint's own chat template renders a past assistant turn without
reasoning as `<think>\\n\\n</think>\\n\\n<|action_start|>...`, exactly as in
training, so the client only ever sends the action string.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import requests
from PIL import Image

from .action import (
    ACTION_END,
    ACTION_START,
    THINK_OPEN,
    Action,
    parse_action,
    split_reasoning,
)
from .prompt import USER_TURN_TEXT

logger = logging.getLogger(__name__)

# Assistant prefill. Both modes use the same system prompt; the empty think
# block is exactly how training renders a turn without reasoning.
THINK_PREFILL = THINK_OPEN
NO_THINK_PREFILL = "<think>\n\n</think>"

_RESAMPLE = {"bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC}


@dataclass
class PolicyConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "gamescaling"
    api_key: str = "EMPTY"
    # Past decisions kept in context (one frame + one action each).
    history_len: int = 5
    # How many of the most recent past turns keep their <think> text; -1 = all.
    history_think: int = 0
    # Frame sent to the model. None keeps the given game screen as-is.
    image_size: tuple[int, int] | None = None
    image_format: str = "jpeg"            # jpeg | png
    jpeg_quality: int = 90
    resample: str = "bilinear"
    # True: prefill "<think>" and let the model decide whether and how long to
    # think. False: prefill an empty think block so the model acts directly.
    reasoning: bool = True
    temperature: float = 1.0
    max_tokens: int = 2048
    stop: list[str] = field(default_factory=lambda: [ACTION_END])
    # Also recover malformed replies (an action inside an unclosed <think>, a
    # keys-only skeleton). Off by default.
    lenient_parse: bool = False
    timeout: float = 120.0

    @property
    def prefill(self) -> str:
        return THINK_PREFILL if self.reasoning else NO_THINK_PREFILL


class Turn(NamedTuple):
    image_url: str
    reasoning: str
    action_text: str


@dataclass
class Decision:
    """The outcome of one :meth:`GameScalingPolicy.act` call."""

    action: Action
    text: str
    reasoning: str
    latency_ms: float
    finish_reason: str | None = None
    completion_tokens: int | None = None
    error: str | None = None


def encode_image(image: Image.Image, cfg: PolicyConfig) -> str:
    """Resize + encode a frame into the data URL the request carries."""
    img = image
    if cfg.image_size is not None and img.size != tuple(cfg.image_size):
        img = img.resize(tuple(cfg.image_size), _RESAMPLE[cfg.resample])
    buf = io.BytesIO()
    if cfg.image_format == "png":
        img.convert("RGB").save(buf, format="PNG")
        mime = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=cfg.jpeg_quality)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def user_turn(image_url: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": USER_TURN_TEXT},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]}


def assistant_turn(action_text: str, reasoning: str = "") -> dict:
    if reasoning:
        action_text = f"{THINK_OPEN}\n{reasoning.strip()}\n</think>\n\n{action_text}"
    return {"role": "assistant", "content": action_text}


def build_request(messages: list[dict], cfg: PolicyConfig) -> dict:
    """OpenAI-compatible /chat/completions body with the vLLM extensions we rely on."""
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": list(messages),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        # The action tags are special tokens; keep them in the decoded text.
        "skip_special_tokens": False,
    }
    if cfg.stop:
        body["stop"] = list(cfg.stop)
    body["messages"].append({"role": "assistant", "content": cfg.prefill})
    body["continue_final_message"] = True
    body["add_generation_prompt"] = False
    return body


def finalize(text: str, finish_reason: str | None, cfg: PolicyConfig) -> str:
    """Undo what the request itself did to the reply.

    The prefill is not part of the returned text, and a stop string is removed
    from it. Both are restored so the reply parses the same way as training data.
    """
    text = text or ""
    if not text.lstrip().startswith(THINK_OPEN):
        text = cfg.prefill + text
    _, tail, closed = split_reasoning(text)
    if closed and ACTION_START not in tail:
        bare = tail.strip()
        if re.match(r"^-?\d", bare) and ";" in bare:
            text = text[: len(text) - len(tail)] + ACTION_START + bare
    if (finish_reason == "stop" and ACTION_START in text
            and ACTION_END not in text[text.rfind(ACTION_START):]):
        text += ACTION_END
    return text


class GameScalingPolicy:
    """Stateful client: call :meth:`reset` at episode start, :meth:`act` per 200 ms."""

    def __init__(self, system_prompt: str, config: PolicyConfig | None = None):
        self.system_prompt = system_prompt
        self.cfg = config or PolicyConfig()
        self._history: deque[Turn] = deque()
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def reset(self) -> None:
        self._history.clear()

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/models", timeout=10,
                                  headers={"Authorization": f"Bearer {self.cfg.api_key}"})
            return r.status_code < 400
        except requests.RequestException:
            return False

    def messages(self, image_url: str) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        n = len(self._history)
        for i, turn in enumerate(self._history):
            out.append(user_turn(turn.image_url))
            age = n - i
            keep = self.cfg.history_think < 0 or age <= self.cfg.history_think
            out.append(assistant_turn(turn.action_text, turn.reasoning if keep else ""))
        out.append(user_turn(image_url))
        return out

    def act(self, image: Image.Image) -> Decision:
        url = encode_image(image, self.cfg)
        body = build_request(self.messages(url), self.cfg)
        t0 = time.perf_counter()
        text, finish, tokens, error = "", None, None, None
        try:
            r = self._session.post(f"{self.base_url}/chat/completions", json=body,
                                   timeout=self.cfg.timeout,
                                   headers={"Authorization": f"Bearer {self.cfg.api_key}"})
            if r.status_code >= 400:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:500]}")
            payload = r.json()
            choice = (payload.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            finish = choice.get("finish_reason")
            tokens = (payload.get("usage") or {}).get("completion_tokens")
        except (requests.RequestException, ValueError, KeyError) as exc:
            error = str(exc)
            logger.error("policy request failed: %s", exc)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = finalize(text, finish, self.cfg) if not error else ""
        action = parse_action(text, lenient=self.cfg.lenient_parse)
        reasoning, _, closed = split_reasoning(text)
        self._history.append(Turn(url, reasoning if closed else "", action.history_text))
        while len(self._history) > self.cfg.history_len:
            self._history.popleft()
        return Decision(action=action, text=text, reasoning=reasoning, latency_ms=latency_ms,
                        finish_reason=finish, completion_tokens=tokens, error=error)
