"""Agent for our Qwen3-VL CPT policy, served OpenAI-compatibly (SGLang/vLLM).

Design goal: reproduce the TRAINING-time conversation format at inference so
there is zero train/test distribution shift. The training format is:

  - Each turn: user content = ``current game screen\n<image>`` (NO absolute
    mouse coordinates -- all sources were unified to drop them), assistant =
    ``<|action_start|>...<|action_end|>`` (one 200ms / 6x33ms chunk).
  - Frames are downsampled to 5Hz in training; here one chunk = 200ms, so the
    per-turn frame cadence is already 5Hz, and an N-frame window == the last N
    turns of history (``history_len``). Note the default 20 (4s) is the eval's
    inherited default, NOT the training window: game samples are built as
    40 turns / 8s (cpt_engine_base.yaml tokens_per_game_sample: 38000
    "# 40-frame", and build_game_think_nonoverlap_8s.py rejects rows that are
    not exactly 40 frames). Every batch so far was measured at half of it.
  - Video-native sources were resized to 1280x720 after decode, so we resize
    each observation to 1280x720 before sending.

This is a self-contained ``BaseAgent`` (no dependency on OGA's VLMAgent/method
machinery) so serving-side choices stay decoupled from the benchmark. It talks
to any OpenAI ``/v1/chat/completions`` endpoint (our SGLang deployment). Action
PARSING + mouse de-normalisation live in ``YourVLAChunkedAdapter``; this agent
only produces the raw text and hands it over as ``{"chunk_text": ...}``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .base import BaseAgent

logger = logging.getLogger(__name__)

# EXACT training system prompt (from training-pipeline/merge_all_sources.py
# SYSTEM_PROMPT). Must match training verbatim or the model reverts to plain
# image captioning instead of emitting <|action_start|>...<|action_end|>.
DEFAULT_SYSTEM_PROMPT = '''You are an experienced PC game player, proficient in keyboard and mouse operations. Based on the current game screen and the corresponding historical trajectory, plan the next 200ms of actions, consisting of 6 steps. Each step is spaced 33ms apart. Every step lasts 33ms from its start time until the next step begins.
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

USER_TURN_TEXT = "current game screen"  # matches training user turn verbatim
TRAIN_RESIZE = (1280, 720)
ACTION_START = "<|action_start|>"
ACTION_END = "<|action_end|>"
IM_END = "<|im_end|>"

# The CPT mix (cpt_engine_scaling_v2_8s.yaml) pairs embodied_reasoner_mt --
# whose assistant turns are ``<think>long trace</think><DecisionMaking>`` -- with
# game action turns that carry an INJECTED EMPTY ``<think></think>`` before
# <|action_start|>. The model therefore keys the output tag off the think block's
# length, not off the system prompt: a free-running <think> prefill lands in the
# embodied-reasoner basin and emits <Decision>, which parses to a no-op (measured
# >99% of steps on THINKING checkpoints). REASONING_STYLES pick how to avoid that.
REASONING_STYLES = ("two_stage", "closed", "free")
# EOS text that can trail an action window when the server keeps special tokens.
_EOS_TEXT = ("<|im_end|>", "<|endoftext|>")

# Qwen3-VL's chat template does NOT pass an assistant prefill through verbatim:
# it splits the message on </think> and re-renders it as
# "<think>\n{reasoning}\n</think>\n\n{content}" (chat_template.jinja L57-69).
# vLLM's continue_final_message then requires the prefill to appear verbatim in
# the rendered chat, so a prefill must already be in that exact shape or the
# request 400s. Build every prefill through this helper.
# Qwen3-VL's own template drops <think> from every assistant turn that precedes
# the last user message, so reasoning in history is silently discarded server-side
# (verified by rendering both templates). This copy keeps it; turns without
# reasoning render byte-identically to the official template, so a run with
# history_reasoning_len=0 is unaffected by which template is sent.
HISTORY_THINK_TEMPLATE = Path(__file__).with_name("chat_template_history_think.jinja")


def _extract_reasoning(text: str) -> str:
    """The <think> body of a reply, empty when it has none."""
    if "<think>" not in text or "</think>" not in text:
        return ""
    return text.split("</think>")[0].split("<think>")[-1].strip()


def _think_prefill(reasoning: str = "", content: str = "") -> str:
    return "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n" + content


class YourVLAAgent(BaseAgent):
    """Multi-turn Qwen3-VL policy client over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_len: int = 20,          # 4s @ 5Hz; training builds 40-turn/8s samples
        history_evict_chunk: int = 1,   # frames dropped once the window is full
        history_reasoning_len: int = 0,  # history turns that keep their <think>; -1 = all
        temperature: float = 1.0,       # EVAL STANDARD; below 1 biases the
                                        # zero-inflated mouse deltas toward 0
        max_tokens: int = 96,           # one chunk is short
        resize_to: tuple[int, int] | None = TRAIN_RESIZE,
        image_quality: int = 90,
        timeout: float = 120.0,
        reasoning: bool = False,        # think-enabled Output Format + prefill "<think>"
        reasoning_style: str = "two_stage",
        reasoning_max_tokens: int = 2048,
        stop: str | list[str] | None = None,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("YOURVLA_BASE_URL", "http://127.0.0.1:30000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("YOURVLA_API_KEY", "EMPTY")
        self.system_prompt = system_prompt
        self.history_len = int(history_len)
        self.history_evict_chunk = max(1, int(history_evict_chunk))
        self.history_reasoning_len = int(history_reasoning_len or 0)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.resize_to = resize_to
        logger.info(
            "[resize] observations -> %s (mouse denorm is adapter screen_w/h, not this)",
            self.resize_to,
        )
        self.image_quality = int(image_quality)
        self.timeout = float(timeout)
        self.reasoning = bool(reasoning)
        self.reasoning_style = str(reasoning_style)
        if self.reasoning_style not in REASONING_STYLES:
            raise ValueError(
                f"reasoning_style must be one of {REASONING_STYLES}, got {reasoning_style!r}"
            )
        self.reasoning_max_tokens = int(reasoning_max_tokens)
        if stop is None:
            self.stop = [ACTION_END]
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)

        # (data_url, reasoning, action_text) triples, newest last, bounded by
        # history_len. The current observation is resized+JPEG'd once and the
        # data URL is reused on later turns -- re-encoding the same 5 history
        # frames every step was a lossless no-op that cost ~30ms. Eviction is
        # explicit rather than deque(maxlen=...) so the window can be dropped
        # in chunks -- see _trim_history. Reasoning is kept even when
        # history_reasoning_len is 0 so the knob stays a rendering decision
        # rather than something the storage has already thrown away.
        self._history: deque[tuple[str, str, str]] = deque()
        self._history_think_verified = False
        # True only when the server default strips history <think> and we must
        # send HISTORY_THINK_TEMPLATE. Qwen3.5 CPT templates already keep it, so
        # the VL request template is neither needed nor a valid length probe.
        self._history_think_request_template = False
        self._session = requests.Session()
        self.last_vlm_response: str | None = None
        self.last_predict_ms: float | None = None
        self.last_encode_ms: float | None = None
        self.last_jpeg_new: int | None = None
        self.last_jpeg_reuse: int | None = None
        self.last_finish_reason: str | None = None
        self.last_stop_reason: str | None = None
        self.last_completion_tokens: int | None = None
        self.last_raw_completion: str | None = None
        self.last_action_end_restored = False
        self._n_steps = 0
        self._n_unparsed = 0
        self._warned_unparsed = False

    # -- BaseAgent API ----------------------------------------------------

    def act(self, obs: dict, task: str, action_schema: dict) -> dict:  # noqa: ARG002
        image: Image.Image = obs["image"]
        t_enc = time.perf_counter()
        current_url = self._data_url(image)
        self.last_encode_ms = (time.perf_counter() - t_enc) * 1000
        self.last_jpeg_new = 1
        self.last_jpeg_reuse = len(self._history)
        messages = self._build_messages(current_url)

        t0 = time.perf_counter()
        try:
            text = self._generate(messages)
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.error("YourVLA predict failed: %s", exc)
            text = ""  # adapter parses empty -> no-op chunk
        self.last_predict_ms = (time.perf_counter() - t0) * 1000

        self.last_vlm_response = text
        # History stores the frame the model reasoned over + its own reply, so
        # the next turn sees this turn as prior context (autoregressive). Store a
        # well-formed no-op action for empty/unparseable replies: storing raw ''
        # poisons the context (the model mimics prior empty assistant turns and
        # collapses to emitting nothing).
        # Split the reply so the action and the reasoning are stored separately;
        # history_reasoning_len then decides how many turns render their reasoning.
        if ACTION_START in text:
            hist_text = text[text.find(ACTION_START):]
        else:
            hist_text = f"{ACTION_START}0 0 0 ; ; ; ; ; ; {ACTION_END}"
        self._check_action_format(text)
        self._history.append((current_url, _extract_reasoning(text), hist_text))
        self._trim_history()
        logger.info(
            "YourVLA encode=%.0fms jpeg_new=%d jpeg_reuse=%d predict=%.0fms | %s",
            self.last_encode_ms or 0.0,
            self.last_jpeg_new or 0,
            self.last_jpeg_reuse or 0,
            self.last_predict_ms,
            text.replace("\n", " ")[:160],
        )
        return {"chunk_text": text}

    # -- Generation -------------------------------------------------------

    def _generate(self, messages: list[dict]) -> str:
        if not self.reasoning:
            return self._finalize(self._complete(messages, max_tokens=self.max_tokens))

        if self.reasoning_style == "closed":
            # Empty think + forced tag: the cheapest way to a parseable action.
            return self._forced_action(messages, reasoning="")

        if self.reasoning_style == "two_stage":
            # Stage 1 needs no prefill at all: add_generation_prompt already ends
            # the rendered chat with "<think>\n" (template L108-109), so the model
            # is mid-think and stops at </think>. Stage 2 then hard-commits the
            # continuation to <|action_start|>; it re-sends the same prompt, so it
            # hits stage 1's prefix cache and only decodes the short action body.
            trace = self._complete(
                messages, max_tokens=self.reasoning_max_tokens, stop=["</think>"],
            )
            return self._forced_action(messages, reasoning=trace.split("</think>")[0])

        # "free": nothing is forced -- DIAGNOSTIC ONLY. THINKING ckpts answer with
        # <Decision>, which parses to a no-op and scores 0.
        text = self._finalize(
            self._complete(messages, prefill="<think>", max_tokens=self.reasoning_max_tokens)
        )
        return text if text.lstrip().startswith("<think>") else "<think>" + text

    def _forced_action(self, messages: list[dict], *, reasoning: str) -> str:
        """Decode the action with <|action_start|> already committed in the prefill."""
        prefill = _think_prefill(reasoning, ACTION_START)
        body = self._complete(messages, prefill=prefill, max_tokens=self.max_tokens)
        # The prefill already opened the tag; a model that re-emits it would
        # leave <|action_start|><|action_start|> and defeat the parser.
        body = body.lstrip()
        while body.startswith(ACTION_START):
            body = body[len(ACTION_START):].lstrip()
        return self._finalize(prefill + body)

    def _complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        prefill: str | None = None,
        stop: list[str] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
            "stop": stop if stop is not None else self.stop,
            # Tags are special tokens in this tokenizer; keep them in the text.
            "skip_special_tokens": False,
        }
        if prefill is not None:
            body["messages"] = messages + [{"role": "assistant", "content": prefill}]
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
        if self._wants_history_think():
            self._verify_history_think()
            if self._history_think_request_template:
                body["chat_template"] = HISTORY_THINK_TEMPLATE.read_text()
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            # The server's message is the only thing that identifies a prefill
            # that the chat template refused; raise_for_status alone drops it.
            logger.error("vLLM %d for prefill=%r: %s",
                         resp.status_code, (prefill or "")[:120], resp.text[:500])
        resp.raise_for_status()
        payload = resp.json()
        choice = (payload.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        self.last_raw_completion = content
        self.last_finish_reason = choice.get("finish_reason")
        self.last_stop_reason = choice.get("stop_reason") or choice.get("matched_stop")
        try:
            self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            self.last_completion_tokens = None
        after = ""
        idx = content.find(ACTION_END)
        if idx >= 0:
            after = content[idx + len(ACTION_END):]
        logger.info(
            "[STOPTEST] stop=%s finish_reason=%s stop_reason=%s completion_tokens=%s "
            "has_action_end=%s has_im_end=%s after_action_end=%r",
            body["stop"], self.last_finish_reason, self.last_stop_reason,
            self.last_completion_tokens, ACTION_END in content, IM_END in content,
            after[:120],
        )
        return content

    def _finalize(self, text: str) -> str:
        """Repair the two ways a well-formed action can lose its tags."""
        # Tags stripped as special tokens -> a bare ``X Y Z ; ...`` body comes
        # back. Only wrap genuine action bodies (leading signed int + a ';') so a
        # captioning failure still stays unparseable.
        if text and ACTION_START not in text:
            bare = text.strip()
            if re.match(r"^-?\d", bare) and ";" in bare:
                text = ACTION_START + bare
        # Same rule as the official `finalize`: close an open action
        # window only when generation ended by stopping (the ``<|action_end|>``
        # stop string was consumed, or the model emitted EOS right after the
        # window). A length-truncated window stays open and counts as unparsed.
        self.last_action_end_restored = False
        start = text.rfind(ACTION_START)
        if start >= 0 and ACTION_END not in text[start:]:
            window = text[start:].rstrip()
            for eos in _EOS_TEXT:
                if window.endswith(eos):
                    window = window[: -len(eos)].rstrip()
            if self.last_finish_reason == "stop":
                text = text[:start] + window + ACTION_END
                self.last_action_end_restored = True
        return text

    def _check_action_format(self, text: str) -> None:
        """Shout early when the policy is emitting unparseable turns.

        An unparseable turn silently becomes a no-op chunk, so a run can burn
        hours and report 0 while the agent never pressed a key -- which is what
        <Decision> did to every THINKING eval before reasoning_style existed.
        """
        self._n_steps += 1
        if ACTION_START not in text:
            self._n_unparsed += 1
        if self._warned_unparsed or self._n_steps < 20:
            return
        rate = self._n_unparsed / self._n_steps
        if rate > 0.5:
            self._warned_unparsed = True
            logger.error(
                "YourVLA: %.0f%% of the first %d turns had no %s -- these parse to "
                "NO-OP chunks and the run will score ~0. reasoning=%s style=%s. "
                "Sample reply: %r",
                rate * 100, self._n_steps, ACTION_START, self.reasoning,
                self.reasoning_style, text.replace("\n", " ")[:200],
            )

    def _trim_history(self) -> None:
        """Bound the window to history_len, dropping history_evict_chunk at a time.

        Dropping one frame per step (chunk=1, the training-faithful window) shifts
        every token after the system prompt, so vLLM's prefix cache only ever hits
        the system prompt -- measured 7% on a full SOLO run, which re-prefills
        ~19k image tokens every step. With chunk=k the prompt is append-only for k-1 steps
        out of k, which is what the cache needs; the cost is a window that cycles
        between history_len-k+1 and history_len frames.
        """
        over = len(self._history) - self.history_len
        if over <= 0:
            return
        for _ in range(max(over, self.history_evict_chunk)):
            if not self._history:
                break
            self._history.popleft()

    def reset(self) -> None:
        self._history.clear()
        self.last_vlm_response = None
        self.last_predict_ms = None
        self.last_encode_ms = None
        self.last_jpeg_new = None
        self.last_jpeg_reuse = None
        if os.environ.get('BENCH_EXTERNAL_RUNTIME') == '1':
            from rt_external.step_transport import begin_episode
            begin_episode(self._session, f'{self.base_url}/chat/completions')
        # Format counters stay cumulative across episodes: the warning is about
        # the run as a whole and should fire once, not once per episode.

    def reset_history(self) -> None:
        self.reset()

    def close(self) -> None:
        self._session.close()

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/models",
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  timeout=5.0)
            return r.status_code < 400
        except requests.RequestException as exc:
            logger.warning("YourVLA health check failed at %s: %s", self.base_url, exc)
            return False

    # -- Message construction (mirrors training) --------------------------

    def _build_messages(self, current_url: str) -> list[dict]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # Prior turns: image -> assistant action, optionally with its reasoning.
        n = len(self._history)
        for i, (past_url, past_think, past_action) in enumerate(self._history):
            messages.append({"role": "user", "content": self._user_content(past_url)})
            if not past_action:
                continue
            # Count from the newest turn back, so history_reasoning_len=3 means
            # "the 3 most recent turns argue for themselves, older ones are actions".
            age = n - i
            keep = self.history_reasoning_len < 0 or age <= self.history_reasoning_len
            if keep and past_think:
                messages.append({"role": "assistant",
                                 "content": _think_prefill(past_think, past_action)})
            else:
                messages.append({"role": "assistant", "content": past_action})
        # Current turn (no assistant yet).
        messages.append({"role": "user", "content": self._user_content(current_url)})
        return messages

    def _wants_history_think(self) -> bool:
        return self.history_reasoning_len != 0 and any(t for _, t, _ in self._history)

    def _verify_history_think(self) -> None:
        """Prove the server keeps reasoning in history before trusting a run.

        Two server-side contracts exist:

        - Qwen3-VL stock templates drop <think> from every assistant turn before
          the last user message, silently. Those runs need HISTORY_THINK_TEMPLATE
          on the request, and the old probe (same messages, our template vs
          default) must get longer by roughly the trace.
        - Qwen3.5 CPT templates already render history <think> as-is. The VL
          request template then tokenizes to the same length as default, which
          used to look like "template ignored" and abort a run that would have
          been valid. Probe that case as action-only vs with-think under the
          server default, and do not send a request template.
        """
        if self._history_think_verified:
            return
        action = f"{ACTION_START}0 0 0 ; ; ; ; ; ; {ACTION_END}"
        think_msg = _think_prefill("REASONING_PROBE " * 16, action)
        probe_think = [
            {"role": "user", "content": USER_TURN_TEXT},
            {"role": "assistant", "content": think_msg},
            {"role": "user", "content": USER_TURN_TEXT},
        ]
        probe_act = [
            {"role": "user", "content": USER_TURN_TEXT},
            {"role": "assistant", "content": action},
            {"role": "user", "content": USER_TURN_TEXT},
        ]
        default_think = self._prompt_len(probe_think, None)
        default_act = self._prompt_len(probe_act, None)
        keep_n = ("all" if self.history_reasoning_len < 0
                  else self.history_reasoning_len)
        if (default_think is not None and default_act is not None
                and default_think - default_act >= 16):
            print(f"[history-think] verified: server default keeps think "
                  f"(action-only {default_act} -> with-think {default_think} tokens; "
                  f"reasoning kept in {keep_n} history turns)", flush=True)
            self._history_think_request_template = False
            self._history_think_verified = True
            return

        tpl = HISTORY_THINK_TEMPLATE.read_text()
        with_tpl = self._prompt_len(probe_think, tpl)
        without = (default_think if default_think is not None
                   else self._prompt_len(probe_think, None))
        if with_tpl is None or without is None:
            raise RuntimeError(
                "cannot verify that reasoning survives in history (the server "
                "answered neither /tokenize nor a 1-token completion). Refusing to "
                "run: history_reasoning_len>0 would silently degrade to action-only."
            )
        if with_tpl - without < 16:
            raise RuntimeError(
                f"the server is not keeping <think> in history turns "
                f"(prompt {without} -> {with_tpl} tokens with our chat template, "
                f"default action-only {default_act} -> with-think {default_think}; "
                f"expected ~+16). vLLM may be ignoring the per-request template; "
                f"check its version. Refusing to run a history_reasoning experiment "
                f"that would actually measure the action-only condition."
            )
        # print, not logger: the driver's log file keeps stdout but not this
        # module's logger, and this line is the only evidence a reader has that the
        # run really carried reasoning in its context.
        print(f"[history-think] verified: probe prompt {without} -> {with_tpl} tokens "
              f"with our chat template (reasoning kept in {keep_n} history turns)",
              flush=True)
        self._history_think_request_template = True
        self._history_think_verified = True

    def _prompt_len(self, messages: list[dict], template: str | None) -> int | None:
        """Prompt token count, via /tokenize or a 1-token completion. None if both fail."""
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if template:
            body["chat_template"] = template
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            r = self._session.post(f"{root}/tokenize", json=body, timeout=self.timeout)
            if r.status_code < 400:
                return int(r.json()["count"])
        except (requests.RequestException, KeyError, ValueError, TypeError):
            pass
        try:
            r = self._session.post(f"{self.base_url}/chat/completions",
                                   json={**body, "max_tokens": 1, "temperature": 0},
                                   timeout=self.timeout)
            if r.status_code < 400:
                return int((r.json().get("usage") or {})["prompt_tokens"])
        except (requests.RequestException, KeyError, ValueError, TypeError):
            pass
        return None

    def _user_content(self, data_url: str) -> list[dict]:
        return [
            {"type": "text", "text": USER_TURN_TEXT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    def _data_url(self, image: Image.Image) -> str:
        img = image
        if self.resize_to is not None and img.size != self.resize_to:
            img = img.resize(self.resize_to, Image.BILINEAR)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.image_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
