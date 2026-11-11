"""Policy client: multi-turn Qwen3-VL over an OpenAI-compatible endpoint.

A trimmed port of `omnigamearena/.../models/yourvla.py`, kept byte-compatible in
every way that touches the wire format so a ViZDoom number and an OGA number
describe the same model under the same serving contract:

  - one turn per 200ms chunk, user content = "current game screen" + the frame,
    assistant content = `<|action_start|>...<|action_end|>`;
  - frames resized to 1280x720 (`resize_to`) before JPEG encoding;
  - `stop = ["<|action_end|>"]`, `skip_special_tokens=False`;
  - the same three reasoning styles, with `free` as the standard (the model
    closes its own `<think>` and emits the action itself; nothing is forced).

What is deliberately different: `system_prompt` is per-map (see `prompts.py`),
because a Doom map's goal is not discoverable from the training prompt alone.
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
from typing import Any, NamedTuple

import requests
from PIL import Image

from .action import ACTION_END, ACTION_START, parse_chunk, split_reasoning

logger = logging.getLogger(__name__)

IM_END = "<|im_end|>"
USER_TURN_TEXT = "current game screen"
# GAME-JEV/mc/mc_s1.py's carry_think wording and cap, verbatim.
PREVIOUS_ANALYSIS = "Your previous analysis (from an earlier step; trust the current screen over it): {think}"
THINK_CAP = 1500
TRAIN_RESIZE = (1280, 720)
REASONING_STYLES = ("two_stage", "closed", "free")
# EOS text that can trail an action window when the server keeps special tokens.
_EOS_TEXT = ("<|im_end|>", "<|endoftext|>")
HISTORY_THINK_TEMPLATE = Path(__file__).with_name("chat_template_history_think.jinja")


def _extract_reasoning(text: str) -> str:
    """Only *closed* reasoning is reusable as history; a truncated ramble is not."""
    reasoning, _, closed = split_reasoning(text)
    return reasoning if closed else ""


def cap_think(think: str) -> str:
    think = " ".join(think.split())
    return think if len(think) <= THINK_CAP else think[:THINK_CAP].rsplit(" ", 1)[0] + " ..."


def _think_prefill(reasoning: str = "", content: str = "") -> str:
    return "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n" + content


class Turn(NamedTuple):
    """One past decision, with the frame's wire form kept alongside the frame.

    `url` is what `_data_url` produced for `image` when that frame was current.
    Holding on to it is the whole point of this type: a frame appears in
    `history_len + 1` consecutive prompts, and re-deriving the data URL each time
    costs a resize + a JPEG encode + a base64 pass. Measured at 854x480 with
    history 5 that was ~8 ms per frame, i.e. ~41 ms per step of pure repeat work,
    which is 21% of the 0.8B checkpoint's end-to-end step. Encoding is
    deterministic, so the cached string is byte-identical to the one the old code
    recomputed -- the request on the wire does not change.

    `image` stays for the callers that need the pixels (the probe's multimodal
    cache test builds prompts out of past frames); it is not used to build
    requests any more.
    """

    image: Image.Image
    url: str
    reasoning: str
    action: str


class DoomPolicyAgent:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        base_url: str | None = None,
        api_key: str | None = None,
        history_len: int = 20,
        history_evict_chunk: int = 1,
        history_reasoning_len: int = 0,
        temperature: float = 1.0,
        max_tokens: int = 96,
        resize_to: tuple[int, int] | None = TRAIN_RESIZE,
        image_quality: int = 90,
        timeout: float = 120.0,
        reasoning: bool = True,
        reasoning_style: str = "free",
        reasoning_max_tokens: int = 2048,   # OGA's budget; see doom_suite.yaml
        stop: str | list[str] | None = None,
        api_flavor: str = "vllm",
        recap_last: bool = False,
        carry_think: bool = False,
        image_only_user: bool = False,
    ):
        if api_flavor not in ("vllm", "openai"):
            raise ValueError(f"api_flavor must be vllm or openai, got {api_flavor!r}")
        # 'vllm' is our own server and accepts the extensions the harness relies
        # on (skip_special_tokens, assistant continuation, a request-supplied
        # chat template). 'openai' is a hosted gateway: unknown keys are a 400,
        # so those are dropped and the model has to hit the format unaided.
        self.api_flavor = api_flavor
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = (base_url or os.getenv("VZD_BASE_URL", "http://127.0.0.1:30000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("VZD_API_KEY", "EMPTY")
        self.history_len = int(history_len)
        self.history_evict_chunk = max(1, int(history_evict_chunk))
        self.history_reasoning_len = int(history_reasoning_len or 0)
        # Repeats the previous turn's reasoning + action as text at the head of the
        # current user turn. User content is rendered verbatim by every chat
        # template, unlike reasoning in assistant history turns.
        self.recap_last = bool(recap_last)
        # The Minecraft S1's prompt: the newest non-empty reasoning opens the first
        # user turn of the window (history turns stay action-only), and user turns
        # are the frame alone, as in the CPT's game samples.
        self.carry_think = bool(carry_think)
        self.image_only_user = bool(image_only_user)
        self._carried = ""
        self.n_think_carried = 0
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.resize_to = tuple(resize_to) if resize_to else None
        self.image_quality = int(image_quality)
        self.timeout = float(timeout)
        self.reasoning = bool(reasoning)
        self.reasoning_style = str(reasoning_style)
        if self.reasoning_style not in REASONING_STYLES:
            raise ValueError(f"reasoning_style must be one of {REASONING_STYLES}")
        self.reasoning_max_tokens = int(reasoning_max_tokens)
        if stop is not None:
            self.stop = [stop] if isinstance(stop, str) else list(stop)
        else:
            # `<|action_end|>` is only a safe stop once the model can no longer
            # be inside <think>. Under free-form reasoning it is not: thinking
            # models quote the action format back at themselves while planning
            # ("wait, the format is <|action_start|>X Y Z ; ...<|action_end|>"),
            # and the stop fires on that quote. Generation then ends mid-thought
            # having never emitted an action -- measured on
            # Qwen3-VL-8B-Thinking as 10/10 turns, finish_reason=stop at
            # 800-2700 tokens of a 16384 budget, every response ending at the
            # quoted tag. The run still produced a full four-map table.
            # Under `closed`/`two_stage` the prefill closes </think> for the
            # model, so the tag cannot appear before the real action.
            self.stop = [] if (self.reasoning and self.reasoning_style == "free") else [ACTION_END]

        self._history: deque[Turn] = deque()
        # Set by `_build_messages` for the frame it just encoded, consumed by
        # `act()` when that frame moves into history. The two always run in that
        # order within one decision, and `reset()` clears it.
        self._current_url: str | None = None
        self._history_think_verified = False
        self._history_think_override = False
        self._session = requests.Session()
        self.last_vlm_response: str | None = None
        self.last_predict_ms: float | None = None
        self.last_think_ms: float | None = None   # trt: engine decode time spent inside <think>
        self.last_finish_reason: str | None = None
        self.last_completion_tokens: int | None = None
        self.last_action_end_restored = False
        self.last_stop_reason = None
        self.n_steps = 0
        self.n_unparsed = 0
        self.n_errors = 0

    # -- public API --------------------------------------------------------

    def act(self, image: Image.Image) -> str:
        """One decision: returns the raw model text (the caller parses it)."""
        messages = self._build_messages(image)
        t0 = time.perf_counter()
        self.last_think_ms = None
        try:
            text = self._generate(messages)
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.error("predict failed: %s", exc)
            self.n_errors += 1
            text = ""
        self.last_predict_ms = (time.perf_counter() - t0) * 1000
        self.last_vlm_response = text

        self.n_steps += 1
        # Must agree with the runner on what counts as an action, so the history
        # shows what was actually executed. A bare `ACTION_START in text` does
        # not: it also matches the tag inside a <think> block, where the model is
        # usually quoting the format spec rather than deciding anything.
        chunk = parse_chunk(text)
        if not chunk.parsed:
            self.n_unparsed += 1
        self._history.append(Turn(
            image=image,
            # `_build_messages` above already encoded this exact frame; falling
            # back only covers a caller that bypassed it.
            url=self._current_url or self._data_url(image),
            reasoning=_extract_reasoning(text),
            action=chunk.history_text,
        ))
        self._current_url = None
        self._trim_history()
        if self.carry_think:
            # unclosed reasoning (budget or stop hit mid-trace) is carried too, as in mc_s1
            think = split_reasoning(text)[0]
            if think:
                self._carried = cap_think(think)
                self.n_think_carried += 1
        return text

    def reset(self) -> None:
        self._history.clear()
        self._carried = ""
        self._current_url = None
        self.last_vlm_response = None
        self.last_predict_ms = None
        if os.environ.get('BENCH_EXTERNAL_RUNTIME') == '1':
            from rt_external.step_transport import begin_episode
            begin_episode(self._session, f'{self.base_url}/chat/completions')

    def close(self) -> None:
        # In step mode Session.close also resets only this episode's server KV.
        self._session.close()

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/models",
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  timeout=10.0)
            return r.status_code < 400
        except requests.RequestException as exc:
            logger.warning("health check failed at %s: %s", self.base_url, exc)
            return False

    def warmup(self) -> float:
        """Pay vLLM's first-image-request cost before the episode clock starts.

        /models answers instantly while the vision tower is still uncompiled, so
        a green health check says nothing about the first real frame: measured
        first-call latency on the 0.8B/2B checkpoints is 107-115 SECONDS, against
        a 160-200ms steady state. Under pdq that only made the run slow, but rt
        turns latency into game time, and it destroyed the first episode of all
        ten jobs -- one decision, ~400 tics of dead world, score -1. So the
        warmup frame is thrown away, deliberately, before reset().
        """
        t0 = time.time()
        if self.history_reasoning_len != 0 and self.api_flavor == "vllm":
            # Settle whose chat template renders history reasoning here, not on the
            # first step that happens to produce a non-empty <think>. On a model
            # that mostly emits empty think blocks that step can be hundreds of
            # decisions in, and the check raises when the server drops reasoning --
            # which would kill the run mid-episode instead of before it starts.
            self._verify_history_think()
        try:
            self.act(Image.new("RGB", self.resize_to or (854, 480), (0, 0, 0)))
        except Exception as exc:                    # noqa: BLE001 - never fatal
            logger.warning("warmup call failed (continuing): %s", exc)
        finally:
            self.reset()                            # the throwaway frame must not
            self.n_steps = 0                        # reach the history or any of
            self.n_unparsed = 0                     # the counters the validity
            self.n_errors = 0                       # gate reads
        return (time.time() - t0) * 1000.0

    # -- generation --------------------------------------------------------

    def _generate(self, messages: list[dict]) -> str:
        if not self.reasoning:
            return self._finalize(self._complete(messages, max_tokens=self.max_tokens))
        if self.reasoning_style == "closed":
            return self._forced_action(messages, reasoning="")
        if self.reasoning_style == "two_stage":
            trace = self._complete(messages, max_tokens=self.reasoning_max_tokens,
                                   stop=["</think>"])
            return self._forced_action(messages, reasoning=trace.split("</think>")[0])
        # free: nothing is forced; the model decides when to stop thinking.
        text = self._finalize(
            self._complete(messages, prefill="<think>", max_tokens=self.reasoning_max_tokens)
        )
        if self.api_flavor == "openai":
            # No continuation happened, so nothing was eaten and there is
            # nothing to restore. Prepending `<think>` here would manufacture an
            # unclosed think out of a perfectly well-formed reply and the parser
            # would reject every turn -- which is exactly how a commercial model first
            # scored 20/20 unparsed while actually emitting clean action tags.
            return text
        return text if text.lstrip().startswith("<think>") else "<think>" + text

    def _forced_action(self, messages: list[dict], *, reasoning: str) -> str:
        prefill = _think_prefill(reasoning, ACTION_START)
        body = self._complete(messages, prefill=prefill, max_tokens=self.max_tokens).lstrip()
        while body.startswith(ACTION_START):
            body = body[len(ACTION_START):].lstrip()
        return self._finalize(prefill + body)

    def _complete(self, messages: list[dict], *, max_tokens: int,
                  prefill: str | None = None, stop: list[str] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
            "stop": stop if stop is not None else self.stop,
        }
        if self.api_flavor == "openai":
            # A hosted OpenAI-compatible gateway rejects the whole request on an
            # unknown key, so everything below this line has to stay off: they
            # are vLLM extensions, not OpenAI parameters.
            #
            # Dropping the prefill is not cosmetic. `free` normally seeds the
            # reply with `<think>` and asks vLLM to continue that assistant
            # message; OpenAI has no continuation, so the model gets a plain
            # turn and must produce the `<think>...</think><|action_start|>...`
            # shape entirely from the system prompt. That is the honest test of
            # whether it follows our format, so DO NOT paper over a missing tag
            # by re-adding one here -- an unparsed rate IS the result.
            #
            # `closed` and `two_stage` do not merely seed the reply, they force
            # `<|action_start|>` into it and keep only what follows. Without a
            # continuation they would silently become `free` and be recorded
            # under the wrong style, so refuse instead.
            if prefill and prefill != "<think>":
                raise ValueError(
                    f"reasoning_style={self.reasoning_style!r} needs assistant "
                    f"continuation, which api_flavor='openai' cannot do. Use "
                    f"reasoning_style='free'.")
        else:
            body["skip_special_tokens"] = False
            if prefill is not None:
                body["messages"] = messages + [
                    {"role": "assistant", "content": prefill}]
                body["continue_final_message"] = True
                body["add_generation_prompt"] = False
            if self._wants_history_think() and self._verify_history_think():
                body["chat_template"] = HISTORY_THINK_TEMPLATE.read_text()
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body, timeout=self.timeout,
        )
        if resp.status_code >= 400:
            logger.error("vLLM %d for prefill=%r: %s",
                         resp.status_code, (prefill or "")[:120], resp.text[:500])
        resp.raise_for_status()
        payload = resp.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if self.api_flavor != "openai" and prefill == "<think>":
            # A server started with a reasoning parser does not hand back the
            # text it generated: it keeps everything up to `</think>` in
            # `reasoning` / `reasoning_content` and leaves only the tail in
            # `content`. We seeded the reply with `<think>`, so splicing that
            # tail back on gives `<think>` + action with no closing tag, the
            # parser rejects it, and the run records 100% unparsed / 100% noop
            # for a model that was in fact reasoning and emitting clean action
            # tags. Put the two halves back together instead.
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning and "</think>" not in content:
                content = reasoning + "\n</think>" + content
        # /step engine (trt): decode time spent on the <think> block, from the
        # engine's own token split; absent on plain vLLM, where trt cannot run.
        es = payload.get("engine_step") or {}
        if es.get("cpt_think_tokens") is not None and es.get("gen_tokens"):
            self.last_think_ms = (self.last_think_ms or 0.0) + float(es["decode_ms"]) * int(es["cpt_think_tokens"]) / int(es["gen_tokens"])
        self.last_finish_reason = choice.get("finish_reason")
        # vLLM: the matched stop string, or the stop token id (e.g. 151645 = <|im_end|>).
        self.last_stop_reason = choice.get("stop_reason")
        try:
            self.last_completion_tokens = int((payload.get("usage") or {}).get("completion_tokens") or 0)
        except (TypeError, ValueError):
            self.last_completion_tokens = None
        return content

    def _finalize(self, text: str) -> str:
        """Repair the two ways a well-formed action loses its tags."""
        if text and ACTION_START not in text:
            bare = text.strip()
            if re.match(r"^-?\d", bare) and ";" in bare:
                text = ACTION_START + bare
        # Same rule as the official `finalize`: close an open action
        # window only when generation ended by stopping (the `<|action_end|>`
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

    # -- context -----------------------------------------------------------

    def _trim_history(self) -> None:
        over = len(self._history) - self.history_len
        if over <= 0:
            return
        for _ in range(max(over, self.history_evict_chunk)):
            if not self._history:
                break
            self._history.popleft()

    def _build_messages(self, current: Image.Image) -> list[dict]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        n = len(self._history)
        lead = PREVIOUS_ANALYSIS.format(think=self._carried) if self._carried else ""
        for i, turn in enumerate(self._history):
            messages.append({"role": "user",
                             "content": self._user_content_url(turn.url, lead=lead)})
            lead = ""
            if not turn.action:
                continue
            age = n - i
            keep = self.history_reasoning_len < 0 or age <= self.history_reasoning_len
            if keep and turn.reasoning:
                messages.append({"role": "assistant",
                                 "content": _think_prefill(turn.reasoning, turn.action)})
            else:
                messages.append({"role": "assistant", "content": turn.action})
        self._current_url = self._data_url(current)
        text = USER_TURN_TEXT
        if self.recap_last and self._history:
            last = self._history[-1]
            recap = _think_prefill(last.reasoning, last.action) if last.reasoning else last.action
            if recap:
                text = f"Previous turn:\n{recap}\n\n{USER_TURN_TEXT}"
        messages.append({"role": "user",
                         "content": self._user_content_url(self._current_url, text, lead)})
        return messages

    def _wants_history_think(self) -> bool:
        return self.history_reasoning_len != 0 and any(t.reasoning for t in self._history)

    def _verify_history_think(self) -> bool:
        """Prove the server keeps <think> in history, and say whose template to use.

        Returns True if the request has to carry `HISTORY_THINK_TEMPLATE`, False if
        the model's own template already preserves history reasoning.

        Qwen3-VL's stock template drops reasoning from every assistant turn before
        the last user message, silently, so a run would measure the action-only
        condition and still produce a full results table. That is what the shipped
        override template is for.

        But some CPT checkpoints train with a template that keeps it: the cpt
        think97 runs render a past turn's real `<think>` verbatim and inject an
        empty think block only when the turn has none. Forcing our template on
        those would replace the exact prompt they were trained on -- and the old
        version of this check would also have *refused to run*, because it compared
        our template against the server default and saw no difference, which it
        read as "reasoning is being dropped" when the truth was the opposite.

        So measure the thing we actually care about instead: under the server's own
        template, does a history turn carrying reasoning tokenize longer than the
        same turn without it?
        """
        if self._history_think_verified:
            return self._history_think_override
        action = f"{ACTION_START}0 0 0 ; ; ; ; ; ; {ACTION_END}"
        head = [{"role": "user", "content": USER_TURN_TEXT}]
        tail = [{"role": "user", "content": USER_TURN_TEXT}]
        with_think = head + [{"role": "assistant",
                              "content": _think_prefill("REASONING_PROBE " * 16,
                                                        action)}] + tail
        no_think = head + [{"role": "assistant", "content": action}] + tail
        base = self._prompt_len(no_think, None)
        native = self._prompt_len(with_think, None)
        if base is None or native is None:
            raise RuntimeError(
                "cannot verify that reasoning survives in history; refusing to run "
                "a history_reasoning experiment that would degrade to action-only"
            )
        if native - base >= 16:
            print(f"[history-think] verified: model's own template keeps history "
                  f"reasoning ({base} -> {native} tokens), not overriding it",
                  flush=True)
            self._history_think_override = False
            self._history_think_verified = True
            return False
        ours = self._prompt_len(with_think, HISTORY_THINK_TEMPLATE.read_text())
        if ours is None or ours - base < 16:
            raise RuntimeError(
                f"the server is not keeping <think> in history turns (prompt {base} "
                f"tokens without reasoning, {native} with it under the model's own "
                f"template, {ours} with ours). vLLM needs "
                f"--trust-request-chat-template."
            )
        print(f"[history-think] verified: override template needed "
              f"({base} -> {ours} tokens; model's own template gave {native})",
              flush=True)
        self._history_think_override = True
        self._history_think_verified = True
        return True

    def _prompt_len(self, messages: list[dict], template: str | None) -> int | None:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if template:
            body["chat_template"] = template
        root = self.base_url[:-len("/v1")] if self.base_url.endswith("/v1") else self.base_url
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

    def _user_content(self, image: Image.Image) -> list[dict]:
        return self._user_content_url(self._data_url(image))

    def _user_content_url(self, url: str, text: str = USER_TURN_TEXT,
                          lead: str = "") -> list[dict]:
        if self.image_only_user and text == USER_TURN_TEXT:
            text = ""
        return [
            *([{"type": "text", "text": lead}] if lead else []),
            *([{"type": "text", "text": text}] if text else []),
            {"type": "image_url", "image_url": {"url": url}},
        ]

    def _data_url(self, image: Image.Image) -> str:
        img = image
        if self.resize_to is not None and img.size != self.resize_to:
            img = img.resize(self.resize_to, Image.BILINEAR)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.image_quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class RandomAgent:
    """Smoke-test policy: emits well-formed chunks without a server.

    Used by `scripts/run_benchmark.py --agent random` to prove the env, the
    parser, the trace writer and the aggregator all work before a GPU is booked.
    """

    def __init__(self, seed: int = 0, **_: Any):
        import random
        self._rng = random.Random(seed)
        self.model = "random"
        self.last_predict_ms = 0.0
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 0
        self.last_vlm_response: str | None = None
        self.n_steps = 0
        self.n_unparsed = 0
        self.n_errors = 0

    def act(self, image: Image.Image) -> str:  # noqa: ARG002
        r = self._rng
        x = r.choice([0, 0, 30, -30, 90, -90, 200, -200])
        groups = []
        for _ in range(6):
            keys = []
            if r.random() < 0.55:
                keys.append("LB")
            if r.random() < 0.4:
                keys.append(r.choice(["W", "A", "D", "S"]))
            groups.append(" ".join(keys))
        self.n_steps += 1
        text = f"{ACTION_START}{x} 0 0 ; " + " ; ".join(groups) + f" {ACTION_END}"
        self.last_vlm_response = text
        return text

    def reset(self) -> None:
        pass

    def health_check(self) -> bool:
        return True
