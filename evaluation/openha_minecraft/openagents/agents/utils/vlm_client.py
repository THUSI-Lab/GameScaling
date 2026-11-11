# --- standard library ---------------------------------------------------------
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import os
import time
import random
import uuid
import traceback
from rich import print
import copy
import io
import math
import base64

from rich.console import Console
from rich.logging import RichHandler
import logging

import openai
from openai import OpenAI
from anthropic import Anthropic
import torch

# vllm is imported lazily inside the two `--vlm_client_mode vllm` code paths below.
# Every other mode (notably `online`, which the benchmark uses) talks HTTP to a separate
# vLLM server and needs nothing but `openai`. Importing it at module scope would (a) make
# openagents unusable unless vllm is installed at the version pyproject pins, and (b) cost
# every rollout process ~20s and a lot of RSS — prohibitive at 30-way concurrency, where
# each rollout is its own process.

from transformers import (
    AutoTokenizer,
    AutoModelForImageTextToText,
    Qwen2VLProcessor,
    Qwen2_5_VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from PIL import Image  
import numpy as np 
import cv2  
import requests  
from io import BytesIO 
from openagents.utils import img_utils

ANTHROPIC_MODEL_LIST = {
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet"
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
}

OPENAI_MODEL_LIST = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
}
DEEPSEEK_MODEL_LIST = {
    "deepseek-chat",
    "deepseek-reasoner",
}

PROPRIETART_MODEL_LIST =set()
PROPRIETART_MODEL_LIST.update(DEEPSEEK_MODEL_LIST,OPENAI_MODEL_LIST,ANTHROPIC_MODEL_LIST)

_OPENAI_REASONING_ATTRS = (
    "reasoning_content",
    "reasoning",
    "reasoning_text",
    "thinking",
)


def _coerce_reasoning_text(value: Any) -> Optional[str]:
    """Turn a gateway reasoning payload into a non-empty string, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        parts = [_coerce_reasoning_text(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        for key in ("content", "text", "reasoning", "summary"):
            if key in value:
                text = _coerce_reasoning_text(value[key])
                if text:
                    return text
        return None
    for attr in ("content", "text", "reasoning", "summary"):
        if hasattr(value, attr):
            text = _coerce_reasoning_text(getattr(value, attr))
            if text:
                return text
    return None


def _extract_openai_reasoning(message: Any) -> Optional[str]:
    """Pull hidden reasoning off an OpenAI-compatible chat message.

    Commercial / gateway models (DeepSeek reasoner, vLLM ``--reasoning-parser``,
    OpenAI reasoning models with ``reasoning_effort``) put the chain of thought
    in a side field. ``hasattr(..., "reasoning_content")`` is not enough: the
    SDK often exposes the attribute as ``None``, or the gateway uses
    ``reasoning`` / ``model_extra``.
    """
    candidates: List[Any] = []
    for name in _OPENAI_REASONING_ATTRS:
        if hasattr(message, name):
            candidates.append(getattr(message, name))
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for name in _OPENAI_REASONING_ATTRS:
            if name in extra:
                candidates.append(extra[name])
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(exclude_none=True)
        except TypeError:
            dumped = dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            for name in _OPENAI_REASONING_ATTRS:
                if name in dumped:
                    candidates.append(dumped[name])
    for candidate in candidates:
        text = _coerce_reasoning_text(candidate)
        if text:
            return text
    return None


class VLMClient(object):
    """Unified client wrapper for online/offline VLM back‑ends.
    """

    _TP_SIZE_DEFAULT: int = torch.cuda.device_count() or 1
    _MAX_MODEL_LEN: int = 12288
    _MAX_NUM_SEQS: int = 1
    _GPU_MEM_UTIL: float = 0.40

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: int= 4096,
        top_logprobs: Optional[int] = None,
        get_sampled_logprobs: Optional[bool] = False,
        extra_body: Optional[dict]= None,
        model_id:Optional[str] = None,
        LLM_backbone: Optional[str] = None, 
        VLM_backbone: Optional[str] = None,
        limit_mm_per_prompt_image : Optional[int] = 5,
        limit_mm_per_prompt_video : Optional[int] = 1,
        if_token_ids: bool = False,
        tokenizer_path: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: Literal["online","openai","anthropic", "vllm", "hf", "streaming", ] = "online",
        **kwargs: Any,
    ) -> None:
        
        # logger
        
        self.console: Console = Console()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger(__name__)
        
        # --- public attrs -------------------------------------------------
        self.mode =  mode
        self.temperature = temperature
        self._send_top_p = top_p is not None
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.get_sampled_logprobs = get_sampled_logprobs
        self.extra_body = {} if extra_body is None else extra_body
        self.model_path = str(model_path) if model_path is not None else model_path
        self.__api_key = api_key if api_key is not None else "EMPTY"
        self.base_url = base_url
        self.model_id = model_id
        self.limit_mm_per_prompt_image = limit_mm_per_prompt_image
        self.limit_mm_per_prompt_video = limit_mm_per_prompt_video
        # streaming backend: whether to forward the task instruction to /step.
        # Captured here so _init_streaming_backend (called below) can honour it.
        self._streaming_send_instruction_cfg = bool(kwargs.get("streaming_send_instruction", False))

        # --- toolkits -----------------------------------------------------
        self.LLM_backbone,self.VLM_backbone = load_visual_model(checkpoint_path=model_path,LLM_backbone=LLM_backbone,VLM_backbone=VLM_backbone)
        self.tokenizer = None
        self.if_token_ids = if_token_ids
        if if_token_ids:
            self.tokenizer_path = tokenizer_path if tokenizer_path is not None else self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True) 
        self.processor_wrapper = None  
        self.set_processor_wrapper(model_name=self.VLM_backbone)

        if self.mode == "online" and model_id in ANTHROPIC_MODEL_LIST:
            self.mode = "anthropic"
                
        if self.mode in {"openai","online","online_lmdeploy"}:
            self._init_online_openai_backend()
        elif self.mode == "anthropic":
            self._init_online_anthropic_backend()
        elif self.mode == "vllm":
            self._init_offline_vllm_backend()
        elif self.mode == "hf": # "hf"
            self._init_hf_backend()
        elif self.mode == "streaming":
            self._init_streaming_backend()
        else:
            raise ValueError("wrong mode: ", self.mode)
        
        self.logger.info(f"[VLMClient] initialized in '{self.mode}' mode")

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------
    def _init_online_openai_backend(self) -> None:
        """Initialise OpenAI compatible HTTP backend.

        mode="online" targets a self-hosted vLLM /v1 endpoint and injects vLLM-only
        sampling params (top_k, skip_special_tokens). mode="openai" targets a
        commercial OpenAI-compatible API (OpenAI, DashScope compatible-mode, ...)
        which rejects unknown params with HTTP 400, so nothing is injected there
        and only what the caller put in extra_body is forwarded.
        """
        if self.mode == "online":
            self.extra_body["top_k"] = self.top_k
        if self.mode in {"online", "online_lmdeploy"}:
            self.extra_body["skip_special_tokens"] = False
        api_key = self.__api_key
        if self.mode == "openai" and api_key in (None, "", "EMPTY"):
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=120.0)
        if not self.model_id:
            models = self.client.models.list()
            self.model_id = models.data[0].id  # choose first available
        self.logger.debug(f"[OpenAI] Using model: {self.model_id}, base_url: {self.base_url}")
        
    def _init_online_anthropic_backend(self) -> None:
        """Initialise Anthropic compatible HTTP backend."""
        self.client = Anthropic(api_key=self.__api_key, base_url=self.base_url)
        if not self.model_id:
            models = self.client.models.list()
            self.model_id = models.data[0].id  # choose first available
        self.logger.debug(f"[Anthropic] Using model: {self.model_id}")

    def _init_streaming_backend(self) -> None:
        """Initialise the bridge to the fixed streaming server.

        That server (the streaming server) is *not* OpenAI-compatible: it
        exposes a custom, stateful ``POST /step`` (one image per call, per-session
        KV-cache history) and ``POST /reset``. It also injects its own system
        prompt at session creation, so the OpenHA side must NOT send a system
        prompt and must let the server own the multi-turn history
        (see OpenHA.__init__: maximum_history_length=0, system_message="").
        """
        base = (self.base_url or "").rstrip("/")
        if not base:
            raise ValueError("streaming mode requires base_url (e.g. http://localhost:5000)")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        self.step_url = f"{base}/step"
        self.reset_url = f"{base}/reset"
        self.sessions_url = f"{base}/sessions"
        # Unique per VLMClient instance (== per rollout). Stable across this
        # rollout's /step calls so they share one KV session, but distinct from
        # other concurrent rollouts in the same process (--parallel_tasks > 1),
        # which would otherwise collide on a shared pid-based id and corrupt each
        # other's server-side history. Cleared each episode via reset() (POST /reset).
        self.session_id = f"openha-{self.model_id or 'default'}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        # Whether the model should receive the task instruction. Controlled by
        # OpenHA(streaming_send_instruction=...) -> --streaming_send_instruction.
        # When True, the instruction is injected into the SYSTEM prompt ONCE at
        # session creation (via custom_system_prompt), matching the SFT training
        # format. The per-step `instruction` field is always empty so each user
        # turn stays the bare "current game screen". When False, no instruction is
        # given (base system prompt only).
        self.streaming_send_instruction = self._streaming_send_instruction_cfg
        # Base system prompt sent (with the instruction appended) as
        # custom_system_prompt. OpenHA.__init__ sets this to the resolved system
        # prompt for streaming mode; default empty -> server uses its own default.
        self._streaming_base_prompt = getattr(self, "_streaming_base_prompt", "")
        self.streaming_timeout = (10, 300)  # (connect, read); server inference is serialized & slow
        self._http = requests.Session()
        # Best-effort clear in case a stale session lingers from a crashed/killed run.
        try:
            r = self._http.get(self.sessions_url, timeout=(5, 10))
            if r.ok:
                prefix = f"openha-{self.model_id or 'default'}-"
                for entry in r.json().get("sessions", []):
                    sid = entry.get("session_id", "")
                    if not sid.startswith(prefix):
                        continue
                    parts = sid[len(prefix):].split("-")
                    stale = True
                    if parts and parts[0].isdigit():
                        try:
                            os.kill(int(parts[0]), 0)
                            stale = False
                        except OSError:
                            stale = True
                    if stale:
                        self._http.post(self.reset_url, json={"session_id": sid},
                                        timeout=self.streaming_timeout)
        except Exception:
            pass
        try:
            self._http.post(self.reset_url, json={"session_id": self.session_id},
                            timeout=self.streaming_timeout)
        except Exception:
            pass
        self.logger.info(f"[VLMClient] streaming backend -> {self.step_url} session={self.session_id}")

    def _init_offline_vllm_backend(self) -> None:
        """Initialise local vLLM engine."""
        from vllm import LLM  # lazy: only --vlm_client_mode vllm needs vllm installed

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self._TP_SIZE_DEFAULT,
            max_model_len=self._MAX_MODEL_LEN,
            max_num_seqs=self._MAX_NUM_SEQS,
            gpu_memory_utilization=self._GPU_MEM_UTIL,
            trust_remote_code=True,
            limit_mm_per_prompt={"image":self.limit_mm_per_prompt_image, "video": self.limit_mm_per_prompt_video},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
        )
        if  self.VLM_backbone == "qwen2_vl":
            self.processor = Qwen2VLProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                do_rescale=False,
                patch_size=14,
                vision_feature_select_strategy="default",
            )
        elif self.VLM_backbone == "qwen2.5_vl":
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                do_rescale=False,
                patch_size=14,
                vision_feature_select_strategy="default",
            )

    def _init_hf_backend(self) -> None:
        """Initialise HuggingFace transformers model backend."""
        processor_cfg: Dict[str, Any] = {
            "patch_size": 14,
            "vision_feature_select_strategy": "default",
        }
        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.float16}
        if self.VLM_backbone == "qwen2_vl":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            processor_cfg.update({
                "min_pixels": 224 * 224,
                "max_pixels": 2048 * 2048,
            })
            self.processor = Qwen2VLProcessor.from_pretrained(self.model_path, trust_remote_code=True, **processor_cfg)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cuda:0",
                **model_kwargs,
            ).eval()
        elif self.VLM_backbone == "qwen2.5_vl":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            processor_cfg.update({
                "min_pixels": 224 * 224,
                "max_pixels": 2048 * 2048,
            })
            self.processor = Qwen2_5_VLProcessor.from_pretrained(self.model_path, trust_remote_code=True,  **processor_cfg)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cuda:0",
                **model_kwargs,
            ).eval()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="cuda:0",
                **model_kwargs,
            ).eval()

    def reset(self) -> None:
        """Refresh HTTP client (token expiry, etc.)."""
        if self.mode in {"openai","online"}:
            self._init_online_openai_backend()
            self.logger.debug("[VLMClient] HTTP client reset")
        elif self.mode == "anthropic":
            self._init_online_anthropic_backend()
            self.logger.debug("[VLMClient] HTTP client reset")
        elif self.mode == "streaming":
            # New episode: clear the server-side KV-cache for our session so the
            # next /step starts a fresh history. Non-fatal — the first /step will
            # re-create the session even if this fails.
            try:
                self._http.post(self.reset_url, json={"session_id": self.session_id},
                                timeout=self.streaming_timeout)
                self.logger.debug(f"[VLMClient] streaming session reset: {self.session_id}")
            except Exception as e:
                self.logger.warning(f"[VLMClient] streaming /reset failed: {e}")

    def set_processor_wrapper(self, model_name: Optional[str] = None) -> None:
        model_name = model_name or self.model_id
        self.processor_wrapper = ProcessorWrapper(model_name=model_name, mode = self.mode)

    def generate(self, messages: List[dict[str, Any]], *, verbose: bool = False, if_token_ids: bool = False, if_token_record: bool = True) -> Dict[str, Any]:
        """Generate response.

        Returns
        -------
        dict with keys ``content``, ``action`` and optionally ``logprob``.
        """
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        if_token_ids = if_token_ids or self.if_token_ids
        verbose_messages = copy.deepcopy(messages)
        for message in verbose_messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    content["image_url"] = ""
                    
        self.logger.debug(f"[generate/{self.mode}] {verbose_messages}")
        if self.mode in {"openai","online",}:
            outputs = self._generate_openai(messages, if_token_ids,if_token_record=if_token_record)
        elif self.mode == "anthropic":
            outputs = self._generate_anthropic(messages, if_token_ids,if_token_record=if_token_record)
        elif self.mode == "vllm":
            outputs =  self._generate_offline_vllm(messages, if_token_ids)
        elif self.mode == "streaming":
            outputs = self._generate_streaming(messages)
        else:
            outputs = self._generate_local_hf(messages, if_token_ids)
        self.logger.debug(f"[generate/{self.mode}] done -------------- ")
        return outputs
    
    def _generate_streaming(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Forward the current step to the fixed streaming /step server.

        Extracts the current frame (and, optionally, the bare instruction) from
        the last user message and POSTs it to /step. The server keeps multi-turn
        KV history per ``session_id``. Returns the raw ``action_text`` (which
        contains ``<|action_start|>...<|action_end|>``) for GuiActionTokenizer.
        """
        # 1) locate the last user message
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg
                break
        if user_msg is None:
            raise ValueError("streaming: no user message found")

        # 2) extract text + image (order-independent)
        text_parts: List[str] = []
        image_url: Optional[str] = None
        for part in user_msg.get("content", []):
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype == "image_url":
                image_url = part.get("image_url", {}).get("url")
            elif ptype == "image":  # tolerate hf-style, just in case
                image_url = part.get("image")
        if not image_url:
            raise ValueError("streaming: no image in last user message")

        # 3) data-url -> raw base64
        image_b64 = image_url.split(",", 1)[1] if str(image_url).startswith("data:") else image_url

        # 4) Training-aligned instruction handling. The task instruction (the text
        #    of the last user message, or self._instruction when native_gui_chat
        #    formatted the user turn) is injected into the SYSTEM prompt ONCE, at
        #    session creation, via custom_system_prompt — using the same wrapper as
        #    the SFT data. The per-step `instruction` field stays empty so every
        #    user turn is the bare "current game screen", matching training.
        raw_instr = getattr(self, "_instruction", "") or ""
        text = raw_instr.strip() or "".join(text_parts).strip()  # the task instruction / goal
        base_prompt = getattr(self, "_streaming_base_prompt", "") or ""
        if self.streaming_send_instruction and text:
            # Training-aligned: the SFT model is trained with the task instruction in
            # the SYSTEM prompt (never the user turn) — see training-pipeline
            # data_preprocess/sft_data_merge.py, where every user turn is the bare
            # "current game screen\n<image>" (no mouse position) and the instruction
            # lives in the system message. We append it in the SAME language as the
            # base system prompt (detected via CJK chars) so we don't mix languages:
            #   en (gui_native_en) -> the EXACT English wrapper the trainer used:
            #       "\n\n    Please complete the task according to the following
            #        instructions\n    <goal>\n    "  (sft_data_merge.instruction_prompt)
            #   zh (gui_native)    -> the Chinese deployment wrapper used by the live
            #       client inference-server/client/agent.py: "你当前要完成的任务是：<goal>" ("The task you need to complete now is: <goal>")
            _is_zh = any("\u4e00" <= ch <= "\u9fff" for ch in base_prompt)
            if _is_zh:
                custom_system_prompt = base_prompt + f"\n你当前要完成的任务是：{text}"
            else:
                custom_system_prompt = (
                    base_prompt
                    + "\n\n    Please complete the task according to the following instructions\n    "
                    + text
                    + "\n    "
                )
        else:
            custom_system_prompt = base_prompt

        # Log the FULL system prompt once per session (shows with --verbose True).
        # This is what the server injects at session creation; it's identical every
        # step, so we only print it the first time to avoid per-frame spam.
        if not getattr(self, "_logged_stream_sys_prompt", False):
            self.logger.debug(
                f"[streaming] custom_system_prompt (session={self.session_id}):\n"
                f"------- BEGIN SYSTEM PROMPT -------\n{custom_system_prompt}\n"
                f"------- END SYSTEM PROMPT -------"
            )
            self._logged_stream_sys_prompt = True

        # 5) POST /step with a small retry; on hard failure return empty (-> no_op).
        #    custom_system_prompt is honored by the server only when it creates the
        #    session (first call after /reset); subsequent calls reuse the session.
        payload = {
            "session_id": self.session_id,
            "image_b64": image_b64,
            "instruction": "",
            "custom_system_prompt": custom_system_prompt,
            "norm_xy": [500, 500],
        }
        action_text = ""
        for attempt in range(36):
            try:
                resp = self._http.post(self.step_url, json=payload, timeout=self.streaming_timeout)
                resp.raise_for_status()
                action_text = resp.json().get("action_text", "") or ""
                while action_text.rstrip().endswith("<|im_end|>"):
                    action_text = action_text.rstrip()[:-len("<|im_end|>")].rstrip()
                break
            except Exception as e:
                self.logger.warning(f"[VLMClient] streaming /step failed (attempt {attempt + 1}): {e}")
                time.sleep(5)
        return {"content": action_text, "action": action_text}

    def _generate_openai(self, messages: List[dict[str, Any]], if_token_ids: bool= False,if_token_record:bool= False,) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {}
        
        if self._send_top_p:
            gen_kwargs["top_p"] = self.top_p
        if self.top_logprobs:
            gen_kwargs.update({"top_logprobs": self.top_logprobs, "logprobs": True})
        elif self.get_sampled_logprobs:
            gen_kwargs.update({"logprobs": True})
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body,
                **gen_kwargs,
            )
            content_choice = chat_completion.choices[0]
            if if_token_record:
                token = {
                    "input" : chat_completion.usage.prompt_tokens,
                    "output" : chat_completion.usage.completion_tokens,
                }
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError):
            # Bad key / no access / unknown model id never recovers on retry. Swallowing it
            # would turn a 600-step rollout into 600 silent no_ops scored as a loss, so
            # surface it and let the rollout die.
            raise
        except Exception as e:
            # NOTE: chat_completion may be undefined if create() itself raised — don't
            # reference it (that UnboundLocalError used to mask the real API error).
            # A single bad step (e.g. a transient 5xx, or a 400 context overflow) must
            # NOT kill the whole rollout: degrade to an empty response so the agent
            # emits a no_op for this step and keeps going.
            traceback.print_exc()
            print(f"[vlm_client] chat.completions.create failed, returning no_op for this step: {e}")
            return {"content": "", "action": ""}
        try:
            content = content_choice.message.content
            action = self._maybe_tokenize(content, if_token_ids)
            outputs: Dict[str, Any] = {"content": content, "action": action}
            if self.top_logprobs:
                outputs["logprob"] = [
                    [(prob.token, prob.logprob) for prob in logprob.top_logprobs]
                    for logprob in content_choice.logprobs.content
                ]
                outputs["response_tokens"] = [[tokenlogprob.token, tokenlogprob.logprob] for tokenlogprob in content_choice.logprobs.content]
            if self.get_sampled_logprobs:
                outputs["sampled_logprobs"] = [
                    [logprob.token, logprob.logprob] for logprob in content_choice.logprobs.content
                ]
            if if_token_record:
                outputs["token"] = token
            reasoning = _extract_openai_reasoning(content_choice.message)
            if reasoning:
                outputs["reasoning_content"] = reasoning
        except Exception as e:
            print(e)
            traceback.print_exc() 
            raise Exception(f"error: {e}")
        return outputs
    
    def _generate_anthropic(self, messages: List[Dict[str, Any]], if_token_ids: bool= False,if_token_record:bool= False,) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = copy.copy(self.extra_body)
        if "thinking" in self.extra_body:
            temperature = 1
        if messages[0].get("role","") == "system":
            extra_body["system"] = messages[0]["content"]
            messages = messages[1:]
        try:
            chat_completion = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                **extra_body,
            )
        except Exception as e:
            print(chat_completion)
            raise Exception(f"error: {e}")
        outputs: Dict[str, Any] = {}
        for block in chat_completion.content:
            if block.type == "thinking":
                outputs["reasoning_content"] = block.thinking
            elif block.type == "text":
                content = block.text
                outputs["content"] = content
                outputs["action"] = self._maybe_tokenize(content, if_token_ids)
        if if_token_record:
            outputs["token"] = {
                "input" : chat_completion.usage.input_tokens,
                "output": chat_completion.usage.output_tokens,
            }
        return outputs

    def _generate_offline_vllm(self, messages: List[dict[str, Any]], if_token_ids: bool) -> Dict[str, Any]:
        from vllm import SamplingParams  # lazy: see the note at the top of this module

        gen_kwargs: Dict[str, Any] = {}
        if self.top_logprobs:
            gen_kwargs.update({"top_logprobs": self.top_logprobs, "logprobs": True})

        sampling = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k =  self.top_k,
            top_p = self.top_p,
            skip_special_tokens=False,
            **gen_kwargs,
        )
        
        assistant_text = ""
        while messages and messages[-1]["role"] == "assistant":
            for content in messages[-1]["content"]:
                assistant_text += content["text"]
            messages = messages[:-1]
            
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += assistant_text
        images: List[Image.Image] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            for part in msg["content"]:
                if part["type"] == "image":
                    img = encode_image_to_pil(part["image"])
                    images.append(img)
                elif part["type"] == "image_url":
                    img = encode_image_to_pil(part["image_url"]['url'])
                    images.append(img)
                else:
                    pass
                    
        if not images:
            images = None
        
        inputs = {
            "prompt": prompt,
        }
        if images is not None:
            inputs["multi_modal_data"] = {"image": images}
            
        model_outputs = self.model.generate(
            inputs,
            sampling_params=sampling,
        )
        output_text = assistant_text + model_outputs[0].outputs[0].text
        #output_text = self.model.chat(messages, sampling_params=sampling, chat_template=self.processor.chat_template)[0].outputs[0].text

        outputs: Dict[str, Any] = {
            "content": output_text,
            "action": self._maybe_tokenize(output_text, if_token_ids),
        }
        self.logger.debug("[generate/vllm] done")
        return outputs

    def _generate_local_hf(self, messages: List[dict[str, Any]], if_token_ids: bool) -> Dict[str, Any]:
        images: List[Image.Image] = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            for part in msg["content"]:
                if part["type"] == "image":
                    img = encode_image_to_pil(part["image"])
                    part["image"] = img
                    images.append(img)
        if not images:
            images = None
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=images, padding=True, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_cfg = {
            "do_sample": bool(self.temperature),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": 1.0,
        }
        if self.top_k > 0:
            gen_cfg["top_k"] = self.top_k
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens - input_len,
            **gen_cfg,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )[0][input_len:]
        content = self.processor.batch_decode([output_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        outputs: Dict[str, Any] = {"content": content, "action": output_ids if if_token_ids else content}
        self.logger.debug("[generate/hf] done")
        return outputs

    def _maybe_tokenize(self, text: str, flag: bool) -> Union[str, List[int]]:
        if not flag:
            return copy.deepcopy(text)
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialised, cannot return token ids")
        return self.tokenizer(text)["input_ids"]


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int,max_ratio:int
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(image: Image.Image,  factor: int, min_pixels: int, max_pixels: int,max_ratio:int) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_ratio=max_ratio,
        )
    image = image.resize((resized_width, resized_height))
    return image

def pil2base64(image):
    """Force the intermediate result to be JPEG.""" 
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def encode_image_to_pil(image_input):
    if isinstance(image_input, str) and image_input[:len("data:image")] == "data:image":
        try:
            base64_str = image_input.split(",")[1]
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception:
            raise ValueError("Invalid base64 image data or unsupported format.")
    elif isinstance(image_input, (str, Path)): 
        try:
            img = Image.open(image_input)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except IOError:
            raise ValueError("Could not open the image file. Check the path and file format.")
    elif isinstance(image_input, np.ndarray):
        try:
            img = Image.fromarray(image_input)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except TypeError:
            raise ValueError("Numpy array is not in an appropriate format to convert to an image.")
    elif isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        return image_input
    else:
        raise TypeError("Unsupported image input type. Supported types are str, pathlib.Path, numpy.ndarray, and PIL.Image.")
    

def encode_image_to_base64(image:Union[str,Path,Image.Image,np.ndarray], format='JPEG') -> str:
    """Encode an image to base64 format, supports URL, numpy array, and PIL.Image."""

    # Case 1: If the input is a URL (str)
    image_encode = None
    if isinstance(image, str) and image[:4]=="http":
        try:
            response = requests.get(image)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve the image from the URL: {e}")
    elif isinstance(image, str) and image[0]=='/':
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    elif isinstance(image,Path):
        with image.open('rb') as image_file:
            image_encode =  base64.b64encode(image_file.read()).decode('utf-8')
        return image_encode
    # Case 3: If the input is a numpy array
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode('utf-8')

    # Case 4: If the input is a PIL.Image
    elif isinstance(image, Image.Image):
        # History frames are re-sent on every turn (maximum_history_length of them);
        # memoize the encoding on the frame object so each frame is JPEG/base64
        # encoded once. Identical bytes also keep the server-side mm-cache hash stable.
        cache_attr = f"_openha_b64_{format}"
        cached = getattr(image, cache_attr, None)
        if cached is not None:
            return cached
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        image_bytes = buffer.read()
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        try:
            setattr(image, cache_attr, encoded)
        except AttributeError:
            pass
        return encoded

    # Raise an error if the input type is unsupported
    else:
        raise ValueError("Unsupported input type. Must be a URL (str), numpy array, or PIL.Image.")

def get_suffix(image:Union[list,str,Path,np.ndarray,Image.Image]):
    if isinstance(image,np.ndarray|Image.Image):
        image_suffix = 'jpeg'
    elif isinstance(image,str):
        image_suffix = image.split(".")[-1]
    elif isinstance(image,Path):
        image_suffix = image.suffix[1:]
    else:
        raise ValueError(f"invalid image type！")
    return image_suffix

def translate_cv2(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        # Convert PIL Image to NumPy array (PIL is in RGB)
        img_array = np.array(image)
        cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Check if the NumPy array is in RGB format and has three channels
        if image.shape[2] == 3:  # Only for color images
            cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            cv2_image = image  # No conversion needed for grayscale images
    elif isinstance(image, (str, Path)):
        # Read the image using cv2 (assumes BGR format)
        cv2_image = cv2.imread(str(image))  # Convert PosixPath to string if necessary
        if cv2_image is None:
            raise ValueError(f"The image path is incorrect or the file is not accessible: {image}")
    else:
        raise ValueError("Unsupported image format or path type")
    
    return cv2_image

    

class ProcessorWrapper:
    def __init__(self, model_name= "qwen2_vl",mode="online"):
        self.mode = mode
        self.model_name = model_name.replace("-","_")
        self.image_factor = 28
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = 1024 * 28 * 28  #16384 * 28 * 28
        self.max_ratio = 200

    def get_image_message(self,source_data:Union[str,Path,np.ndarray,Image.Image]):
        image_suffix = get_suffix(source_data)
        image = None
        if self.mode == "hf":
            image_message = {
                "type": "image",
                "image": source_data,
            }
        elif self.mode == "anthropic":
            image_suffix = "jpeg" if image_suffix=="jpg" else image_suffix
            image_message = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image_suffix}",
                        "data": encode_image_to_base64(source_data)
                    },
                }
        else:
            image = { "url": f"data:image/{image_suffix};base64,{encode_image_to_base64(source_data)}"}
            image_message = {
                    "type": "image_url",
                    "image_url": image,
                }
        return image_message

    def create_system_prompt(self,system_prompt:str=""):
        message = {
            "role":"system",
            "content": f"{system_prompt}\n",
        }
        return message

    def create_message_vllm(self,
                            role:Literal["user","assistant"]="user",
                            input_type:Literal["image","text"]="image",
                            image:Union[list,str,Path,np.ndarray,Image.Image]=None,
                            prompt:Union[list,str]="",):
        if role not in {"user","assistant"}:
            raise ValueError(f"a invalid role {role}")
        if isinstance(prompt,str):
            prompt = [prompt]
        message = {
            "role": role,
            "content": [],
        }
        if input_type=="image":
            if not isinstance(image,list):
                image = [image]
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}"
                })
                if idx < len(image):
                    message["content"].append(self.get_image_message(image[idx]))
            for idx in range(len(prompt), len(image)):
                message["content"].append(self.get_image_message(image[idx])) 
        else:
            for idx, text in enumerate(prompt):
                message["content"].append({
                    "type": "text",
                    "text": f"{text}"
                })
        return message
    
    def create_text_input(self,conversations:list):
        text_prompt = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
        return text_prompt
    
    def create_image_input(self,image_input:Union[np.ndarray,str, Path,Image.Image ]):
        image = img_utils.encode_image_to_pil(image_input)
        # image = image_pixels
        # if image_path:
        #     image = Image.open(image_path)
        # if not isinstance(image, Image.Image):
        #     image = Image.fromarray(image.astype('uint8'))
        if "qwen2_vl" in self.model_name:
            image = fetch_image(image,factor=self.image_factor,min_pixels=self.min_pixels,max_pixels=self.max_pixels,max_ratio=self.max_ratio,)
        return image

def load_visual_model(checkpoint_path ="",LLM_backbone:Optional[str]=None,VLM_backbone:Optional[str]=None,**kwargs):
    if not checkpoint_path:
        return "",""
    
    checkpoint_path = checkpoint_path.lower().replace('-','_').replace("qwen2vl","qwen2_vl").replace("qwen2.5vl","qwen2.5_vl")
    
    if LLM_backbone is not None and LLM_backbone!="":
        pass
    elif "mistral" in checkpoint_path:
        LLM_backbone = "mistral"
    elif "vicuna" in checkpoint_path:
        LLM_backbone = "llama-2"
    elif "llama_3" in checkpoint_path or "llama3" in  checkpoint_path:
        LLM_backbone = "llama-3"
    elif "qwen2_vl" in checkpoint_path:
        LLM_backbone = "qwen2_vl"
    elif "qwen2.5_vl" in checkpoint_path:
        LLM_backbone = "qwen2.5_vl"
    else:
        LLM_backbone = ""
       
    if VLM_backbone is not None and VLM_backbone!="":
        pass
    elif 'llava_next' in checkpoint_path or 'llava_v1.6'  in checkpoint_path:
        VLM_backbone = "llava-next"
    elif "qwen2_vl" in checkpoint_path:
        VLM_backbone = "qwen2_vl"
    elif "qwen2.5_vl" in checkpoint_path:
        VLM_backbone = "qwen2.5_vl"
    else:
        VLM_backbone = ""

    return LLM_backbone,VLM_backbone
    
if __name__ == "__main__":
    import os
    from ultron.dataset.api import message_generate
    vlm_client = VLMClient(model_path="/share_data/limuyao/checkpoints/train/mc-motion-coa-qwen2-vl-7b-250809-A800-c32-e1-b8-a1/checkpoint-3850",if_token_ids=False,base_url="http://localhost:11000/v1",mode="online",get_sampled_logprobs=1)
    vlm_client.reset()
    messages = []
    messages.append(message_generate.create_system_message(system_prompt="\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <think> {thought with steps separated with '\n\n'} </think> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, Now, try to solve the following question through the above guidelines:\n",model_type="gpt-4o"))
    messages.append(message_generate.create_user_message(input_type="text",user_prompt="Give you nothing in the inventory, generate a step-by-step plan to obtain diamonds.",model_type="gpt-4o"))
    a = vlm_client.generate(messages=messages)
    print(a["sampled_logprobs"])