
from typing import Literal, Dict,List, Any, Union, Optional
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
import random
from time import time
import json
import copy
import os
import math
import re
import torch
import cv2
from rich import print
import traceback

from openagents.agents.base import SEGMENT_MAPPING
from openagents.agents.utils import action_mapping
from openagents.agents.utils.vlm_client import VLMClient
from openagents.agents.utils.system_prompt import get_system_prompt, MODE_SYSTEM_PROMPT_MAP
from openagents.agents import base,base_agent
from openagents.assets import INSTRUCTION_FILE,POLICY_RESOLUTION

from openagents.utils import extract, img_utils
from openagents.agents.base import SAMSession
from openagents.agents.utils.motion_policy import load_motion_policy, MotionPolicy
from minestudio.models import RocketPolicy, load_rocket_policy

OUTPUT_FORMAT_MODE_MAP = {
    "motion_coa": {"fusion","eager","greedy", "motion"},
    "grounding_coa":  {"fusion","eager","greedy", "grounding"},
    "text_action": {"eager", "greedy", "text_action",},
    "grounding":{"eager", "grounding" },
    "motion":{"eager", "motion" },
    None: {None},
}

class OpenHA(VLMClient, base_agent.MineCraftAgent):
    def __init__(
        self, 
        model_path, 
        
        output_mode:Literal["fusion", "eager", "greedy", "grounding", "motion", "text_action"] = "greedy",
        output_format:Optional[Literal["motion_coa", "grounding_coa", "coa", "text_action", "grounding", "motion"]]=None, #offline
        
        motion_policy_path:Optional[str]=None, 
        grounding_policy_path:Optional[str]=None, 
        sam_path:Optional[str]=None, 
        segment_type:str = "Explore", 
        no_approach:bool = False,
        
        grounding_inference_interval: int = 1, 
        motion_inference_interval: int = 1,
        action_chunk_len=1,
        raw_action_type: Literal["reserved", "text", "gui"] = "text",
        gui_reasoning_mode: Literal["direct", "optional", "forced"] = "direct",
        # True when the checkpoint's chat template ends its generation prompt with
        # "<think>\n", so the response carries only the closing </think>.
        gui_think_prefilled: bool = False,
        # Accept a gui action with no <think> wrapper at all (commercial models omit
        # the empty tag our checkpoints always emit). optional mode only.
        gui_allow_missing_think: bool = False,
        # Movement binding accepted by the native GUI decoder (wasd | arrows | both).
        gui_key_map: str = "wasd",
        
        grounding_start_token:str = "Grounding:", grounding_end_token:str = "",
        motion_start_token:str = "Motion:", motion_end_token:str = "",
        action_start_token:str = "Action:", action_end_token:str = "",
        
        maximum_history_length=0,
        gui_history_content: Literal["full", "action_only"] = "full",
        
        vlm_client_mode: Literal["online", "openai", "anthropic", "vllm", "hf", "streaming"] = "online",
        streaming_send_instruction: bool = False,
        instruction_prefix: bool = False,
        per_turn_user_text: str = "",
        native_gui_chat: bool = False,
        native_mouse_pos: str = "[500, 500]",
        model_input_width: Optional[int] = None,
        model_input_height: Optional[int] = None,
        model_url:Optional[str] = None, model_id:str = "", api_key:str = "EMPTY",
        temperature:float = 1.0, 
        top_p:int = 0.99, 
        top_k:float = -1, 
        max_tokens = 1024,
        top_logprobs:int = None,
        get_sampled_logprobs:int=None,
        # Backend-specific sampling params forwarded verbatim as the OpenAI client's
        # extra_body (vlm_client merges its own skip_special_tokens/top_k on top).
        # Needed for checkpoints that keep generating past the first action block:
        # {"stop_token_ids": [151670], "include_stop_str_in_output": true} halts on
        # <|action_end|> and keeps the marker, so one request yields exactly one action.
        extra_body: Optional[dict] = None,
        system_message = None,
        system_message_tag:str = "coa",
        enforce_format: bool = False,
        enforce_prefix:str = "",
        
        LLM_backbone = "", VLM_backbone="",tokenizer_path="",
        
        **kwargs
    ):
        """
        Core multimodal agent class that integrates large language models (LLMs), 
        motion policies, grounding policies, and system prompts. 
        
        Key parameters:
        - output_format:
            Determines the format of agent outputs:
                "motion_coa"     - Motion action with reasoning chain
                "grounding_coa"  - Grounding action with reasoning chain
                "text_action"    - Raw action output in text
                "grounding"      - Grounding action only
                "motion"         - Motion action only

        - vlm_client_mode:
            Defines the backend/mode for VLM inference:
                "online"    - Call API directly
                "openai"    - Use OpenAI interface
                "anthropic" - Use Anthropic interface
                "vllm"      - Use vLLM inference engine
                "hf"        - Use HuggingFace Transformers locally

        - raw_action_type:
            Determines how raw actions are represented:
                "reserved"  - Predefined discrete tokenized actions
                "text"      - Natural language based action text

        - output_mode:
            Controls the strategy for action inference:
                "eager"      - Prefer explicit policies (Grounding / Motion)
                "greedy"     - Always output raw action text
                "grounding"  - Use Grounding policy only
                "motion"     - Use Motion policy only
                "text_action"- Raw action text only
        """
        base_agent.MineCraftAgent.__init__(self, **kwargs)
        super().__init__(
            mode=vlm_client_mode,
            streaming_send_instruction=streaming_send_instruction,
            model_path=model_path,
            api_key=api_key,
            base_url=model_url,
            model_id=model_id,
            LLM_backbone = LLM_backbone, 
            VLM_backbone=VLM_backbone,
            tokenizer_path=tokenizer_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max(max_tokens, 400*(maximum_history_length+1)),
            top_logprobs=top_logprobs,
            get_sampled_logprobs=get_sampled_logprobs,
            extra_body=extra_body,
            limit_mm_per_prompt_image=maximum_history_length+1,
            **kwargs,
        )
        self._action_type = "env"  # For consistency, agent outputs must always be environment actions
        assert output_mode in {"eager", "greedy", "grounding", "motion", "text_action"}, "wrong output_mode"
        self._output_mode = output_mode
        if output_format is None:
            if output_mode in {"text_action"}:
                output_format = "text_action"
            elif output_mode in {"motion"}:
                output_format = "motion"
            elif output_mode in {"grounding"}:
                output_format = "grounding"
            elif output_mode in {"greedy", "eager", "fusion"}:
                # CoA modes do not uniquely determine the output_format from the
                # output_mode alone; derive it from the system_message_tag
                # (grounding_coa / motion_coa), which is what the prompt elicits.
                if system_message_tag in {"grounding_coa", "motion_coa"}:
                    output_format = system_message_tag
        assert output_format in {"motion_coa", "grounding_coa", "text_action", "grounding", "motion"}, "wrong output_format"
        if output_format is not None:
            assert output_mode in OUTPUT_FORMAT_MODE_MAP[output_format]
        self._output_format = output_format
        if enforce_format:
            assert vlm_client_mode=="vllm", f"use enforce_format, only support vllm, but you use {vlm_client_mode}"
            self.enforce_prefix = enforce_prefix
            if not enforce_prefix:
                if output_format in {"grounding_coa", "grounding"}:
                    self.enforce_prefix = grounding_start_token
                elif output_format in {"motion_coa", "motion"}:
                    self.enforce_prefix = motion_start_token
                elif output_format in {"text_action"}:
                    self.enforce_prefix = action_start_token
        self._enforce_format = enforce_format
        self.maximum_history_length = maximum_history_length
        if gui_history_content not in {"full", "action_only"}:
            raise ValueError("gui_history_content must be full|action_only")
        self._gui_history_content = gui_history_content
        self._gui_think_prefilled = bool(gui_think_prefilled)
        # When True (non-streaming modes only), the per-task goal is wrapped as
        # "你当前要完成的任务是：{instruction}" ("The task you need to complete now is: ...") before being appended to the system
        # prompt, matching the phrasing the streaming server injects. Off by
        # default so the OpenHA-native text path is unchanged.
        self._instruction_prefix = instruction_prefix
        # Text placed on EVERY user turn (history + current) in non-streaming
        # modes. Reference OpenHA models were trained with blank per-turn text
        # (default ""), so leaving this empty preserves their behavior. Our
        # streaming-trained model instead saw a per-turn user line ("current game
        # screen") each step; setting this to that string realigns the online
        # multi-image history re-feed with how the model was trained, which stops
        # it collapsing into repeated no-ops when maximum_history_length>0.
        self._per_turn_user_text = per_turn_user_text
        # Faithfully reproduce the training-pipeline chat format for online/vllm/hf
        # modes: a real `system` role message (self.system_message), and every user
        # turn built as "current game screen\n<image>\ncurrent mouse position\n{pos}"
        # (image inline, mouse-position line after). When instruction_prefix is
        # enabled, prepend the task goal to the first user turn for SFT evaluation;
        # leave it disabled for the instruction-free CPT format.
        self._native_gui_chat = native_gui_chat
        self._native_mouse_pos = native_mouse_pos
        if (model_input_width is None) != (model_input_height is None):
            raise ValueError("model_input_width and model_input_height must be set together")
        self._model_input_size = (
            (int(model_input_width), int(model_input_height))
            if model_input_width is not None else None
        )
        
        self._grounding_start_token = grounding_start_token
        self._grounding_end_token = grounding_end_token
        self._motion_start_token = motion_start_token
        self._motion_end_token = motion_end_token
        self._action_start_token = action_start_token
        self._action_end_token = action_end_token
        
        self.action_tokenizer = None
        self.raw_action_type = raw_action_type
        if raw_action_type == "reserved":
            self.action_tokenizer = action_mapping.OneActionTokenizer(tokenizer_type=self.LLM_backbone,action_chunk_len=action_chunk_len)
            self.action_policy_action_mapper = self.action_tokenizer.action_mapper
            self.action_policy_action_transformer = self.action_tokenizer.action_transformer
        elif raw_action_type == "text":
            self.action_tokenizer = action_mapping.TextActionTokenizer(action_chunk_len=action_chunk_len,act_beg_token=self._action_start_token,act_end_token=self._action_end_token)
        elif raw_action_type == "gui":
            # Native GUI actions have two explicit protocols. Thinking models must
            # close <think> before emitting the action; direct models must not emit
            # thinking tags. The tokenizer validates the selected contract.
            self.action_tokenizer = action_mapping.GuiActionTokenizer(
                action_chunk_len=action_chunk_len,
                reasoning_mode=gui_reasoning_mode,
                think_tag_prefilled=gui_think_prefilled,
                allow_missing_think=gui_allow_missing_think,
                key_map=gui_key_map,
            )
        else:
            raise ValueError("wrong raw_action type")
        self.motion_tokenizer =  action_mapping.MotionTokenizer(act_beg_token=self._motion_start_token,act_end_token=self._motion_end_token,keep_no_op_p=1)
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if output_mode in {"eager", "motion"}:
            assert motion_policy_path is not None
            self.motion_policy = self.build_motion_policy(motion_policy_path)
            assert motion_inference_interval > 0
            self.motion_inference_interval = motion_inference_interval
            self._motion_action_pattern = re.compile(rf'{self._motion_start_token}:\s*(.+?)(?=\n|{self._motion_end_token}|$)', re.IGNORECASE)
            self.motion_policy_action_mapper, self.motion_policy_action_transformer = self.init_action_mapper_and_transformer()
        if output_mode in {"eager", "grounding"}:
            assert segment_type is not None
            assert grounding_policy_path is not None
            self._segment_type = segment_type
            self.grounding_policy = self.build_grounding_policy(grounding_policy_path)
            self.sam_session = SAMSession(sam_path, segment_type)
            self.grounding_no_approach = no_approach
            assert grounding_inference_interval > 0
            self.grounding_inference_interval = grounding_inference_interval
            self._grounding_action_pattern = rf"{self._grounding_start_token}\s*([^\s\n]+)\s*([^\n]*?)(?=\n|{self._grounding_end_token}|$)"
            self.grounding_policy_action_mapper, self.grounding_policy_action_transformer = self.init_action_mapper_and_transformer()

        if system_message is None:
            suitable_system_prompt_map = MODE_SYSTEM_PROMPT_MAP.get(self._output_mode, {system_message_tag})
            assert system_message_tag in suitable_system_prompt_map, "wrong system_message tag"
            self.system_message = get_system_prompt(system_message_tag)
        else:
            self.system_message = system_message

        if self.mode == "streaming":
            # The fixed streaming server owns multi-turn KV history. Disable
            # OpenHA-side history (gen_response/refresh_history take their
            # no-history branches).
            #
            # Training format: the task instruction is injected into the SYSTEM
            # prompt ONCE at session creation, while each user turn stays the bare
            # "current game screen". To reproduce that, keep the resolved base
            # system prompt here so the streaming client can send it (with the
            # instruction appended) as custom_system_prompt at session creation;
            # blank system_message so the per-turn user text remains bare.
            self.maximum_history_length = 0
            self._streaming_base_prompt = self.system_message
            self.system_message = ""

        self.reset(task_name="")
    
    def reset(self, instruction:str=None, task_name:str=None, history: Optional[Dict[str,Any]] = None, num:int=1):
        """
        Reset the entire agent state, including history, current task name, 
        cached inference states, and counters. 
        This should be called at the beginning of each new episode/task.
        """
        super().reset()
        self._action_type = "env"
        self.task_name = task_name
        self.num = num
        if instruction is None:
            instruction = random.choice(INSTRUCTION_FILE.get(task_name, [task_name.replace(":"," ").replace("_", " ")]))

        self._instruction = instruction
        self._logprobs = None
        self._sampled_logprobs = None
        self._response_tokens = None
        self._generated_this_step = False
        self._points = [[320,180]]
        self._response = ""
        self._current_obs = None
        self._history = []
        if history:
            self.set_history(history=history)
        self._actions = []
        self._action = None
        self._thought = ""
        self._reasoning_content = ""
        self._policy_type = None
        self._last_policy_type = None
        self._interaction_times = 0
        self._grounding_policy_memory = None 
        self._motion_policy_memory = None
        self._fast_iteration_tick = 0
        self._offline_mode = False
    
    def set_segment_type(self,segment_type ):
        self._segment_type = segment_type
    
    def build_grounding_policy(self, grounding_policy_path=None):
        """
        Build and load the grounding policy model. 
        """
        assert grounding_policy_path is not None, "ROCKET model_path must be specified"
        if os.path.exists(grounding_policy_path):
            policy = load_rocket_policy(grounding_policy_path)
        else:
            policy = RocketPolicy.from_pretrained(grounding_policy_path)
        policy = policy.to(self._device)
        policy.eval()
        self.logger.info("Grounding policy successfully loaded!")
        return policy
    
    def build_motion_policy(self, motion_policy_path=None):
        """
        Build and load the motion policy model.
        """
        assert motion_policy_path is not None, "Motion model_path must be specified"
        if os.path.exists(motion_policy_path):
            policy = load_motion_policy(motion_policy_path)
        else:
            policy = MotionPolicy.from_pretrained(motion_policy_path)
        policy = policy.to(self._device)
        policy.eval()
        return policy
    
    @property
    def device(self):
        return self._device
    
    @property
    def policy_type(self):
        return self._policy_type
    
    @property
    def points(self):
        return self._points
    
    @property
    def response(self):
        return self._response
    
    @property
    def thought(self):
        return self._thought

    @property
    def reasoning_content(self):
        return self._reasoning_content
    
    @property
    def logprobs(self):
        return self._logprobs
    
    @property
    def sampled_logprobs(self):
        return self._sampled_logprobs

    @property
    def response_tokens(self):
        return self._response_tokens

    @property
    def generated_this_step(self):
        return self._generated_this_step
    
    @staticmethod
    def _strip_think(text: str) -> str:
        """Reduce a history response to just its action line.

        Thinking models (e.g. Qwen3.5) reason before answering, and multi-turn
        thinking is meant to keep ONLY the final answer in history. Re-feeding the
        reasoning of all 15 history turns alongside 16 frames blows past the 32k
        context limit, which the server rejects with a 400 — every step of the
        rollout then degrades to a no_op.

        Keying off ``</think>`` is not enough: when the model reasons until it hits
        max_tokens the reply has no closing tag at all (33.3% of steps on an
        unconstrained run), so a tag-based strip passes the entire ramble through and
        that is exactly what overflows the context. So extract the action line
        directly instead — the parser only ever reads ``Action:``, and anything else
        is reasoning that does not belong in history. A reply that never produced an
        action carries no information forward, so it becomes an explicit no_op rather
        than poisoning the next turn with a rambling exemplar."""
        if not text:
            return text
        matches = re.findall(r"Action:.*", text)
        if matches:
            return matches[-1].strip()
        return "Action: no_op"

    def _format_instruction(self, instruction: str) -> str:
        """Optionally wrap the task goal in the system prompt's language.

        English native prompts get an English wrapper and Chinese prompts retain
        the deployment-compatible Chinese wrapper. Returns the raw instruction
        when instruction_prefix is off, or when there is no instruction text.
        """
        if self._instruction_prefix and instruction:
            is_zh = any("\u4e00" <= ch <= "\u9fff" for ch in (self.system_message or ""))
            if is_zh:
                return f"\n你当前要完成的任务是：{instruction}"
            return f"\nYour current task is: {instruction}"
        return instruction

    def _user_turn_text(self, is_first: bool, instruction: str) -> str:
        """Text for one user turn in non-streaming modes. The first turn carries
        the system prompt + task goal; every turn additionally carries the
        per-turn user text when configured. Returns "" when neither applies
        (the original OpenHA behavior for history turns)."""
        parts = []
        if is_first:
            parts.append(self.system_message + self._format_instruction(instruction))
        if self._per_turn_user_text:
            parts.append(self._per_turn_user_text)
        return "\n".join(p for p in parts if p)

    def _build_user_message(self, image, is_first: bool, instruction: str):
        """Build one user turn. In native_gui_chat mode this reproduces the
        training-pipeline per-turn content exactly — text, image, text:
            current game screen
            <image>
            current mouse position
            [500, 500]
        (the system prompt is sent separately as a system role). For SFT models,
        instruction_prefix adds the task goal before "current game screen" on the
        first turn. Otherwise falls back to the OpenHA-native layout."""
        if self._native_gui_chat:
            first_turn_text = ""
            if is_first and self._instruction_prefix:
                first_turn_text = self._format_instruction(instruction) + "\n"
            prompt = [first_turn_text + "current game screen\n"]
            # Some training pipelines contain no mouse-position field. An empty
            # native_mouse_pos means the user turn ends immediately after <image>.
            if self._native_mouse_pos:
                prompt.append(f"\ncurrent mouse position\n{self._native_mouse_pos}")
            return self.processor_wrapper.create_message_vllm(
                role="user", input_type="image",
                prompt=prompt,
                image=image)
        prompt_input = self._user_turn_text(is_first, instruction)
        return self.processor_wrapper.create_message_vllm(
            role="user", input_type="image", prompt=[copy.copy(prompt_input)], image=image)

    def gen_response(self, obs: Dict[str, Any], instruction:str, info: Optional[Dict[str, Any]] = None, verbose:bool=False, if_token_ids:bool=False):
        """
        Generate a response using the large model:
        - Prepares multimodal input (image + text + history).
        - Sends the query to the VLM client backend.
        - Extracts relevant actions and confidence scores.
        - Updates agent’s internal state.
        """
        if info is not None:
            real_obs = info['pov']
        elif isinstance(obs, np.ndarray):
            real_obs = obs
        else:
            raise ValueError("NO OBS")
        # Keep Minecraft's native POV/GUI geometry unchanged and resize only at the
        # model boundary.  The GUI-game training loader used OpenCV bilinear resize
        # from the recorded 640x360 frames to its 1280x720 model canvas.
        if self._model_input_size is not None:
            target_w, target_h = self._model_input_size
            if real_obs.shape[1] != target_w or real_obs.shape[0] != target_h:
                real_obs = cv2.resize(
                    real_obs, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
        image = self.processor_wrapper.create_image_input(real_obs)
        self._current_obs = [image]
        messages = []
        has_history = False
        # native_gui_chat: send the system prompt ONCE as a real system role, matching
        # how the trainer built its samples (messages[0] = {'role':'system',...}).
        if self._native_gui_chat and self.system_message:
            # content as a list of parts (not a bare string) so VLMClient.generate's
            # content-redaction loop and the OpenAI backend both handle it.
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_message}]})
        if self.maximum_history_length:
            history = self._history[-self.maximum_history_length:]
            for hdx,history_message in enumerate(history):
                if history_message is None:
                    continue
                obs, response =  copy.copy(history_message["obs"]), copy.copy(history_message["response"])
                if self.raw_action_type == "gui":
                    # Keep current-turn logging/parsing unchanged. Only the next
                    # request's assistant history is reduced to the executed action
                    # (action_only) or normalised to a complete think+action turn (full;
                    # restores the opening <think> a pre-filling template consumed).
                    if self._gui_history_content == "action_only":
                        response = action_mapping.gui_action_only_history(response)
                    else:
                        response = action_mapping.gui_full_history(
                            response, think_tag_prefilled=self._gui_think_prefilled)
                else:
                    response = self._strip_think(response)
                messages.append(self._build_user_message(obs, not has_history, instruction))
                messages.append(self.processor_wrapper.create_message_vllm(role="assistant",input_type="text",prompt=[response],))
                has_history = True

        is_first_turn = not has_history
        messages.append(self._build_user_message(self._current_obs, is_first_turn, instruction))
        if self._enforce_format:
            messages.append(self.processor_wrapper.create_message_vllm(role="assistant",input_type="text",prompt=[self.enforce_prefix],))
        output = self.generate(messages=messages, verbose=verbose, if_token_ids=if_token_ids)
        content_action,content = output["action"],output["content"]
        self._logprobs = output.get("logprob")
        self._response_tokens = output.get("response_tokens")
        self._sampled_logprobs = output.get("sampled_logprobs")
        if verbose:
            print(content)
            
        self._points = []
        hierarchical_actions = extract.extract_hierarchical_action(content)
        grounding_actions = hierarchical_actions["grounding"]
        for grounding_action in grounding_actions:
            self._points.extend(grounding_action["point"])
        if not self._points:
            self._points = [[320,180]]
        api_reasoning = output.get("reasoning_content") or ""
        if not isinstance(api_reasoning, str):
            api_reasoning = ""
        if isinstance(self.action_tokenizer, action_mapping.GuiActionTokenizer):
            tag_thinking = self.action_tokenizer.parse_thinking(content or "")
        else:
            tag_thinking = action_mapping.parse_gui_thinking(content or "")
        # Visible <think> is the protocol field; commercial APIs (OpenAI
        # reasoning_effort, Anthropic thinking blocks) put the same text in a
        # separate message field that never appears in content.
        self._thought = tag_thinking or api_reasoning
        self._reasoning_content = api_reasoning
        return content_action, content
    
    def create_history(self, actions:List[Dict] = None, observations: List[List[Union[np.ndarray,str, Path,dict ]]] = None, history: Optional[Dict[str,Any]] = None):
        """
        Construct conversation history for the agent.

        Each history entry contains:
            - the observation (pre-processed images)
            - the response (textual action or encoded policy output)

        This allows the agent to maintain temporal consistency across episodes.
        """
        # Truncate
        assert self._output_format is not None, "forget to set output_format"
        if history is None:
            history = []
        else:
            # Ensure history entries have both response and obs fields
            assert all(isinstance(entity, dict) and "response" in entity and "obs" in entity for entity in history)
        if actions is None:
            actions = []
        if observations is None:
            observations = []
            
        # Truncate history to maximum length
        actions = actions[-self.maximum_history_length:]
        observations = observations[-self.maximum_history_length:]
        assert len(actions) == len(observations)

        for observation, action_entity in zip(observations, actions):
            # Ensure observation is always a list of frames
            if not isinstance(observation, list):
                observation = [observation]
            new_observation = []
            for o in observation:
                new_observation.append(self.processor_wrapper.create_image_input(o))
            
            response = ""
            # If skill-level action exists, add it to response
            if "skill_action" in action_entity:
                response += action_entity["skill_action"] + "\n, "
            
            # If need grounding action
            if self._output_format in {"grounding_coa", "grounding"}:
                action_type = action_entity["grounding_action"]["action"]
                points = action_entity["grounding_action"].get("point",)
                if not points:
                    points = []
                else:
                    points = [points]
                caption = action_entity["grounding_action"].get("label","")
                grounding_action = action_mapping.create_grounding_action(
                    point_type=self.VLM_backbone,
                    action_type=action_type,
                    points=points,
                    labels=caption,
                    act_beg_token=self._grounding_start_token,
                    act_end_token=self._grounding_end_token,
                )
                response += grounding_action + "\n, "
            
            raw_action = ""
            env_action = None

            # If need raw action (text or encoded form)
            if self._output_format in {"text_action","motion_coa", "grounding_coa",}:
                if "env_action" in action_entity:
                    env_action = action_entity["env_action"]
                elif "json_action" in action_entity:
                    json_action = action_entity["json_action"]
                    env_action,_ = action_mapping.json_action_to_env_action(json_action)
                trajectory = {"actions":[env_action]}
                try:
                    encoded_trajectory = self.action_tokenizer.encode(trajectory=trajectory)
                    raw_action = encoded_trajectory[0]["action"]
                except Exception as e:
                    print("fail to encode raw actions", e )
                    traceback.print_exc()
            
            # If need motion action (from policy or encoded trajectory)
            if self._output_format in {"motion_coa", "motion"}:
                if "motion_action" in action_entity:
                    motion_action = action_entity["motion_action"]
                else:
                    assert env_action is not None, "Neither motion nor action provided"
                    info = action_entity.get("info", dict())
                    trajectory = {"actions":[env_action], "infos": [info]}
                    try:
                        encoded_trajectory = self.motion_tokenizer.encode(trajectory=trajectory)
                        motion_action = encoded_trajectory[0]["motion_prompt"][0]
                    except Exception as e:
                        print("fail to encode motion actions", e )
                        traceback.print_exc()
                    response += motion_action + "\n, "
                      
            # Always append raw action at the end
            response += raw_action

            response = response.rstrip("\n, ")
            history.append(dict(
                response=response,
                obs=new_observation,
            ))
        
        # When creating history, reset response/cache states
        self._response = None
        self._current_obs = None
        # Update internal history
        self._history = history
        self.refresh_history()
        return history
    
    def refresh_history(self, ):
        """
        Refresh and update history entries.

        Keeps history length within maximum_history_length.
        Adds the latest response + obs if available.
        """
        # Return early if memory is not needed
        if not self.maximum_history_length:
            return
        # Update
        if self._response is not None:
            history_message = dict(
                response=self._response,
                obs=self._current_obs,
            )
            self._history.append(history_message)
        # Trim history to max length
        self._history = self._history[-self.maximum_history_length:]
        return self._history
    
    def set_history(self, history: Dict[str,Any]):
        self._history = history
        self.refresh_history()
    
    def get_history(self):
        return self._history
    
    def get_offline_action(self, obs: Union[np.ndarray,str, Path,dict ], instruction:str = None, history: Optional[Dict[str,Any]] =None, verbose=False):
        """
        Generate an action in offline mode (without real-time environment execution).

        Steps:
        1. Convert observation into numpy image and wrap as dict.
        2. Prepare instruction/history if provided.
        3. Call gen_response() to obtain model response.
        4. Route response to appropriate policy (motion/grounding/raw).
        5. Return the final environment action.
        """
        self._offline_mode = True
        image = img_utils.encode_image_to_np(obs)   # Convert raw obs to numpy image
        obs = dict(image=image)                     # Wrap into dict for downstream policies
        info = dict(pov=image)                      # POV = point of view image for consistency
        if instruction is not None:
            self._instruction = instruction
        if history is not None:
            self.set_history(history=history)
        
        # Get response from model given obs + instruction + info
        raw_action_input, self._response = self.gen_response(obs, self._instruction, info)
        
        # Route response to corresponding action executor
        action = self.router(obs=obs, info=info,response=self._response)
        self._interaction_times += 1
        return action
    
    def get_action(self, obs: Union[Dict[str, Any],np.ndarray], info: Dict[str, Any], instruction:str=None, history: Optional[Dict[str,Any]] =None, verbose=False):
        """
        Main action inference API.
        Based on observation, info, and agent mode, automatically dispatches
        to motion, grounding, or raw action inference.

        Args:
            obs: Current observation (image or dict).
            info: Additional metadata (e.g., inventory, POV).
            instruction: Optional high-level instruction.
            history: Optional past interaction history.
            verbose: Print model output if True.
        """
        # Ensure we are not in offline mode
        assert not self._offline_mode, "use offline mode right now, please reset and try again"  
        obs, info = obs.copy(), info.copy()
        if instruction is not None:
            self._instruction = instruction
        if history is not None:
            self.set_history(history=history)
        self._generated_this_step = not self._fast_iteration_tick
        if self._generated_this_step:
            # Only query LLM if tick counter is expired
            raw_action_input, self._response = self.gen_response(obs, self._instruction, info, verbose=verbose)
            self.refresh_history()
        
        # Route response to appropriate action executor
        action = self.router(obs=obs, info=info,response=self._response)
        self._interaction_times += 1
        return action
    
    def router(self, obs: Dict[str, Any], info: Dict[str, Any], response:str):
        """
        Central routing function for action decisions.

        Logic:
        - In "eager" mode, first try grounding; if uncertain, fall back to motion.
        - In "grounding"/"motion" mode, force corresponding policy.
        - Otherwise, decode raw action.

        Keeps track of:
        - current policy type
        - fast iteration tick (to reduce redundant model calls)
        """
        self._last_policy_type = copy.copy(self._policy_type)
        if not self._fast_iteration_tick:
            self._policy_type = "raw_action"
            if self._output_mode == "eager":
                segment_type, _ = self.extract_segment_type(response)
                if SEGMENT_MAPPING.get(segment_type,-1) != -1:
                    self._policy_type = "grounding"
                elif self.whether_certain_motion(self.get_motion_instruction(response)):
                    self._policy_type = "motion"     
            elif self._output_mode == "grounding":
                self._policy_type = "grounding"
            elif self._output_mode == "motion":
                self._policy_type = "motion"
        
        # Reset fast tick if policy type changes
        if self._last_policy_type != self._policy_type:
            self._fast_iteration_tick = None
        
        # Dispatch to correct policy
        if self._policy_type == "grounding":
            if not self._fast_iteration_tick:
                self._fast_iteration_tick = self.grounding_inference_interval
            action = self.get_action_from_grounding_policy(obs=obs, info=info,response=response)
        elif self._policy_type == "motion":
            if not self._fast_iteration_tick:
                self._fast_iteration_tick = self.motion_inference_interval
            action = self.get_action_from_motion_policy(obs=obs, info=info,response=response)
        elif self._policy_type == "raw_action":
            action = self.get_action_from_action_policy(obs=obs,info=info, response=response)
            # On a fresh decode, get_action_from_action_policy() sets
            # _fast_iteration_tick to the number of buffered actions (GUI substeps /
            # action_chunk_len plan) so the chunk replays without re-querying the model.
            # When it didn't set it (cache-pop tick, or decode failure), fall back to 1
            # = re-infer every tick (unchanged text_action default behavior).
            if not self._fast_iteration_tick:
                self._fast_iteration_tick = 1

        self._fast_iteration_tick -= 1
        return action
    
    def get_action_from_grounding_policy(self, obs: Dict[str, Any], info: Dict[str, Any], response:str):
        """
        Execute action using grounding policy.

        Steps:
        1. Extract grounding instructions (segment type + points).
        2. If parsing fails, fall back to SAM segmentation.
        3. Merge segmentation result into obs.
        4. Call grounding policy to produce action.
        5. Convert model-specific action into environment action.
        """
        obs, info, response = obs.copy(), info.copy(), copy.deepcopy(response)
        segment = self.get_point_instruction(response, obs, info)
        if segment is None:  # fallback segmentation if parsing failed
            segment = self.sam_session.get_segment(obs, info, points=None)
        obs = {**obs, **segment}
        self.segment = segment
        action, self._grounding_policy_memory = self.grounding_policy.get_action(obs, self._grounding_policy_memory, input_shape="*")
        action = self.agent_action_to_env_action(
            action=action, 
            action_mapper=self.grounding_policy_action_mapper, 
            action_transformer=self.grounding_policy_action_transformer,
        )
        return action
    
    def get_action_from_motion_policy(self, obs: Dict[str, Any], info: Dict[str, Any], response:str):
        """
        Execute action using motion policy.

        Steps:
        1. Parse motion instructions from response.
        2. Attach motion info to obs dict.
        3. Downsample image to (128,128) for faster policy inference.
        4. Call motion policy to get action.
        5. Map agent-specific action into environment action.
        """
        obs, info, response = obs.copy(), info.copy(), copy.deepcopy(response)
        motion = {"motion": self.get_motion_instruction(response)}
        self.cur_motion = motion
        self.motion = motion
        obs = {**obs, **motion}
        obs["image"] = cv2.resize(obs["image"], (128, 128))
        action, self._motion_policy_memory = self.motion_policy.get_action(obs, self._motion_policy_memory, input_shape="*")
        action = self.agent_action_to_env_action(
            action=action, 
            action_mapper=self.motion_policy_action_mapper, 
            action_transformer=self.motion_policy_action_transformer,
        )
        return action
    
    def get_action_from_action_policy(self, obs: Dict[str, Any], info: Dict[str, Any], response:str):
        """
        Directly decode environment actions from raw model response.
        
        - If cached actions exist, reuse them (to avoid re-decoding).
        - Otherwise, decode response into env actions.
        - If decoding fails, return a null (no-op) action.
        - Map to env action if using reserved tokenizer.
        """
        obs, info, response = obs.copy(), info.copy(), copy.deepcopy(response)
        if self._actions and self._last_policy_type == "raw_action":
            action = self._actions.pop(0)
        else:
            if isinstance(self.action_tokenizer, action_mapping.GuiActionTokenizer):
                # MineRL reports GUI state every tick; cursor moves live in a different
                # coordinate space than free-camera rotation (see GUI_STATE_AWARE_SCALE).
                actions = self.action_tokenizer.decode(
                    response, is_gui_open=bool(info.get("isGuiOpen", False))
                )
            else:
                actions = self.action_tokenizer.decode(response)
            if not actions:
                # Return null action if decoding fails
                return self.action_tokenizer.env_null_action()
            self._actions = actions
            action = self._actions.pop(0)
            # Replay the remaining buffered actions (GUI substeps, or an
            # action_chunk_len>1 plan) over the next len(self._actions) ticks WITHOUT
            # re-querying the model — keeps the streaming server's history at the
            # model's native window cadence instead of one frame per Minecraft tick.
            # +1 compensates for the decrement at the end of router(). For a single
            # decoded action (text_action default) this is 1 → unchanged behavior.
            self._fast_iteration_tick = len(self._actions) + 1
        if self.raw_action_type == "reserved":
            action = self.agent_action_to_env_action(
                action=action, 
                action_mapper=self.action_policy_action_mapper, 
                action_transformer=self.action_policy_action_transformer
            )
        return action
        
    #---------------------
    
    def get_point_instruction(self, response, obs, info):
        """
        Parse segment type, object name, and points from the model response,
        then call the SAM session to obtain segmentation results.

        """
        segment_type, grounding_response = self.extract_segment_type(response)
        if segment_type == "Approach" and self.grounding_no_approach:
            segment_type = self._segment_type 
        self.sam_session.segment_type = segment_type

        obj, points = self.extract_object_and_points(grounding_response)
        
        if len(points) == 0:
            return {
                'segment': {
                    'obj_mask': torch.zeros(obs["image"][:, :, 0].shape, dtype=torch.uint8),
                    'obj_id': torch.tensor(SEGMENT_MAPPING[self.sam_session.segment_type])
                }
            }

        # Convert from normalized 0–1000 scale to actual pixel coordinates
        points = [(math.ceil((x) * info['pov'].shape[1] / 1000.0),
                   math.ceil((y) * info['pov'].shape[0] / 1000.0)) for x, y in points]

        self._points = points
        segment_type = self.sam_session.segment_type
        segment = self.sam_session.get_segment(obs, info, points, segment_type)
        self.segment = segment
        return segment
    
    def extract_segment_type(self, text):
        """
        Extract the grounding segment type and content using regex.
        """
        grounding_actions = re.findall(self._grounding_action_pattern, text, flags=re.DOTALL)
        if not grounding_actions:
            return self._segment_type, text
        grounding_action = grounding_actions[0][0]
        whole = f"{self._grounding_start_token} {grounding_action} {grounding_actions[0][1]} {self._grounding_end_token}".strip()
        return grounding_action, whole
    
    def extract_object_and_points(self, text):
        """
        Parse object references and associated points/boxes from structured tags.

        """
        if 'point_start' in text:
            # Regex patterns for object and points
            object_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
            points_pattern = r"<\|point_start\|>(.*?)<\|point_end\|>"

            object_match = re.search(object_pattern, text)
            object_name = object_match.group(1) if object_match else None

            points_match = re.search(points_pattern, text)
            points = []
            if points_match:
                points_str = points_match.group(1)
                if ',' not in points_str:
                    return None, []
                points = [tuple(map(float, p.replace('(', '').replace(')', '').split(',')))
                          for p in points_str.split('),(')]

            return object_name, points

        elif 'box_start' in text:
            # Regex for object references and bounding boxes
            object_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
            points_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

            object_match = re.search(object_pattern, text)
            object_name = object_match.group(1) if object_match else None

            points_matches = re.findall(points_pattern, text)
            res_points = []
            if points_matches:
                all_points = []
                for points_str in points_matches:
                    points = [tuple(map(int, p.replace('(', '').replace(')', '').split(',')))
                              for p in points_str.split('),(')]
                    all_points += points
                # Use centroid of all points as the representative point
                point_x = sum(p[0] for p in all_points) // len(all_points)
                point_y = sum(p[1] for p in all_points) // len(all_points)
                res_points = [(point_x, point_y)]
            return object_name, res_points

        else:
            return "null", []
        
    #---------------------
    
    def get_motion_instruction(self, response: str):
        """
        Extract motion instruction string from the response using regex.
        """
        match = self._motion_action_pattern.search(response)
        if match:
            motion = match.group(1)
            # Normalize formatting
            motion = motion.replace(", ", ",")   # remove spaces after commas
            motion = motion.replace(" / ", "/")  # unify separators
            return motion
        else:
            return "none"
    
    def whether_certain_motion(self, motion: str):
        """
        Determine whether a motion instruction is specific and safe enough
        to be executed directly by the motion policy.
        """
        if motion == "none":
            return False
        motion_list = [m.strip() for m in motion.split(',')]
        for m in motion_list:
            if "turn" in m:
                return False
            elif "cursor move" in m:
                return False
        return True
    
    def __str__(self):
        return "openha"
