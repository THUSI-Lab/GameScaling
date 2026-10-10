"""Inference-side contract of the GameScaling checkpoints: prompt, client, action decoding."""

from .action import (
    ACTION_END,
    ACTION_START,
    CHUNK_MS,
    CHUNK_STEPS,
    KEY_VOCAB,
    NOOP_ACTION_TEXT,
    STEP_MS,
    Action,
    parse_action,
    split_reasoning,
)
from .policy import (
    NO_THINK_PREFILL,
    THINK_PREFILL,
    Decision,
    GameScalingPolicy,
    PolicyConfig,
    build_request,
    encode_image,
    finalize,
)
from .prompt import (
    USER_TURN_TEXT,
    build_system_prompt,
    translate_controls,
)

__version__ = "0.1.0"

__all__ = [
    "ACTION_END", "ACTION_START", "CHUNK_MS", "CHUNK_STEPS", "KEY_VOCAB", "NOOP_ACTION_TEXT",
    "STEP_MS", "Action", "parse_action", "split_reasoning",
    "NO_THINK_PREFILL", "THINK_PREFILL", "Decision", "GameScalingPolicy", "PolicyConfig",
    "build_request", "encode_image", "finalize",
    "USER_TURN_TEXT", "build_system_prompt", "translate_controls",
]
