"""ViZDoom FPS benchmark harness (Game-TARS Figure 8, arXiv 2510.23691).

Four maps -- Battle-1, Battle-2, Defend the Center, Defend the Line -- played
through the same unified keyboard/mouse action space as OmniGameArena, so a CPT
checkpoint, a Qwen3.5 base model and a Qwen3-VL-Thinking base model are all
measured under one protocol.
"""

from .games import GAMES, SUITE, GameSpec, get_game  # noqa: F401

__all__ = ["GAMES", "SUITE", "GameSpec", "get_game"]
