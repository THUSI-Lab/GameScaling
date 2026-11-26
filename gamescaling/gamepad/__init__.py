"""Play GameScaling policies with a PS5 controller (DualSense).

The checkpoints emit keyboard/mouse chunks; this package maps each chunk onto a virtual controller:

    from gamescaling.gamepad import ControllerProfile, GamepadExecutor, get_backend

    with get_backend("auto") as pad:                 # uinput on Linux, vigem on Windows
        ex = GamepadExecutor(pad, ControllerProfile())
        decision = policy.act(frame)
        ex.execute(decision.action)                  # 6 x 33 ms on the controller
"""

from .backends import DummyGamepad, UInputGamepad, ViGEmGamepad, VirtualGamepad, get_backend
from .executor import GamepadExecutor
from .mapping import BUTTONS, DEFAULT_BUTTONS, DEFAULT_MOVE_KEYS, ControllerProfile, ControllerStep, translate

__all__ = [
    "BUTTONS", "DEFAULT_BUTTONS", "DEFAULT_MOVE_KEYS", "ControllerProfile", "ControllerStep", "translate",
    "GamepadExecutor", "DummyGamepad", "UInputGamepad", "ViGEmGamepad", "VirtualGamepad", "get_backend",
]
