"""Virtual PS5 controllers the executor can drive.

| Backend | OS | Appears to games as |
|---|---|---|
| ``uinput`` | Linux | DualSense (054c:0ce6) with the kernel ``hid-playstation`` evdev layout (``pip install evdev``) |
| ``vigem``  | Windows | DualShock 4 through ViGEmBus (``pip install vgamepad``); same buttons, sticks and triggers |
| ``dummy``  | any | nothing: records every step, for dry runs and tests |

macOS has no backend: it does not allow creating virtual HID gamepads without a signed DriverKit extension.
"""

from __future__ import annotations

import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple

from .mapping import ControllerStep


def to_byte(v: float, centred: bool) -> int:
    """``-1..1`` (centred) or ``0..1`` -> 0..255 as in a DualSense/DS4 report."""
    if centred:
        return int(round((max(-1.0, min(1.0, v)) + 1.0) * 127.5))
    return int(round(max(0.0, min(1.0, v)) * 255))


def dpad_xy(buttons) -> Tuple[int, int]:
    x = (1 if "Right" in buttons else 0) - (1 if "Left" in buttons else 0)
    y = (1 if "Down" in buttons else 0) - (1 if "Up" in buttons else 0)
    return x, y


class VirtualGamepad(ABC):
    name = "base"

    @abstractmethod
    def apply(self, step: ControllerStep) -> None:
        """Set sticks, triggers and the full set of held buttons, and push them to the OS."""

    def neutral(self) -> None:
        self.apply(ControllerStep())

    def close(self) -> None:
        self.neutral()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DummyGamepad(VirtualGamepad):
    name = "dummy"

    def __init__(self, clock=time.perf_counter, verbose: bool = False):
        self._clock = clock
        self.verbose = verbose
        self.timeline: List[Tuple[float, ControllerStep]] = []

    def apply(self, step: ControllerStep) -> None:
        t = self._clock()
        self.timeline.append((t, step))
        if self.verbose:
            r = lambda v: tuple(round(a, 2) for a in v)  # noqa: E731
            print(f"{t:10.3f}  L={r(step.left)}  R={r(step.right)}  L2={step.l2:.1f} R2={step.r2:.1f}  "
                  f"buttons={sorted(step.buttons)}")


# Button name -> evdev key code, following the kernel's hid-playstation driver.
_EVDEV_BUTTONS = {
    "Cross": "BTN_SOUTH", "Circle": "BTN_EAST", "Triangle": "BTN_NORTH", "Square": "BTN_WEST",
    "L1": "BTN_TL", "R1": "BTN_TR", "L2": "BTN_TL2", "R2": "BTN_TR2",
    "Create": "BTN_SELECT", "Options": "BTN_START", "PS": "BTN_MODE",
    "L3": "BTN_THUMBL", "R3": "BTN_THUMBR",
}


class UInputGamepad(VirtualGamepad):
    """Linux virtual DualSense via /dev/uinput. Needs write access to /dev/uinput.

    The touchpad press is not emitted: the kernel exposes the DualSense touchpad as a separate device.
    """

    name = "uinput"

    def __init__(self, evdev_module=None):
        if evdev_module is None:
            try:
                import evdev as evdev_module  # type: ignore
            except ImportError as e:  # pragma: no cover - host dependent
                raise RuntimeError("uinput backend needs python-evdev: pip install evdev") from e
        ec = evdev_module.ecodes
        stick = evdev_module.AbsInfo(value=128, min=0, max=255, fuzz=0, flat=0, resolution=0)
        trig = evdev_module.AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)
        hat = evdev_module.AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)
        caps = {
            ec.EV_KEY: [getattr(ec, c) for c in _EVDEV_BUTTONS.values()],
            ec.EV_ABS: [(ec.ABS_X, stick), (ec.ABS_Y, stick), (ec.ABS_RX, stick), (ec.ABS_RY, stick),
                        (ec.ABS_Z, trig), (ec.ABS_RZ, trig), (ec.ABS_HAT0X, hat), (ec.ABS_HAT0Y, hat)],
        }
        self._ec = ec
        self._ui = evdev_module.UInput(caps, name="Sony Interactive Entertainment DualSense Wireless Controller",
                                       vendor=0x054C, product=0x0CE6, version=0x8111, bustype=ec.BUS_USB)
        self._last: dict = {}
        self._warned = False
        self.neutral()

    def _write(self, etype: str, code: str, value: int) -> None:
        c = getattr(self._ec, code)
        if self._last.get(c) != value:
            self._last[c] = value
            self._ui.write(getattr(self._ec, etype), c, value)

    def apply(self, step: ControllerStep) -> None:
        if "Touchpad" in step.buttons and not self._warned:
            warnings.warn("uinput backend: Touchpad press is not supported on Linux and is ignored")
            self._warned = True
        for code, v in (("ABS_X", step.left[0]), ("ABS_Y", step.left[1]),
                        ("ABS_RX", step.right[0]), ("ABS_RY", step.right[1])):
            self._write("EV_ABS", code, to_byte(v, True))
        self._write("EV_ABS", "ABS_Z", to_byte(step.l2, False))
        self._write("EV_ABS", "ABS_RZ", to_byte(step.r2, False))
        for name, code in _EVDEV_BUTTONS.items():
            self._write("EV_KEY", code, 1 if name in step.buttons else 0)
        hx, hy = dpad_xy(step.buttons)
        self._write("EV_ABS", "ABS_HAT0X", hx)
        self._write("EV_ABS", "ABS_HAT0Y", hy)
        self._ui.syn()

    def close(self) -> None:
        try:
            self.neutral()
        finally:
            self._ui.close()


_DS4_BUTTONS = {
    "Cross": "DS4_BUTTON_CROSS", "Circle": "DS4_BUTTON_CIRCLE", "Square": "DS4_BUTTON_SQUARE",
    "Triangle": "DS4_BUTTON_TRIANGLE", "L1": "DS4_BUTTON_SHOULDER_LEFT", "R1": "DS4_BUTTON_SHOULDER_RIGHT",
    "L2": "DS4_BUTTON_TRIGGER_LEFT", "R2": "DS4_BUTTON_TRIGGER_RIGHT", "L3": "DS4_BUTTON_THUMB_LEFT",
    "R3": "DS4_BUTTON_THUMB_RIGHT", "Create": "DS4_BUTTON_SHARE", "Options": "DS4_BUTTON_OPTIONS",
}
_DS4_SPECIAL = {"PS": "DS4_SPECIAL_BUTTON_PS", "Touchpad": "DS4_SPECIAL_BUTTON_TOUCHPAD"}
_DS4_DPAD = {
    (0, 0): "DS4_BUTTON_DPAD_NONE", (0, -1): "DS4_BUTTON_DPAD_NORTH", (1, -1): "DS4_BUTTON_DPAD_NORTHEAST",
    (1, 0): "DS4_BUTTON_DPAD_EAST", (1, 1): "DS4_BUTTON_DPAD_SOUTHEAST", (0, 1): "DS4_BUTTON_DPAD_SOUTH",
    (-1, 1): "DS4_BUTTON_DPAD_SOUTHWEST", (-1, 0): "DS4_BUTTON_DPAD_WEST", (-1, -1): "DS4_BUTTON_DPAD_NORTHWEST",
}


class ViGEmGamepad(VirtualGamepad):
    """Windows virtual DualShock 4 via ViGEmBus. DualSense-only features (adaptive triggers, haptics)
    are unavailable; buttons, sticks and triggers match."""

    name = "vigem"

    def __init__(self, vg_module=None):
        if vg_module is None:
            try:
                import vgamepad as vg_module  # type: ignore
            except ImportError as e:  # pragma: no cover - host dependent
                raise RuntimeError("vigem backend needs vgamepad and ViGEmBus: pip install vgamepad") from e
        self._vg = vg_module
        self._pad = vg_module.VDS4Gamepad()
        self._held: set = set()
        self.neutral()

    def apply(self, step: ControllerStep) -> None:
        vg, pad = self._vg, self._pad
        # raw DS4 report values: 0 = left/up, 255 = right/down (matches +y down)
        pad.left_joystick(x_value=to_byte(step.left[0], True), y_value=to_byte(step.left[1], True))
        pad.right_joystick(x_value=to_byte(step.right[0], True), y_value=to_byte(step.right[1], True))
        pad.left_trigger(value=to_byte(step.l2, False))
        pad.right_trigger(value=to_byte(step.r2, False))
        for name, attr in _DS4_BUTTONS.items():
            want, have = name in step.buttons, name in self._held
            if want and not have:
                pad.press_button(button=getattr(vg.DS4_BUTTONS, attr))
            elif have and not want:
                pad.release_button(button=getattr(vg.DS4_BUTTONS, attr))
        for name, attr in _DS4_SPECIAL.items():
            want, have = name in step.buttons, name in self._held
            if want and not have:
                pad.press_special_button(special_button=getattr(vg.DS4_SPECIAL_BUTTONS, attr))
            elif have and not want:
                pad.release_special_button(special_button=getattr(vg.DS4_SPECIAL_BUTTONS, attr))
        pad.directional_pad(direction=getattr(vg.DS4_DPAD_DIRECTIONS, _DS4_DPAD[dpad_xy(step.buttons)]))
        pad.update()
        self._held = set(step.buttons)


def get_backend(name: str = "auto", **kwargs) -> VirtualGamepad:
    """``auto`` = uinput on Linux, vigem on Windows. ``dummy`` never touches the OS."""
    if name == "auto":
        if sys.platform.startswith("linux"):
            name = "uinput"
        elif sys.platform == "win32":
            name = "vigem"
        else:
            raise RuntimeError("no virtual-controller backend on this OS (macOS cannot create virtual "
                               "gamepads without a signed driver); use Linux or Windows, or backend='dummy'")
    if name == "dummy":
        return DummyGamepad(**kwargs)
    if name == "uinput":
        return UInputGamepad(**kwargs)
    if name == "vigem":
        return ViGEmGamepad(**kwargs)
    raise ValueError(f"unknown backend {name!r} (auto, uinput, vigem, dummy)")
