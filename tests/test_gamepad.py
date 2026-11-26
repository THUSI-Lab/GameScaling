"""Offline tests for the PS5 controller (DualSense) executor. `python -m pytest tests` or `python tests/test_gamepad.py`."""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamescaling import parse_action  # noqa: E402
from gamescaling.gamepad import (  # noqa: E402
    ControllerProfile,
    DummyGamepad,
    GamepadExecutor,
    UInputGamepad,
    ViGEmGamepad,
    translate,
)
from gamescaling.gamepad.backends import dpad_xy, to_byte  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ------------------------------------------------------------------ mapping
def test_reference_doom_reply_maps_to_fire_and_turn():
    with open(os.path.join(ROOT, "examples", "doom_battle1_action.txt")) as f:
        a = parse_action(f.read())
    steps, unmapped = translate(a, ControllerProfile())
    assert len(steps) == 6 and unmapped == []
    assert all("R2" in s.buttons and s.r2 == 1.0 for s in steps)          # LB -> R2 held all chunk
    assert approx(steps[0].left[1], 1.0) and steps[1].left == (0.0, 0.0)  # S in step 1 -> stick back
    assert approx(steps[0].right[0], -37 / 150) and steps[0].right[1] == 0.0  # Y=-3 inside deadzone


def test_diagonal_is_unit_length_and_opposites_cancel():
    a = parse_action("<|action_start|>0 0 0 ; W D ; W S ; A D ; ; ; <|action_end|>")
    steps, _ = translate(a, ControllerProfile())
    x, y = steps[0].left
    assert approx(x, 0.7071, 1e-3) and approx(y, -0.7071, 1e-3)
    assert steps[1].left == (0.0, 0.0) and steps[2].left == (0.0, 0.0)


def test_mouse_scale_clip_and_invert():
    p = ControllerProfile(mouse_full_scale=100, invert_y=True)
    steps, _ = translate(parse_action("<|action_start|>500 50 0 ; ; ; ; ; ; <|action_end|>"), p)
    assert steps[0].right == (1.0, -0.5) and steps[5].right == (1.0, -0.5)


def test_unparsed_is_neutral_and_unmapped_reported():
    steps, unmapped = translate(parse_action("garbage"), ControllerProfile())
    assert all(s.buttons == frozenset() and s.left == (0.0, 0.0) for s in steps) and unmapped == []
    steps, unmapped = translate(parse_action("<|action_start|>0 0 0 ; Space Z ; ; ; ; ; <|action_end|>"),
                                ControllerProfile())
    assert steps[0].buttons == {"Cross"} and unmapped == ["Z"]


def test_scroll_taps_button_in_first_step():
    p = ControllerProfile(scroll_up="Triangle")
    steps, _ = translate(parse_action("<|action_start|>0 0 2 ; ; ; ; ; ; <|action_end|>"), p)
    assert steps[0].buttons == {"Triangle"} and steps[1].buttons == frozenset()


def test_profile_json_merge_unbind_and_validation():
    path = os.path.join(ROOT, "prompts", "controller_profile_example.json")
    p = ControllerProfile.from_json(path)
    assert p.buttons["E"] == "Square" and "Q" not in p.buttons and p.buttons["LB"] == "R2"
    assert p.mouse_full_scale == 120 and p.scroll_up == "Triangle"
    for bad in ({"buttons": {"E": "Jump"}}, {"buttons": {"NotAKey": "Cross"}}):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bad, f)
        try:
            ControllerProfile.from_json(f.name)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        finally:
            os.unlink(f.name)


# ------------------------------------------------------------------ executor timing
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.0001)


def test_executor_runs_on_33ms_grid():
    clk = FakeClock()
    pad = DummyGamepad(clock=clk)
    ex = GamepadExecutor(pad, clock=clk, sleep=clk.sleep)
    ex.execute("<|action_start|>0 0 0 ; W ; W ; ; ; ; LB<|action_end|>")
    ex.execute(parse_action("<|action_start|>0 0 0 ; Space ; ; ; ; ; <|action_end|>"))
    ex.finish()
    tl = pad.timeline
    assert len(tl) == 13
    for i, (t, _) in enumerate(tl[:12]):
        assert abs(t - i * 0.033333) < 0.0015, (i, t)
    assert tl[5][1].buttons == {"R2"} and tl[6][1].buttons == {"Cross"}
    assert tl[-1][1].buttons == frozenset() and tl[-1][0] >= 12 * 0.0333 - 0.0015


def test_byte_and_dpad_helpers():
    assert (to_byte(-1, True), to_byte(0, True), to_byte(1, True)) == (0, 128, 255)
    assert (to_byte(0, False), to_byte(1, False)) == (0, 255)
    assert dpad_xy({"Up", "Right"}) == (1, -1) and dpad_xy({"Left", "Right"}) == (0, 0)


# ------------------------------------------------------------------ backends against stand-in OS modules
def _fake_evdev():
    names = ["EV_KEY", "EV_ABS", "BUS_USB", "ABS_X", "ABS_Y", "ABS_RX", "ABS_RY", "ABS_Z", "ABS_RZ",
             "ABS_HAT0X", "ABS_HAT0Y", "BTN_SOUTH", "BTN_EAST", "BTN_NORTH", "BTN_WEST", "BTN_TL", "BTN_TR",
             "BTN_TL2", "BTN_TR2", "BTN_SELECT", "BTN_START", "BTN_MODE", "BTN_THUMBL", "BTN_THUMBR"]

    class UInput:
        def __init__(self, caps, **kw):
            self.caps, self.kw, self.events, self.syns, self.closed = caps, kw, [], 0, False

        def write(self, t, c, v):
            self.events.append((t, c, v))

        def syn(self):
            self.syns += 1

        def close(self):
            self.closed = True

    return types.SimpleNamespace(ecodes=types.SimpleNamespace(**{n: n for n in names}),
                                 AbsInfo=lambda **kw: kw, UInput=UInput)


def test_uinput_backend():
    pad = UInputGamepad(evdev_module=_fake_evdev())
    ui = pad._ui
    assert ui.kw["vendor"] == 0x054C and ui.kw["product"] == 0x0CE6
    ui.events.clear()
    steps, _ = translate(parse_action("<|action_start|>-300 0 0 ; W LB Up ; ; ; ; ; <|action_end|>"),
                         ControllerProfile())
    pad.apply(steps[0])
    ev = {(c, v) for _, c, v in ui.events}
    assert ("ABS_Y", 0) in ev and ("ABS_RX", 0) in ev and ("ABS_RZ", 255) in ev
    assert ("BTN_TR2", 1) in ev and ("ABS_HAT0Y", -1) in ev
    pad.close()
    assert ui.closed and ("EV_KEY", "BTN_TR2", 0) in ui.events


def _fake_vgamepad():
    enum = lambda *n: types.SimpleNamespace(**{x: x for x in n})  # noqa: E731
    vg = types.SimpleNamespace(
        DS4_BUTTONS=enum("DS4_BUTTON_CROSS", "DS4_BUTTON_CIRCLE", "DS4_BUTTON_SQUARE", "DS4_BUTTON_TRIANGLE",
                         "DS4_BUTTON_SHOULDER_LEFT", "DS4_BUTTON_SHOULDER_RIGHT", "DS4_BUTTON_TRIGGER_LEFT",
                         "DS4_BUTTON_TRIGGER_RIGHT", "DS4_BUTTON_THUMB_LEFT", "DS4_BUTTON_THUMB_RIGHT",
                         "DS4_BUTTON_SHARE", "DS4_BUTTON_OPTIONS"),
        DS4_SPECIAL_BUTTONS=enum("DS4_SPECIAL_BUTTON_PS", "DS4_SPECIAL_BUTTON_TOUCHPAD"),
        DS4_DPAD_DIRECTIONS=enum("DS4_BUTTON_DPAD_NONE", "DS4_BUTTON_DPAD_NORTH", "DS4_BUTTON_DPAD_NORTHEAST",
                                 "DS4_BUTTON_DPAD_EAST", "DS4_BUTTON_DPAD_SOUTHEAST", "DS4_BUTTON_DPAD_SOUTH",
                                 "DS4_BUTTON_DPAD_SOUTHWEST", "DS4_BUTTON_DPAD_WEST", "DS4_BUTTON_DPAD_NORTHWEST"))

    class Pad:
        def __init__(self):
            self.log = []

        def __getattr__(self, name):
            return lambda **kw: self.log.append((name, kw))

    vg.VDS4Gamepad = Pad
    return vg


def test_vigem_backend():
    pad = ViGEmGamepad(vg_module=_fake_vgamepad())
    log = pad._pad.log
    log.clear()
    steps, _ = translate(parse_action("<|action_start|>0 0 0 ; D Space Tab Right ; ; ; ; ; <|action_end|>"),
                         ControllerProfile())
    pad.apply(steps[0])
    assert ("left_joystick", {"x_value": 255, "y_value": 128}) in log
    assert ("press_button", {"button": "DS4_BUTTON_CROSS"}) in log
    assert ("press_special_button", {"special_button": "DS4_SPECIAL_BUTTON_TOUCHPAD"}) in log
    assert ("directional_pad", {"direction": "DS4_BUTTON_DPAD_EAST"}) in log and log[-1] == ("update", {})
    log.clear()
    pad.neutral()
    assert ("release_button", {"button": "DS4_BUTTON_CROSS"}) in log


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
