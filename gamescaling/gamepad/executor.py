"""Execute GameScaling actions on a virtual PS5 controller: 6 steps of 33 ms per chunk (README §5)."""

from __future__ import annotations

import time
from typing import List, Optional, Union

from ..action import STEP_MS, Action, parse_action
from .backends import VirtualGamepad
from .mapping import ControllerProfile, ControllerStep, translate


class GamepadExecutor:
    """Plays each 200 ms chunk on a fixed 33 ms grid, so timing errors do not accumulate across chunks.

    If inference falls behind by more than one step, the grid re-anchors to now instead of rushing to
    catch up. Unparsed actions (``Action.parsed is False``) run as six neutral steps.
    """

    def __init__(self, pad: VirtualGamepad, profile: Optional[ControllerProfile] = None,
                 step_ms: float = STEP_MS, clock=time.perf_counter, sleep=time.sleep):
        self.pad = pad
        self.profile = profile or ControllerProfile()
        self.step_s = step_ms / 1000.0
        self._clock = clock
        self._sleep = sleep
        self._next: Optional[float] = None
        self.unmapped: List[str] = []  # tokens the model used that the profile does not bind

    def _wait_until(self, deadline: float) -> None:
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._sleep(remaining - 0.002 if remaining > 0.004 else min(remaining, 0.001))

    def execute(self, action: Union[Action, str]) -> List[ControllerStep]:
        if isinstance(action, str):
            action = parse_action(action)
        steps, unmapped = translate(action, self.profile)
        for tok in unmapped:
            if tok not in self.unmapped:
                self.unmapped.append(tok)
        now = self._clock()
        if self._next is None or now > self._next + self.step_s:
            self._next = now
        for step in steps:
            self._wait_until(self._next)
            self.pad.apply(step)
            self._next += self.step_s
        return steps

    def finish(self) -> None:
        """Let the last step run its full 33 ms, then return the controller to neutral."""
        if self._next is not None:
            self._wait_until(self._next)
        self.pad.neutral()
        self._next = None
