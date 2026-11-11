#!/usr/bin/env python3
"""Patch MineStudio 1.1.6 for NumPy >= 2.

In `minestudio/simulator/minerl/herobraine/hero/spaces.py`, `Text.sample` calls
`np.array(np.reshape(strings, self.shape), np.dtype)`, which fails under
NumPy 2.x partway through a rollout. This replaces the second argument with
`dtype=self.dtype`. Idempotent.
"""
import inspect
import pathlib
import sys

import minestudio.simulator.minerl.herobraine.hero.spaces as spaces

BROKEN = "np.array(np.reshape(strings, self.shape), np.dtype)"
FIXED = "np.array(np.reshape(strings, self.shape), dtype=self.dtype)"

path = pathlib.Path(inspect.getsourcefile(spaces))
text = path.read_text()
if FIXED in text and BROKEN not in text:
    print(f"already patched: {path}")
elif BROKEN in text:
    path.write_text(text.replace(BROKEN, FIXED))
    print(f"patched: {path}")
else:
    sys.exit(f"unexpected MineStudio source, patch target not found: {path}")
