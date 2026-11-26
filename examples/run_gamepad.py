"""Play GameScaling actions on a virtual PS5 controller (DualSense).

The model's keyboard/mouse chunk is mapped onto a controller with a profile (gamescaling/gamepad/mapping.py):
WASD -> left stick, mouse -> right stick, LB -> R2, RB -> L2, Space -> Cross, ...

    # dry run: map the reference Doom reply and print the 6 controller steps (no device, no server)
    python examples/run_gamepad.py --backend dummy

    # execute model replies, one per line, from stdin
    my_client | python examples/run_gamepad.py --stdin

    # live loop: grab the screen, ask the policy, drive the controller for N seconds
    bash scripts/serve_vllm.sh THUSI-Lab/GameScaling-9B-S3 8000
    python examples/run_gamepad.py --live --prompt prompts/my_game.txt --seconds 120 --profile my_profile.json

Backends: ``uinput`` (Linux, virtual DualSense, pip install evdev), ``vigem`` (Windows, virtual DualShock 4,
pip install vgamepad), ``dummy`` (prints only). The live loop grabs the screen with PIL.ImageGrab (Windows, or
Linux with X11).
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gamescaling import parse_action  # noqa: E402
from gamescaling.gamepad import ControllerProfile, GamepadExecutor, get_backend  # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto", choices=["auto", "uinput", "vigem", "dummy"])
    ap.add_argument("--profile", help="JSON controller profile (default: built-in shooter layout)")
    ap.add_argument("--reply", action="append", default=[], help="model reply text to execute (repeatable)")
    ap.add_argument("--stdin", action="store_true", help="execute one model reply per stdin line")
    ap.add_argument("--live", action="store_true", help="screen -> policy -> controller loop")
    ap.add_argument("--prompt", help="system prompt file for --live")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--seconds", type=float, default=60.0, help="length of the --live loop")
    ap.add_argument("--no-reasoning", action="store_true")
    args = ap.parse_args()

    profile = ControllerProfile.from_json(args.profile) if args.profile else ControllerProfile()
    kwargs = {"verbose": True} if args.backend == "dummy" else {}
    pad = get_backend(args.backend, **kwargs)
    ex = GamepadExecutor(pad, profile)

    try:
        if args.live:
            from PIL import ImageGrab

            from gamescaling import GameScalingPolicy, PolicyConfig
            if not args.prompt:
                ap.error("--live needs --prompt")
            system = open(args.prompt).read().rstrip("\n")
            policy = GameScalingPolicy(system, PolicyConfig(base_url=args.base_url,
                                                            reasoning=not args.no_reasoning))
            if not policy.health_check():
                print(f"server not reachable at {args.base_url}")
                return 2
            policy.reset()
            end, n, failed = time.monotonic() + args.seconds, 0, 0
            print("switch to the game window; starting in 3 s")
            time.sleep(3)
            while time.monotonic() < end:
                decision = policy.act(ImageGrab.grab())
                failed += not decision.action.parsed
                ex.execute(decision.action)
                n += 1
            print(f"{n} decisions, {failed} unparsed")
        else:
            replies = list(args.reply)
            if args.stdin:
                replies += [line for line in sys.stdin if line.strip()]
            if not replies:
                replies = [open(os.path.join(EXAMPLES, "doom_battle1_action.txt")).read()]
                print("reference reply:", repr(replies[0]))
            for text in replies:
                a = parse_action(text)
                print(f"action: parsed={a.parsed} mouse={a.mouse} steps={a.steps}")
                ex.execute(a)
    finally:
        ex.finish()
        pad.close()
    if ex.unmapped:
        print(f"tokens not bound by profile {profile.name!r} (ignored): {ex.unmapped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
