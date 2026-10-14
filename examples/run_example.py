"""One Doom frame in, one action out.

`examples/doom_battle1.jpg` is a frame from a Doom Battle-1 episode played by
GameScaling-9B-S3, and `examples/doom_battle1_action.txt` is the reply the model
gave on that frame (a monster just left of the crosshair: turn left and fire).

    python examples/run_example.py --offline            # parse the reference reply only
    bash scripts/serve_vllm.sh THUSI-Lab/GameScaling-9B-S3 8000
    python examples/run_example.py                      # prefill <think>
    python examples/run_example.py --no-reasoning       # prefill <think>\\n\\n</think>

Sampling is at temperature 1.0 and the reference came with five earlier frames
in context, so a live reply will differ in detail. The check is that every
reply parses into a valid action chunk; the script also reports whether it
fires (LB), as the reference does.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from gamescaling import GameScalingPolicy, PolicyConfig, parse_action  # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples")


def describe(action) -> str:
    fires = any("LB" in g for g in action.steps)
    return f"parsed={action.parsed}  mouse={action.mouse}  steps={action.steps}  fires={fires}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--image", default=os.path.join(EXAMPLES, "doom_battle1.jpg"))
    ap.add_argument("--prompt", default=os.path.join(ROOT, "prompts", "example_doom_battle1.txt"))
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--offline", action="store_true", help="only parse the reference reply")
    args = ap.parse_args()

    reference = open(os.path.join(EXAMPLES, "doom_battle1_action.txt")).read()
    print("reference reply:", repr(reference))
    print("reference action:", describe(parse_action(reference)))
    if args.offline:
        return 0

    system = open(args.prompt).read().rstrip("\n")
    policy = GameScalingPolicy(system, PolicyConfig(base_url=args.base_url,
                                                    reasoning=not args.no_reasoning))
    if not policy.health_check():
        print(f"server not reachable at {args.base_url}")
        return 2

    frame = Image.open(args.image)
    ok = 0
    for i in range(args.trials):
        policy.reset()
        d = policy.act(frame)
        print(f"--- trial {i + 1}  ({d.latency_ms:.0f} ms, {d.completion_tokens} tokens)")
        print(repr(d.text))
        print(describe(d.action))
        ok += d.action.parsed and not d.action.truncated
    print(f"{ok}/{args.trials} replies parsed")
    return 0 if ok == args.trials else 1


if __name__ == "__main__":
    sys.exit(main())
