#!/usr/bin/env python3
"""Run ViZDoom benchmark units against one model endpoint.

A "unit" is one (map, episode) pair. The suite is sharded at that granularity
rather than per map, because a battle episode ends whenever the agent dies and a
defend_the_* episode almost always runs to the 2100-tic timeout -- sharding per
map would leave one worker with 36 minutes of work and another with 6.

Examples
--------
  # local smoke, no server: random policy, 20 steps per episode
  python3 scripts/run_benchmark.py --agent random --served-name smoke \
      --units battle1:0 defend_the_center:0 --max-steps 20

  # one worker's share of a real run
  python3 scripts/run_benchmark.py --served-name my_ckpt_doom \
      --base-url http://127.0.0.1:30000/v1 \
      --units battle1:0 battle1:1 battle1:2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vizdoom_arena.games import SUITE, get_game  # noqa: E402
from vizdoom_arena.objectives import DEFAULT_OBJECTIVES, load_objectives  # noqa: E402
from vizdoom_arena.prompts import OBJECTIVE_POSITIONS  # noqa: E402
from vizdoom_arena.runner import run_episode  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "doom_suite.yaml")


def load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print(f"[warn] pyyaml missing; ignoring {path} and using built-in defaults")
        return {}
    if not os.path.isfile(path):
        print(f"[warn] config {path} not found; using built-in defaults")
        return {}
    return yaml.safe_load(open(path)) or {}


def parse_units(tokens: list[str], maps: list[str], episodes: int) -> list[tuple[str, int]]:
    if tokens:
        units = []
        for tok in tokens:
            for piece in tok.replace(",", " ").split():
                name, _, idx = piece.partition(":")
                get_game(name)                      # fail fast on a typo
                units.append((name, int(idx or 0)))
        return units
    return [(m, e) for m in maps for e in range(episodes)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--served-name", required=True,
                    help="vLLM --served-model-name; also the aggregation key")
    ap.add_argument("--model-id", default=None,
                    help="id put in the request body. Defaults to --served-name. "
                         "Set it when the endpoint serves a fixed catalogue and "
                         "--served-name is only a batch label")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-flavor", default="vllm",
                    choices=["vllm", "openai"],
                    help="openai drops the vLLM-only request keys and the "
                         "assistant continuation, for a hosted gateway")
    ap.add_argument("--agent", default="policy", choices=["policy", "random"])
    ap.add_argument("--maps", default=None, help="space-separated subset of the suite")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--units", nargs="*", default=None,
                    help="explicit map:episode units, e.g. battle1:0 battle1:1")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--prompt-style", default=None,
                    choices=["oga", "guided", "native"])
    ap.add_argument("--mouse-mode", default=None,
                    choices=["projective", "linear"])
    ap.add_argument("--history", type=int, default=None)
    ap.add_argument("--history-reasoning", type=int, default=None)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--reasoning", type=int, default=None)
    ap.add_argument("--reasoning-style", default=None, choices=["free", "two_stage", "closed"])
    ap.add_argument("--reasoning-max-tokens", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--clock", default=None, choices=["pdq", "lcrt", "rt", "trt"],
                    help="pdq freezes the world while thinking; lcrt charges the "
                         "latency as forced no-ops; rt is the pipelined model "
                         "real agents use (action persistence + staleness)")
    ap.add_argument("--resize", default=None, help="WxH the policy sees, or 'none'")
    ap.add_argument("--turn-deg", type=float, default=None,
                    help="linear gain: degrees for |X|=1000. Ignored by projective.")
    ap.add_argument("--look-deg", type=float, default=None,
                    help="linear gain for |Y|=1000. Defaults to --turn-deg * 720/1280, "
                         "which is what a shared mouse sensitivity implies.")
    ap.add_argument("--stop", default=None)
    ap.add_argument("--record-video", dest="record_video", action="store_true", default=None)
    ap.add_argument("--no-record-video", dest="record_video", action="store_false")
    ap.add_argument("--keep-going", action="store_true",
                    help="a failed unit does not abort the remaining ones")
    ap.add_argument("--objective-set", default=None,
                    help="set name in --objectives; adds 'Current objective: ...' "
                         "to each map's system prompt (oga style only)")
    ap.add_argument("--objectives", default=DEFAULT_OBJECTIVES)
    ap.add_argument("--objective-pos", default="rules", choices=OBJECTIVE_POSITIONS,
                    help="rules = last line of Game Rules; end = last line of the "
                         "whole system prompt")
    ap.add_argument("--recap-last", action="store_true",
                    help="start each user turn with the previous turn's reasoning + "
                         "action as text")
    ap.add_argument("--carry-think", action="store_true",
                    help="GAME-JEV mc_s1: the newest non-empty reasoning opens the "
                         "first user turn of the window")
    ap.add_argument("--image-only-user", action="store_true",
                    help="user turns are the frame alone, without 'current game screen'")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    env_cfg = cfg.get("env") or {}
    agent_cfg = cfg.get("agent") or {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    maps = (args.maps.split() if args.maps else cfg.get("maps") or list(SUITE))
    episodes = args.episodes or int(cfg.get("episodes_per_map", 3))
    units = parse_units(args.units or [], maps, episodes)

    output_root = args.output_root or cfg.get("output_root", "runs/doom")
    if not os.path.isabs(output_root):
        output_root = os.path.join(root, output_root)

    resize: tuple[int, int] | None
    if args.resize is None:
        resize = tuple(agent_cfg.get("resize") or (1280, 720))
    elif args.resize.lower() in ("none", "off", "0"):
        resize = None
    else:
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    kwargs = dict(
        served_name=args.served_name,
        model_id=args.model_id,
        output_root=output_root,
        base_url=args.base_url or agent_cfg.get("base_url"),
        api_flavor=args.api_flavor,
        agent_kind=args.agent,
        prompt_style=args.prompt_style or agent_cfg.get("prompt_style", "oga"),
        history_len=args.history if args.history is not None else int(agent_cfg.get("history_len", 20)),
        history_reasoning_len=(args.history_reasoning if args.history_reasoning is not None
                               else int(agent_cfg.get("history_reasoning_len", 0))),
        temperature=args.temp if args.temp is not None else float(agent_cfg.get("temperature", 1.0)),
        max_tokens=args.max_tokens or int(agent_cfg.get("max_tokens", 96)),
        reasoning=(bool(args.reasoning) if args.reasoning is not None
                   else bool(agent_cfg.get("reasoning", True))),
        reasoning_style=args.reasoning_style or agent_cfg.get("reasoning_style", "free"),
        reasoning_max_tokens=(args.reasoning_max_tokens
                              or int(agent_cfg.get("reasoning_max_tokens", 2048))),
        resize=resize,
        resolution=tuple(env_cfg.get("resolution") or (1280, 720)),
        render_hud=bool(env_cfg.get("render_hud", True)),
        max_steps=args.max_steps or env_cfg.get("max_steps"),
        clock_mode=args.clock or cfg.get("clock_mode", "pdq"),
        mouse_mode=(args.mouse_mode
                    or str(env_cfg.get("mouse_mode", "projective"))),
        turn_deg_per_full_x=(args.turn_deg if args.turn_deg is not None
                             else float(env_cfg.get("turn_deg_per_full_x", 90.0))),
        # Both axes share one mouse sensitivity, so the vertical gain follows the
        # frame's aspect ratio rather than being a second free parameter: Y is
        # normalised by the capture HEIGHT, so the same raw delta maps to
        # height/width as many degrees.
        look_deg_per_full_y=(
            args.look_deg if args.look_deg is not None
            else args.turn_deg * 720.0 / 1280.0 if args.turn_deg is not None
            else float(env_cfg.get("look_deg_per_full_y", 45.0))),
        stop=args.stop,
        record_video=(args.record_video if args.record_video is not None
                      else bool(cfg.get("record_video", False))),
        video_fps=int(cfg.get("video_fps", 30)),
    )
    if args.recap_last:
        kwargs["recap_last"] = True
    if args.carry_think:
        kwargs["carry_think"] = True
    if args.image_only_user:
        kwargs["image_only_user"] = True

    print(f"[vzd] served_name={args.served_name} "
          f"model_id={args.model_id or args.served_name} agent={args.agent} "
          f"units={' '.join(f'{g}:{e}' for g, e in units)}", flush=True)
    print(f"[vzd] output_root={output_root}", flush=True)
    objectives = (load_objectives(args.objectives, args.objective_set,
                                  sorted({g for g, _ in units}))
                  if args.objective_set else {})

    print(f"[vzd] knobs: prompt_style={kwargs['prompt_style']} history={kwargs['history_len']} "
          f"temp={kwargs['temperature']} reasoning={kwargs['reasoning']}/"
          f"{kwargs['reasoning_style']} clock={kwargs['clock_mode']} "
          f"max_steps={kwargs['max_steps']} resize={kwargs['resize']} "
          f"objective_set={args.objective_set or '-'}"
          f"{'@' + args.objective_pos if args.objective_set else ''} "
          f"recap_last={int(args.recap_last)} carry_think={int(args.carry_think)} "
          f"image_only_user={int(args.image_only_user)}", flush=True)

    failures = 0
    for game, ep in units:
        texts = objectives.get(game)
        objective = texts[ep % len(texts)] if texts else ""
        try:
            run_episode(game, ep, objective=objective,
                        objective_set=args.objective_set or "",
                        objective_pos=args.objective_pos, **kwargs)
        except Exception:                       # noqa: BLE001
            failures += 1
            print(f"[vzd][FAIL] {game}:{ep}", flush=True)
            traceback.print_exc()
            if not args.keep_going:
                return 1
    if failures:
        print(f"[vzd] {failures}/{len(units)} units failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
