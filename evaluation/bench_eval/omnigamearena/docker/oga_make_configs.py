#!/usr/bin/env python3
"""Derive one single-game run config per game from a suite yaml.

The parallel runner drives every game in its own driver process, so it needs one
config per game rather than the single filtered config the sequential/sharded path
uses. Every knob is applied exactly as eval_ckpt.sh applies it, so results
stay comparable with runs produced by the sequential path.

PvP/Coop games take the other shape run_benchmark dispatches on: ``game:`` plus a
``players:`` list instead of ``games:`` plus ``agents:``. That branch lives here
rather than in a second script so a knob can never mean one thing for solo and
another for two-player.

usage: oga_make_configs.py SRC_SUITE OUT_DIR MODEL_ID GAME [GAME ...]

env (all optional, same names/semantics as eval_ckpt.sh):
  MAP_WAIT TEMP REASONING REASONING_MAX_TOKENS REASONING_STYLE
  HISTORY_EVICT_CHUNK INVERT_Y STOP HISTORY HISTORY_REASONING RESIZE
  BASE_URL  vLLM endpoint for these configs. Must be set explicitly per replica:
            factory.py passes the config's base_url straight to the agent, so it
            overrides YOURVLA_BASE_URL and every driver would hit replica 0.

Prints one "<game>\t<config path>" line per game.
"""
import os
import sys
from pathlib import Path

import yaml


def _game_mode(game: str) -> str:
    """solo / pvp / coop, straight from the registry the runner dispatches on.

    Never guess: a wrong mode emits a config in the wrong shape, and a
    two-player game written as a solo suite entry only surfaces much later as
    run_benchmark's "suite mode supports solo games only".
    """
    try:
        from omni_game_arena.benchmark.games import get_game
    except ImportError as exc:
        sys.exit(f"[cfg][FATAL] cannot import the game registry ({exc}); "
                 f"set PYTHONPATH to the omnigamearena package root")
    return getattr(get_game(game), "mode", "solo")


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    src, out_dir, model_id, games = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]

    data = yaml.safe_load(Path(src).read_text())
    have = data.get("games") or []
    missing = [g for g in games if g not in have]
    if missing:
        sys.exit(f"[cfg][FATAL] games not in suite: {missing} (suite has {have})")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for game in games:
        cfg = yaml.safe_load(Path(src).read_text())
        mode = _game_mode(game)
        two_player = mode in {"pvp", "coop"}
        if two_player:
            # env_defaults/players/games -> the single-match shape: one game, an
            # `env:` block, and a players list. Ports stay unset so the launcher's
            # --port gives player 1 the worker's UE5 port and player 2 base+1.
            cfg.pop("games", None)
            cfg.pop("agents", None)
            cfg["game"] = game
            cfg["env"] = {**(cfg.pop("env_defaults", None) or {}),
                          "host": "127.0.0.1", "task": "", "map": game}
        else:
            cfg["games"] = [game]
            cfg.setdefault("env_defaults", {})
            # Same as the two-player branch: pin env.map to the registry key so
            # a missing GameSpec.map cannot silently play UE5's boot level
            # (ObstacleRun3D). Per-game game_env.map still wins below.
            cfg["env_defaults"].setdefault("map", game)
            # Official cold-start cap from GameSpec (scene_escape / cue_chase=500).
            from omni_game_arena.benchmark.games import get_game
            spec_steps = getattr(get_game(game), "max_steps", None)
            if spec_steps:
                cfg["env_defaults"]["max_steps"] = spec_steps
            # Per-game env (e.g. scene_escape/cue_chase max_steps=500). Applied
            # here so the parallel driver, which never goes through
            # run_benchmark's suite expander, still sees it.
            override = (cfg.pop("game_env", None) or {}).get(game) or {}
            cfg["env_defaults"].update(override)

        map_wait = float(os.environ.get("MAP_WAIT") or 60)
        env_block = cfg["env"] if two_player else cfg["env_defaults"]
        env_block["map_wait"] = map_wait
        if two_player:
            override = (cfg.pop("game_env", None) or {}).get(game) or {}
            env_block.update(override)

        if two_player:
            players = cfg.get("players") or []
            if len(players) != 2:
                sys.exit(f"[cfg][FATAL] {game} is {mode} but the suite lists "
                         f"{len(players)} player(s); need exactly 2")
            for p in players:
                if isinstance(p, dict):
                    p["model"] = model_id
        else:
            agents = cfg.get("agents") or []
            for a in agents:
                if isinstance(a, dict) and "model" in a:
                    a["model"] = model_id
            if not agents:
                cfg["agents"] = [{"model": model_id}]

        temp = (os.environ.get("TEMP") or "").strip()
        if temp:
            cfg.setdefault("params", {})["temperature"] = float(temp)

        defaults_key = "players_defaults" if two_player else "agents_defaults"
        extra = cfg.setdefault(defaults_key, {}).setdefault("extra", {})
        base_url = (os.environ.get("BASE_URL") or "").strip()
        if base_url:
            extra["base_url"] = base_url
        # YourVLA trains Y>0 = look down, UE RemoteInput dy>0 = look up. Default ON.
        extra["invert_y"] = os.environ.get("INVERT_Y", "1") != "0"
        stop = (os.environ.get("STOP") or "").strip()
        if stop:
            extra["stop"] = stop
        if os.environ.get("REASONING") == "1":
            extra["reasoning"] = True
            extra["reasoning_max_tokens"] = int(os.environ.get("REASONING_MAX_TOKENS") or 2048)
            extra["reasoning_style"] = os.environ.get("REASONING_STYLE") or "two_stage"
        evict = int(os.environ.get("HISTORY_EVICT_CHUNK") or 1)
        if evict > 1:
            extra["history_evict_chunk"] = evict
        # HISTORY is also what sets vLLM's --limit-mm-per-prompt, so the two can
        # never disagree about how many frames a prompt may carry.
        history = (os.environ.get("HISTORY") or "").strip()
        if history:
            cfg.setdefault("params", {})["history_len"] = int(history)
        # Reasoning kept in the history turns: 0 action-only (standard), -1 all
        # turns, N the N most recent. Needs the patched chat template, which the
        # agent sends per request -- see yourvla.HISTORY_THINK_TEMPLATE.
        hist_reason = (os.environ.get("HISTORY_REASONING") or "").strip()
        if hist_reason:
            extra["history_reasoning_len"] = int(hist_reason)
        # Observation resize only (WxH). Mouse denorm stays extra.screen_w/h.
        resize = (os.environ.get("RESIZE") or "").strip()
        if resize:
            w, h = resize.lower().replace(",", "x").split("x")
            extra["resize_to"] = [int(w), int(h)]

        dst = Path(out_dir) / f"{Path(src).stem}_{model_id}_{game}.yaml"
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        yaml.safe_load(dst.read_text())          # validate round-trip
        print(f"{game}\t{dst}")


if __name__ == "__main__":
    main()
