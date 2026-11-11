"""Episode loop: one (map, episode) unit in, one run directory out.

Each unit writes its own directory so the aggregator never has to guess which
episode a number came from, and so a worker that dies mid-suite leaves every
completed episode intact:

    runs/doom/<map>/<timestamp>__<served_name>__ep<k>/
        result.json     scores + metrics + the parameters that produced them
        steps.jsonl     one line per decision (model text, parsed action, reward)
        episode.mp4     what the policy saw, when recording is on

`result.json` keeps OmniGameArena's shape where it matters -- `agent.model` is
the aggregation key and the score lives at `metrics.game.score` -- so reading a
ViZDoom summary next to an OGA one needs no translation.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from collections import Counter
from typing import Any

from .action import CHUNK_STEPS, ActionChunk, parse_chunk, split_reasoning
from .agent import DoomPolicyAgent, RandomAgent
from .env import DEFAULT_RESOLUTION, VizDoomEnv
from .games import TICS_PER_SECOND, get_game
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)

# Fixed base seed so episode k of a map is the same fight for every model.
# Two checkpoints compared on different layouts would differ by map luck, and
# battle1/battle2 spawn variance is large enough to swamp a real effect.
BASE_SEED = 20260831


def episode_seed(game: str, episode_index: int) -> int:
    return BASE_SEED + abs(hash(game)) % 1000 + episode_index * 7919


class _VideoWriter:
    """MP4 writer that refuses to pretend.

    OmniGameArena lost the MP4s of several batches because the recorder only
    logged the missing ffmpeg backend at debug level: every frame failed and the
    runs still looked clean. Fail at construction instead.
    """

    def __init__(self, path: str, fps: int = 30):
        try:
            import imageio.v2 as imageio  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "record_video was requested but imageio is not installed; "
                "`pip install imageio imageio_ffmpeg` or pass --no-record-video"
            ) from exc
        self._writer = imageio.get_writer(path, fps=fps, macro_block_size=None,
                                          codec="libx264", quality=6)

    def append(self, frame) -> None:
        import numpy as np  # noqa: PLC0415
        self._writer.append_data(np.asarray(frame.convert("RGB")))

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception:                      # noqa: BLE001
            pass


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def run_episode(
    game: str,
    episode_index: int,
    *,
    served_name: str,
    output_root: str,
    # The id sent in the request body. Defaults to served_name, which is right
    # for our own vLLM. A hosted gateway serves a fixed catalogue, so the wire
    # id has to stay the catalogue name while served_name carries the batch
    # label that keeps runs apart in aggregation.
    model_id: str | None = None,
    base_url: str | None = None,
    agent_kind: str = "policy",
    prompt_style: str = "oga",
    history_len: int = 20,
    history_reasoning_len: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 96,
    reasoning: bool = True,
    reasoning_style: str = "free",
    reasoning_max_tokens: int = 2048,
    resize: tuple[int, int] | None = DEFAULT_RESOLUTION,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    render_hud: bool = True,
    max_steps: int | None = None,
    clock_mode: str = "pdq",
    mouse_mode: str = "projective",
    turn_deg_per_full_x: float = 90.0,
    look_deg_per_full_y: float = 45.0,
    stop: str | None = None,
    api_flavor: str = "vllm",
    record_video: bool = False,
    video_fps: int = 30,
    seed: int | None = None,
    timeout: float = 120.0,
    objective: str = "",
    objective_set: str = "",
    objective_pos: str = "rules",
    recap_last: bool = False,
    carry_think: bool = False,
    image_only_user: bool = False,
) -> dict[str, Any]:
    spec = get_game(game)
    steps_budget = int(max_steps or spec.max_steps)
    seed = episode_seed(game, episode_index) if seed is None else int(seed)
    system_prompt = build_system_prompt(spec, prompt_style, reasoning=reasoning,
                                        no_think=reasoning_style == "closed",
                                        objective=objective, objective_pos=objective_pos)

    run_dir = os.path.join(
        output_root, game,
        f"{time.strftime('%Y%m%d_%H%M%S')}__{served_name}__ep{episode_index}",
    )
    os.makedirs(run_dir, exist_ok=True)

    if agent_kind == "random":
        agent: Any = RandomAgent(seed=seed)
    else:
        agent = DoomPolicyAgent(
            model=model_id or served_name,
            system_prompt=system_prompt, base_url=base_url,
            history_len=history_len, history_reasoning_len=history_reasoning_len,
            temperature=temperature, max_tokens=max_tokens, resize_to=resize,
            reasoning=reasoning, reasoning_style=reasoning_style,
            reasoning_max_tokens=reasoning_max_tokens, stop=stop, timeout=timeout,
            api_flavor=api_flavor, recap_last=recap_last, carry_think=carry_think,
            image_only_user=image_only_user,
        )
        if not agent.health_check():
            raise RuntimeError(f"no model server at {agent.base_url}")
        warmup_ms = agent.warmup()
        if warmup_ms > 5000:
            logger.info("warmup absorbed %.0f s of first-call cost", warmup_ms / 1000)

    env = VizDoomEnv(
        spec, resolution=resolution, render_hud=render_hud,
        mouse_mode=mouse_mode,
        turn_deg_per_full_x=turn_deg_per_full_x,
        look_deg_per_full_y=look_deg_per_full_y, clock_mode=clock_mode,
    )
    video = _VideoWriter(os.path.join(run_dir, "episode.mp4"), video_fps) if record_video else None

    latencies: list[float] = []
    charged: list[float] = []     # trt: latency charged to the game (think decode excluded)
    think_chars: list[int] = []
    completion_tokens: list[int] = []
    abs_x: list[float] = []
    key_uses: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    unclosed_why: Counter[str] = Counter()
    n_noop = n_attack = n_think = n_mouse = n_truncated = 0
    total_reward = 0.0
    termination = "max_steps"
    t_start = time.time()

    # `rt` bookkeeping. One loop iteration is one decision cycle lasting
    # max(CHUNK_STEPS, latency) tics: the chunk decided one cycle ago runs its 6
    # tics, then its key state is held for whatever is left while the next
    # decision is still in flight. So a cycle never rounds up to a chunk
    # boundary, and decision_rate = CHUNK_STEPS / max(CHUNK_STEPS, latency) comes
    # out continuous instead of collapsing to 1/ceil(latency / chunk).
    cur: ActionChunk = ActionChunk()      # noop only for the very first cycle
    n_chunks = 0
    noop_startup_tics = 0
    staleness_tics: list[int] = []

    trace = open(os.path.join(run_dir, "steps.jsonl"), "w", encoding="utf-8")
    try:
        obs = env.reset(seed=seed)
        step = 0
        while n_chunks < steps_budget:
            if env.done:
                termination = "episode_finished"
                break
            if video is not None:
                video.append(obs)

            # The frame the model is about to look at is the world as it stands
            # now; the chunk it returns will only start executing one cycle
            # later, which is where rt's staleness comes from.
            text = agent.act(obs)
            chunk = parse_chunk(text)
            latency_ms = agent.last_predict_ms or 0.0
            # trt: the <think> decode time is not charged (world frozen while thinking).
            think_ms = float(getattr(agent, "last_think_ms", None) or 0.0) if clock_mode == "trt" else 0.0
            charged_ms = max(0.0, latency_ms - think_ms)
            charged.append(charged_ms)
            latencies.append(latency_ms)
            if chunk.is_noop:
                n_noop += 1
            abs_x.append(abs(chunk.mouse[0]))
            if chunk.mouse[0] or chunk.mouse[1]:
                n_mouse += 1
            key_uses.update(chunk.pressed_keys)
            dropped.update(chunk.dropped_tokens)
            think, _, think_closed = split_reasoning(text)
            if think:
                n_think += 1
                think_chars.append(len(think))
            if chunk.truncated:
                n_truncated += 1
                # "length" = the reasoning budget ran out. Anything else (in
                # practice "stop") = generation was cut while still thinking,
                # which means a stop sequence matched inside the trace.
                unclosed_why[agent.last_finish_reason or "unknown"] += 1
            if agent.last_completion_tokens:
                completion_tokens.append(agent.last_completion_tokens)
            step += 1

            hold_tics = 0
            if clock_mode in ("rt", "trt"):
                latency_tics = int(round(charged_ms / 1000.0 * TICS_PER_SECOND))
                # A decision arriving before the running chunk is done does NOT
                # preempt it: the 6 per-tic mouse deltas are one intended
                # rotation, so cutting the chunk short would execute only part of
                # the turn and a faster model would systematically under-aim.
                hold_tics = max(0, latency_tics - CHUNK_STEPS)
                staleness_tics.append(CHUNK_STEPS + hold_tics)
                if n_chunks == 0:
                    # The world runs for one cycle before the first decision can
                    # possibly land, with no input at all. Counted, not hidden.
                    noop_startup_tics = CHUNK_STEPS + hold_tics

            # `cur` is the chunk decided one cycle ago; lcrt instead charges the
            # latency as no-ops after the chunk, and pdq never pays anything.
            applied = cur if clock_mode in ("rt", "trt") else chunk
            info = env.step(
                applied,
                latency_s=(latency_ms / 1000.0 if clock_mode == "lcrt" else 0.0),
                hold_tics=hold_tics)
            cur = chunk
            total_reward = info["total_reward"]
            if info["attack_tics"]:
                n_attack += 1
            obs = env.observe()
            n_chunks += 1

            row = {
                "step": step,
                "latency_ms": round(latency_ms, 1),
                "parsed": applied.parsed,
                "mouse": applied.mouse,
                "keys": applied.steps,
                "reward": info["reward"],
                "total_reward": info["total_reward"],
                "attack_tics": info["attack_tics"],
                "displacement": round(info["displacement"], 2),
                "game_variables": {k: round(v, 2) for k, v in info["game_variables"].items()},
                "think_chars": len(think),
                "think_closed": think_closed,
                "finish_reason": agent.last_finish_reason,
                "completion_tokens": agent.last_completion_tokens,
                "action_end_restored": getattr(agent, "last_action_end_restored", False),
                "stop_reason": getattr(agent, "last_stop_reason", None),
                "vlm_response": text,
            }
            if clock_mode == "trt":
                row["think_ms_paused"] = round(think_ms, 1)
                row["charged_latency_ms"] = round(charged_ms, 1)
            if clock_mode in ("rt", "trt"):
                # `hold_tics` is how long this chunk's key state kept running
                # after its 6 tics because the next decision had not arrived,
                # and `staleness_tics` is the age of the frame this chunk was
                # decided from. Note `vlm_response` on a row is the decision
                # ISSUED during this cycle, while `keys`/`mouse` are the chunk
                # EXECUTED in it -- one cycle apart, by construction.
                row["hold_tics"] = hold_tics
                row["staleness_tics"] = staleness_tics[-1]
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")

            if info["done"]:
                termination = "episode_finished"
                break
        else:
            termination = "max_steps"

        if video is not None:
            video.append(obs)
        score = env.episode_score()
        gvars = env.game_variables()
        if termination == "episode_finished" and (gvars.get("HEALTH", 1.0) or 0.0) <= 0:
            termination = "death"
    finally:
        trace.close()
        if video is not None:
            video.close()
        env.close()
        close_agent = getattr(agent, 'close', None)
        if callable(close_agent):
            close_agent()

    n = max(1, len(latencies))
    result = {
        "game": game,
        "episode_index": episode_index,
        "seed": seed,
        "agent": {
            "model": served_name,
            "kind": agent_kind,
            "extra": {
                "base_url": getattr(agent, "base_url", None),
                "reasoning": reasoning,
                "reasoning_style": reasoning_style,
                "prompt_style": prompt_style,
            },
        },
        "metrics": {
            "game": {
                "score": score,
                "score_mode": spec.score,
                "total_reward": total_reward,
                "raw": gvars,
            },
            "steps": len(latencies),
            "tics": env.tics_elapsed,
            "latency": {
                "n": len(latencies),
                "mean_ms": round(sum(latencies) / n, 1),
                "p50_ms": _percentile(latencies, 0.50),
                "p90_ms": _percentile(latencies, 0.90),
            },
            "action": {
                "unparsed": agent.n_unparsed,
                "unparsed_rate": round(agent.n_unparsed / n, 4),
                "noop": n_noop,
                "noop_rate": round(n_noop / n, 4),
                "attack_steps": n_attack,
                # Denominator is CHUNKS, not decisions: firing is a property of
                # what the world executed. The two are equal under pdq/lcrt;
                # under rt a held action keeps firing without a new decision.
                "attack_rate": round(n_attack / max(1, n_chunks), 4),
                # An FPS agent that never aims is structurally scoring 0 no
                # matter how good the model is; the gate flags it as NO-AIM.
                "mouse_nonzero_rate": round(n_mouse / n, 4),
                "mean_abs_x": round(sum(abs_x) / n, 2),
                "key_uses": dict(key_uses.most_common(20)),
                "server_errors": agent.n_errors,
                "dropped_tokens": dict(dropped.most_common(20)),
            },
            "reasoning": {
                "steps_with_think": n_think,
                "think_rate": round(n_think / n, 4),
                "mean_think_chars": round(statistics.mean(think_chars), 1) if think_chars else 0.0,
                # Turns whose generation ended while still inside <think>, so no
                # action was ever emitted. A high rate means the serving
                # configuration is measuring itself, not the policy;
                # `unclosed_finish_reasons` says which knob is at fault.
                "unclosed_think": n_truncated,
                "unclosed_think_rate": round(n_truncated / n, 4),
                "unclosed_finish_reasons": dict(unclosed_why),
                "mean_completion_tokens": (round(statistics.mean(completion_tokens), 1)
                                           if completion_tokens else None),
            },
            "clock": {
                "mode": clock_mode,
                # Under rt one "chunk" is one decision cycle of
                # max(CHUNK_STEPS, latency) tics, so cycles != tics/6.
                "chunks": n_chunks,
                "fastforward_tics": env.fastforward_tics,
                # Under lcrt this is the share of game time the policy acted in
                # at all (the rest is forced no-op). Under rt nothing is forced,
                # so it is ~1.0 minus the episode-start gap before the first
                # decision lands -- read `decision_rate` and `staleness` there.
                "agent_controlled_fraction": (
                    round(1.0 - (env.fastforward_tics + noop_startup_tics)
                          / env.tics_elapsed, 4)
                    if env.tics_elapsed else None
                ),
                # rt only. `decision_rate` = the share of game tics running a
                # freshly decided chunk rather than a held one, which works out
                # to min(1, chunk / latency). `staleness` = age of the frame the
                # executing chunk was decided from, the price every real
                # pipelined agent pays (OpenAI Five 167-267ms, Lumine ~200ms,
                # humans ~250ms).
                "decisions": len(latencies) if clock_mode in ("rt", "trt") else None,
                "decision_rate": (
                    round(CHUNK_STEPS * len(latencies) / env.tics_elapsed, 4)
                    if clock_mode in ("rt", "trt") and env.tics_elapsed else None),
                "held_tics": env.held_tics if clock_mode in ("rt", "trt") else None,
                "mean_staleness_tics": (round(statistics.mean(staleness_tics), 2)
                                        if staleness_tics else None),
                "mean_staleness_ms": (
                    round(statistics.mean(staleness_tics) / TICS_PER_SECOND * 1000, 1)
                    if staleness_tics else None),
                "startup_noop_tics": noop_startup_tics if clock_mode in ("rt", "trt") else None,
                "charged_latency_mean_ms": (round(sum(charged) / len(charged), 1)
                                            if clock_mode == "trt" and charged else None),
            },
        },
        "parameters": {
            "history_len": history_len,
            "history_reasoning_len": history_reasoning_len,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_max_tokens": reasoning_max_tokens,
            "prompt_style": prompt_style,
            "resize": list(resize) if resize else None,
            "resolution": list(resolution),
            "render_hud": render_hud,
            "max_steps": steps_budget,
            "clock_mode": clock_mode,
            "mouse_mode": mouse_mode,
            "turn_deg_per_full_x": turn_deg_per_full_x,
            "look_deg_per_full_y": look_deg_per_full_y,
            "doom_skill": spec.doom_skill,
            "episode_timeout_tics": spec.episode_timeout,
            "tics_per_step": 6,
            "tics_per_second": TICS_PER_SECOND,
        },
        "termination": termination,
        "timing": {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
            "wall_s": round(time.time() - t_start, 1),
        },
    }
    if objective:
        result["parameters"]["objective_set"] = objective_set
        result["parameters"]["objective"] = objective
        result["parameters"]["objective_pos"] = objective_pos
    if recap_last:
        result["parameters"]["recap_last"] = True
    if carry_think:
        result["parameters"]["carry_think"] = True
        result["parameters"]["n_think_carried"] = getattr(agent, "n_think_carried", 0)
    if image_only_user:
        result["parameters"]["image_only_user"] = True
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[episode end] {game} ep{episode_index} score={score} "
          f"steps={len(latencies)} termination={termination} "
          f"unparsed={agent.n_unparsed} noop={n_noop} attack={n_attack} "
          f"wall={result['timing']['wall_s']}s -> {run_dir}", flush=True)
    return result
