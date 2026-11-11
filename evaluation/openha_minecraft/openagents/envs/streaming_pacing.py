"""Real-time ("streaming") pacing for OpenHA rollouts.

The default rollout is *stop* pacing: the world freezes on the frame the model
sees, the model thinks for as long as it needs, and only then does the env step.
Model latency is free.

Streaming pacing charges that latency to the game clock, the way a human player
cannot pause between glances. It is simulated rather than threaded so it does not
depend on JVM step speed under a 30-way parallel eval:

    frame f_i is snapshotted -> model call (wall-clock latency L_i)
    -> the env advances ceil(L_i / tick_ms) ticks while "waiting", playing the
       PREVIOUS reply's remaining substeps first and then the idle action
    -> the new reply's substeps become the pending chunk

capture_mode=on_arrival (default) captures the next frame the instant a reply
lands and plays the plan during the following call (pipelined; a new reply
pre-empts what is left of the old plan). capture_mode=after_chunk instead plays
the whole reply first and only then captures, so the model sees what it just
did. With a 200 ms native window (4 Minecraft ticks) and multi-second API
latency, most game time is spent idle in either mode -- that gap is exactly what
this mode measures.
"""

import copy
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from openagents.agents.utils import action_mapping

IDLE_MODES = ("noop", "hold")
# When the next frame is captured and sent to the model:
#   on_arrival  : look the instant a reply lands, while its plan starts playing (default)
#   after_chunk : play the whole plan first, then look (model sees what it just did)
CAPTURE_MODES = ("on_arrival", "after_chunk")


def decode_response(agent, response: str, info: Dict[str, Any]) -> List[Dict]:
    """Decode one model response into a list of env actions (empty on failure)."""
    tokenizer = agent.action_tokenizer
    if isinstance(tokenizer, action_mapping.GuiActionTokenizer):
        actions = tokenizer.decode(response, is_gui_open=bool(info.get("isGuiOpen", False)))
    else:
        actions = tokenizer.decode(response)
    actions = list(actions) if actions else []
    # Parse failures come back as a single env-null action; that is "no plan", not a plan.
    null = getattr(tokenizer, "_env_null_action", None)
    if null is not None and len(actions) == 1 and _is_null_action(actions[0], null):
        return []
    return actions


def _is_null_action(action: Dict, null: Dict) -> bool:
    for key, ref in null.items():
        val = action.get(key)
        if hasattr(ref, "__len__"):
            if val is None or not np.array_equal(np.asarray(val), np.asarray(ref)):
                return False
        elif val != ref:
            return False
    return True


def idle_action(env, mode: str, last_action: Optional[Dict]) -> Dict:
    """What the agent does on ticks where no plan is pending.

    noop : release everything (the model's 200 ms plan is over, hands off)
    hold : keep the last key state, stop the mouse (a player mid-thought keeps
           holding W / the mouse button but does not keep turning)
    """
    if mode == "hold" and last_action is not None:
        action = copy.deepcopy(last_action)
        if "camera" in action:
            action["camera"] = np.zeros_like(np.asarray(action["camera"], dtype=float))
        return action
    if hasattr(env, "noop_action"):
        return env.noop_action()
    return env.action_space.no_op()


def run_streaming_episode(
    agent,
    env,
    obs,
    info,
    *,
    max_steps: int,
    tick_ms: float,
    max_latency_ms: float,
    idle_mode: str,
    rows: List[Dict],
    capture_mode: str = "on_arrival",
    min_latency_ms: float = 0.0,
    verbose: bool = False,
    logprobs_path: Optional[str] = None,
    record_logprobs: bool = False,
) -> Dict[str, Any]:
    """Drive one episode under streaming pacing.

    Appends one frame-aligned row per env step to ``rows`` (same schema as the
    stop loop, plus streaming fields). Returns a dict with ``success``,
    ``frames``, ``obs``/``info`` at exit and a ``stats`` block.
    """
    if idle_mode not in IDLE_MODES:
        raise ValueError(f"idle_mode must be one of {IDLE_MODES}, got {idle_mode!r}")
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"capture_mode must be one of {CAPTURE_MODES}, got {capture_mode!r}")
    tick_ms = float(tick_ms)
    if tick_ms <= 0:
        raise ValueError("tick_ms must be > 0")

    import json  # local: only needed when logprobs are recorded

    frame = 0
    response_id = -1
    pending: List[Dict] = []
    last_action: Optional[Dict] = None
    success = False
    latencies: List[float] = []
    stats = dict(
        calls=0,
        idle_ticks=0,
        chunk_ticks=0,
        preempted_substeps=0,
        capped_calls=0,
        floored_calls=0,
        empty_decodes=0,
    )

    pbar = tqdm(total=max_steps)
    while frame < max_steps:
        if capture_mode == "after_chunk":
            # Finish executing the last plan before looking at the screen again, so
            # the model always sees the outcome of what it just did. The world does
            # not wait for the model at any point; it only waits for the hands.
            while pending and frame < max_steps:
                action = pending.pop(0)
                stats["chunk_ticks"] += 1
                rows.append({
                    "points": agent._points,
                    "raw_action": "",
                    "action_type": agent._policy_type,
                    "model_generated": False,
                    "response_id": None,
                    "executing_response_id": response_id,
                    "stream_source": "chunk",
                })
                obs, reward, terminated, truncated, info = env.step(action)
                last_action = action
                frame += 1
                pbar.update(1)
                if reward > 0:
                    success = True
                    break
            if success or frame >= max_steps:
                break
        seen_frame = frame
        t0 = time.time()
        _, response = agent.gen_response(obs, agent._instruction, info, verbose=verbose)
        latency_ms = (time.time() - t0) * 1000.0
        agent._response = response
        agent.refresh_history()
        agent._policy_type = "raw_action"
        new_chunk = decode_response(agent, response, info)
        if not new_chunk:
            stats["empty_decodes"] += 1

        effective_ms = latency_ms
        if max_latency_ms and latency_ms > max_latency_ms:
            effective_ms = max_latency_ms
            stats["capped_calls"] += 1
        # Floor: a reply that lands faster than the previous chunk's play-out time
        # (200 ms = 4 ticks for the native window) must not pre-empt that chunk. The
        # world still advances at least min_latency_ms per call; the real latency is
        # kept in latency_ms for the stats.
        if min_latency_ms and effective_ms < min_latency_ms:
            effective_ms = min_latency_ms
            stats["floored_calls"] += 1
        inflight = max(1, int(math.ceil(effective_ms / tick_ms)))
        response_id += 1
        stats["calls"] += 1
        latencies.append(latency_ms)

        if record_logprobs and logprobs_path:
            with open(logprobs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "frame": seen_frame,
                    "response": agent.response,
                    "response_tokens": agent.response_tokens,
                    "top_logprobs": agent.logprobs,
                    "sampled_logprobs": agent.sampled_logprobs,
                    "latency_ms": round(latency_ms, 1),
                }, ensure_ascii=False) + "\n")

        # Advance the world for as long as this call was in flight. The previous
        # plan keeps executing; once it runs out the agent idles.
        for k in range(inflight):
            if frame >= max_steps:
                break
            if pending:
                action = pending.pop(0)
                source = "chunk"
                stats["chunk_ticks"] += 1
            else:
                action = idle_action(env, idle_mode, last_action)
                source = "idle"
                stats["idle_ticks"] += 1
            row = {
                "points": agent._points,
                "raw_action": response if k == 0 else "",
                "action_type": agent._policy_type,
                "model_generated": k == 0,
                "response_id": response_id if k == 0 else None,
                "executing_response_id": (response_id - 1) if source == "chunk" else None,
                "stream_source": source,
            }
            if k == 0:
                row["thought"] = agent.thought or ""
                if agent.reasoning_content:
                    row["reasoning_content"] = agent.reasoning_content
                row["latency_ms"] = round(latency_ms, 1)
                row["inflight_ticks"] = inflight
                row["applied_frame"] = seen_frame + inflight
            rows.append(row)
            obs, reward, terminated, truncated, info = env.step(action)
            last_action = action
            frame += 1
            pbar.update(1)
            if reward > 0:
                success = True
                break
        if success:
            break
        if pending:
            stats["preempted_substeps"] += len(pending)
        pending = new_chunk
    pbar.close()

    stats["latency_ms_mean"] = round(float(np.mean(latencies)), 1) if latencies else 0.0
    stats["latency_ms_p50"] = round(float(np.median(latencies)), 1) if latencies else 0.0
    stats["latency_ms_max"] = round(float(np.max(latencies)), 1) if latencies else 0.0
    stats["tick_ms"] = tick_ms
    stats["max_latency_ms"] = max_latency_ms
    stats["min_latency_ms"] = min_latency_ms
    stats["idle_mode"] = idle_mode
    stats["capture_mode"] = capture_mode
    stats["frames"] = frame
    return {"success": success, "frames": frame, "obs": obs, "info": info, "stats": stats}
