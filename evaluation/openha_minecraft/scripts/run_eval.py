#!/usr/bin/env python3
"""Concurrent batch evaluation over the 30-task OpenHA benchmark.

Runs `rollouts_per_task` rollouts for each task in the benchmark list
(10 kill_entity + 10 mine_block + 10 craft_item, from
CrossAgent/STRL/data_processor/utils/task_list.json), aggregates the
success rate per task / per category / overall, prints a summary table
and writes a JSON report.

Concurrency model: the sweep is flattened to ONE WORK ITEM PER ROLLOUT
(build_work_list) and handed to a ThreadPoolExecutor of --parallel_tasks
(alias --parallel) workers. Each in-flight rollout owns one Minecraft JVM,
so the ceiling is CPU/RAM, not the GPU; the VLM is a shared vLLM server
that batches the concurrent HTTP requests, so more concurrency also raises
GPU utilisation (a lone rollout leaves the GPU idle while Minecraft steps).

Pooling per rollout rather than per task is what makes --parallel mean
anything. With a per-task pool a task held its worker for all
`rollouts_per_task` rollouts sequentially, so the sweep could never finish
faster than rollouts_per_task * per_rollout_time however large --parallel
was, and a task that finished early left its slot idle. Flattened,
--parallel may exceed the task count and every free worker immediately
picks up the next outstanding rollout. The cost is that rollouts of the
same task can overlap — harmless, since each writes its own timestamped dir.

Success/failure is read straight off disk: every rollout writes either
`success.json` or `loss.json` into its own run directory
(`<record_path>/<model_id>-<output_mode>/<task>/<timestamp>/`). Because of
that, this script is **resumable** — rollouts already on disk are counted
and only the missing ones are executed (use the same --record_path).

Works for both back-ends:

  OpenHA baseline (vllm serve, OpenAI /v1):
    python scripts/run_eval.py \
        --model_id minecraft-openha-qwen2vl-7b-2509 \
        --model_ips localhost --model_ports 8001 \
        --output_mode text_action --raw_action_type text \
        --system_message_tag text_action \
        --rollouts_per_task 5 --record_path ./eval_runs

  Our model (fixed streaming /step server, instruction aligned):
    python scripts/run_eval.py \
        --model_id streaming \
        --model_ips <MODEL_IP> --model_ports 5000 \
        --vlm_client_mode streaming \
        --output_mode text_action --raw_action_type gui \
        --system_message_tag gui_native \
        --streaming_send_instruction \
        --maximum_history_length 0 --parallel_tasks 1 \
        --rollouts_per_task 5 --record_path ./eval_runs_streaming
"""
import argparse
import copy
import glob
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Make `examples` / `openagents` importable when run from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WORKER_PY = os.path.join(REPO_ROOT, "scripts", "_rollout_worker.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_eval")

DEFAULT_TASK_FILE = os.path.join(
    REPO_ROOT, "CrossAgent", "STRL", "data_processor", "utils", "task_list.json"
)
CATEGORY_ORDER = ["kill_entity", "mine_block", "craft_item"]


def category_of(task_name: str) -> str:
    return task_name.split(":", 1)[0]


def load_tasks(task_file: str, categories: str, explicit: str = ""):
    if explicit:
        # explicit comma-separated task list overrides task_file + categories
        tasks = [t.strip() for t in explicit.split(",") if t.strip()]
    else:
        with open(task_file, "r") as f:
            tasks = json.load(f)
        if categories and categories.lower() != "all":
            wanted = {c.strip() for c in categories.split(",") if c.strip()}
            tasks = [t for t in tasks if category_of(t) in wanted]
    # stable order: group by category, keep input order within category
    tasks.sort(key=lambda t: (CATEGORY_ORDER.index(category_of(t))
                              if category_of(t) in CATEGORY_ORDER else 99))
    return tasks


def build_rollout_args(a, task: str, base_url_override: str = None):
    """Build the argparse.Namespace that examples.rollout_openha.rollout() expects."""
    return argparse.Namespace(
        # --- routing / parsing ---
        output_mode=a.output_mode,
        raw_action_type=a.raw_action_type,
        gui_reasoning_mode=a.gui_reasoning_mode,
        gui_think_prefilled=a.gui_think_prefilled,
        gui_allow_missing_think=a.gui_allow_missing_think,
        gui_key_map=a.gui_key_map,
        env_pacing=a.env_pacing,
        stream_tick_ms=a.stream_tick_ms,
        stream_max_latency_ms=a.stream_max_latency_ms,
        stream_min_latency_ms=a.stream_min_latency_ms,
        stream_idle_action=a.stream_idle_action,
        stream_capture=a.stream_capture,
        vlm_client_mode=a.vlm_client_mode,
        streaming_send_instruction=a.streaming_send_instruction,
        instruction_prefix=a.instruction_prefix,
        per_turn_user_text=a.per_turn_user_text,
        native_gui_chat=a.native_gui_chat,
        native_mouse_pos=a.native_mouse_pos,
        system_message=a.system_message,
        system_message_tag=a.system_message_tag,
        # --- model service ---
        model_id=a.model_id,
        model_ips=a.model_ips,
        model_ports=a.model_ports,
        base_url=base_url_override if base_url_override is not None else a.base_url,
        model_path=a.model_path,
        api_key=a.api_key,
        # --- task / episode ---
        task=task,
        difficulty=a.difficulty,
        max_steps_num=a.max_steps_num,
        num_rollouts=1,            # always single-process here; parallelism is across tasks
        record_path=a.record_path,
        # --- history / chunking ---
        maximum_history_length=a.maximum_history_length,
        gui_history_content=a.gui_history_content,
        action_chunk_len=a.action_chunk_len,
        instruction_type=a.instruction_type,
        # --- backbones / tokenizer ---
        LLM_backbone=a.LLM_backbone,
        VLM_backbone=a.VLM_backbone,
        tokenizer_path=a.tokenizer_path,
        # --- aux policies (unused for text_action/gui, but rollout reads them) ---
        sam_path=a.sam_path,
        grounding_policy_path=a.grounding_policy_path,
        motion_policy_path=a.motion_policy_path,
        grounding_inference_interval=a.grounding_inference_interval,
        motion_inference_interval=a.motion_inference_interval,
        # --- sampling ---
        temperature=a.temperature,
        top_p=None if a.omit_top_p else a.top_p,
        top_k=a.top_k,
        top_logprobs=a.top_logprobs,
        get_sampled_logprobs=a.record_sampled_logprobs,
        extra_body=json.loads(a.extra_body_json) if a.extra_body_json else None,
        # --- enforce / misc ---
        enforce_format=a.enforce_format,
        enforce_prefix=a.enforce_prefix,
        verbose=a.verbose,
        fps=a.fps,
        no_render=a.no_render,
        render_width=a.render_width,
        render_height=a.render_height,
        model_input_width=a.model_input_width,
        model_input_height=a.model_input_height,
        instruction_lang=a.instruction_lang,
        gpu_per_rollout=a.gpu_per_rollout,
    )


def existing_counts(a, task: str):
    """(success, total) already on disk for this task under record_path."""
    base = os.path.join(a.record_path, f"{a.model_id}-{a.output_mode}", task)
    s = len(glob.glob(os.path.join(base, "*", "success.json")))
    l = len(glob.glob(os.path.join(base, "*", "loss.json")))
    return s, s + l


def _write_timeout_marker(a, task: str):
    """Record a killed/timed-out rollout as a loss so it counts as done (won't
    retry forever) and is distinguishable from a genuine in-game failure."""
    base = os.path.join(a.record_path, f"{a.model_id}-{a.output_mode}", task)
    d = os.path.join(base, datetime.now().strftime('%y-%m-%d-%H-%M-%S')
                     + '-' + task.replace(':', '_') + '-timeout-' + uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "loss.json"), "w") as f:
        json.dump({"frames": 0, "timeout": True, "task": task}, f, indent=2)


def _run_rollout_subprocess(a, task: str, timeout: float, base_url_override: str = None):
    """Run one rollout in its own process group and hard-kill it (plus its
    Minecraft JVM / Xvfb children) if it exceeds `timeout` seconds. A hung
    Minecraft instance blocks in native socket code that a thread cannot
    interrupt, so isolating each rollout in a killable subprocess is the only
    way to guarantee the sweep keeps making progress."""
    payload = json.dumps(vars(build_rollout_args(a, task, base_url_override=base_url_override)))
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # start_new_session=True -> child is a session/group leader, so os.killpg
    # reaches the whole tree (python -> xvfb-run -> java).
    p = subprocess.Popen(
        [sys.executable, WORKER_PY],
        stdin=subprocess.PIPE, env=env, start_new_session=True,
    )
    try:
        p.communicate(payload.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("[%s] rollout exceeded %ss — killing process group %d",
                     task, timeout, p.pid)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        _write_timeout_marker(a, task)
        return
    # A worker that dies before writing success.json/loss.json (OOM kill, JVM launch
    # failure, segfault in the native Malmo layer) would otherwise vanish silently: the
    # rollout never appears on disk and the summary quietly reports one rollout fewer.
    # Surface it, and leave nothing on disk so a resumed run retries this rollout.
    if p.returncode != 0:
        logger.error("[%s] rollout worker exited with code %s — no result written; "
                     "re-run with the same --record_path to retry it", task, p.returncode)


_LAUNCH_LOCK = threading.Lock()
_LAST_LAUNCH = [0.0]


def _stagger_launch(seconds: float):
    """Space out rollout process starts by `seconds`.

    MineStudio's InstanceManager picks a Malmo port by probing for a free one, but the JVM
    only binds it seconds later. Two rollouts starting at the same instant can therefore
    claim the same port; the loser dies with 'Address already in use' and its reset() then
    blocks on a socket nobody will open. Rollouts are separate PROCESSES here, so an
    in-process lock cannot serialise the port probe itself — spacing the launches is what
    keeps the race window from being hit, and it costs only parallel*seconds of one-time
    ramp-up per sweep.
    """
    if seconds <= 0:
        return
    with _LAUNCH_LOCK:
        wait = _LAST_LAUNCH[0] + seconds - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_LAUNCH[0] = time.monotonic()


_URL_SLOT_POOL = None


def run_one_rollout(a, task: str, label: str = ""):
    """Execute exactly ONE rollout of `task`. Never raises."""
    slot_url = _URL_SLOT_POOL.get() if _URL_SLOT_POOL is not None else None
    try:
        _stagger_launch(getattr(a, "launch_stagger", 0.0))
        timeout = getattr(a, "rollout_timeout", 0) or 0
        logger.info("[%s] starting rollout %s%s", task, label,
                    f" -> {slot_url}" if slot_url else "")
        try:
            if timeout > 0:
                _run_rollout_subprocess(a, task, timeout, base_url_override=slot_url)
            else:
                # Imported here, not at module scope: the watchdog path never needs it, and
                # pulling in torch/minestudio costs the parent process ~30s for nothing.
                from examples.rollout_openha import rollout
                rollout(build_rollout_args(a, task, base_url_override=slot_url))
        except Exception as e:  # one bad rollout must not abort the whole sweep
            logger.exception("[%s] rollout failed: %s", task, e)
    finally:
        if _URL_SLOT_POOL is not None:
            _URL_SLOT_POOL.put(slot_url)


def build_work_list(a, tasks):
    """Flatten the sweep into one entry per rollout still missing from disk."""
    work = []
    for t in tasks:
        done_s, done_total = existing_counts(a, t)
        remaining = max(0, a.rollouts_per_task - done_total)
        if done_total:
            logger.info("[%s] %d/%d already on disk, running %d more",
                        t, done_s, done_total, remaining)
        for i in range(remaining):
            work.append((t, f"{done_total + i + 1}/{a.rollouts_per_task}"))
    return work


def main():
    p = argparse.ArgumentParser(description="OpenHA 30-task benchmark evaluation")
    # model service
    p.add_argument("--model_id", required=True,
                   help="Model name (must match --served-model-name in vllm serve; "
                        "any label in streaming mode).")
    p.add_argument("--model_ips", default="localhost")
    p.add_argument("--model_ports", default="11000")
    p.add_argument("--base_url", default=None,
                   help="full OpenAI-compatible endpoint, e.g. https://api.openai.com/v1 or "
                        "https://dashscope.aliyuncs.com/compatible-mode/v1. Overrides "
                        "--model_ips/--model_ports. Use with --vlm_client_mode openai.")
    p.add_argument("--model_path", default="CraftJarvis/minecraft-openha-qwen2vl-7b-2509")
    p.add_argument("--api_key", default="EMPTY",
                   help="API key for --vlm_client_mode openai/anthropic. 'EMPTY' (default) falls "
                        "back to $OPENAI_API_KEY in openai mode; self-hosted vLLM ignores it.")
    p.add_argument("--vlm_client_mode", default="online",
                   choices=["online", "openai", "anthropic", "vllm", "hf", "streaming"],
                   help="online = self-hosted vLLM /v1 (injects vLLM-only top_k/skip_special_tokens); "
                        "openai = commercial OpenAI-compatible API (no vLLM extras injected).")
    # routing / parsing
    p.add_argument("--output_mode", default="text_action",
                   choices=["greedy", "grounding", "motion", "text_action"])
    p.add_argument("--raw_action_type", default="text", choices=["reserved", "text", "gui"])
    p.add_argument("--gui_reasoning_mode", default="direct",
                   choices=["direct", "optional", "forced"],
                   help="raw_action_type=gui output contract: direct rejects think; "
                        "optional accepts empty or non-empty complete think; forced "
                        "requires a non-empty complete think before every action.")
    p.add_argument("--system_message_tag", default="text_action",
                   help="'text_action' for OpenHA baseline, 'gui_native' for our model")
    p.add_argument("--system_message", default=None)
    p.add_argument("--streaming_send_instruction", action="store_true",
                   help="streaming mode: forward the task goal to /step (align instruction with OpenHA)")
    p.add_argument("--instruction_prefix", action="store_true",
                   help="non-streaming modes: wrap the per-task goal as '你当前要完成的任务是：{goal}' "
                        "(\"The task you need to complete now is: {goal}\") "
                        "before appending to the system prompt (matches the streaming server phrasing)")
    p.add_argument("--per_turn_user_text", default="",
                   help="non-streaming modes: text put on EVERY user turn (history + current), e.g. "
                        "'current game screen'. Realigns online multi-image history with a "
                        "streaming-trained model; leave empty for OpenHA-native models.")
    p.add_argument("--native_gui_chat", action="store_true",
                   help="reproduce the training-pipeline chat format exactly: system prompt as a real "
                        "system role + every user turn = 'current game screen\\n<image>\\ncurrent mouse "
                        "position\\n{pos}' (no task instruction). Fixes no-op collapse with history>0 for "
                        "the streaming-trained model. Pair with --system_message <the training prompt>.")
    p.add_argument("--native_mouse_pos", default="[500, 500]",
                   help="native_gui_chat: the mouse-position line value (Minecraft has no cursor; "
                        "matches the streaming server's default norm_xy). Pass an empty string "
                        "to omit the mouse-position line entirely.")
    # eval control
    p.add_argument("--rollouts_per_task", type=int, default=5,
                   help="Rollouts per task (more = more stable success rate)")
    p.add_argument("--max_steps_num", type=int, default=200)
    p.add_argument("--categories", default="all",
                   help="Comma-separated categories to run, e.g. 'kill_entity,mine_block'. Default: all.")
    p.add_argument("--tasks", default="",
                   help="Explicit comma-separated task list (e.g. 'kill_entity:sheep,kill_entity:cow'). "
                        "Overrides --task_file/--categories.")
    p.add_argument("--parallel_tasks", "--parallel", type=int, default=30,
                   dest="parallel_tasks",
                   help="max ROLLOUTS running concurrently (each holds one Minecraft JVM, "
                        "~2.3 cores / 2 GB). Pooling is per-rollout, not per-task, so this "
                        "may exceed the task count and rollouts of the same task can "
                        "overlap. Set 1 for a single Minecraft instance.")
    p.add_argument("--difficulty", default="zero", choices=["zero", "hard"])
    p.add_argument("--record_path", required=True)
    p.add_argument("--task_file", default=DEFAULT_TASK_FILE)
    # history / chunking
    p.add_argument("--maximum_history_length", type=int, default=15)
    p.add_argument("--gui_history_content", choices=["full", "action_only"], default="full",
                   help="GUI assistant history: full response or action block only")
    p.add_argument("--action_chunk_len", type=int, default=1)
    p.add_argument("--instruction_type", default="final")
    # backbones
    p.add_argument("--LLM_backbone", default="")
    p.add_argument("--VLM_backbone", default="")
    p.add_argument("--tokenizer_path", default="")
    # aux policies (kept for rollout() compatibility)
    p.add_argument("--sam_path", default="facebook/sam2-hiera-base-plus")
    p.add_argument("--grounding_policy_path", default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    p.add_argument("--motion_policy_path", default="CraftJarvis/Minecraft-Motion_policy-2509")
    p.add_argument("--grounding_inference_interval", type=int, default=4)
    p.add_argument("--motion_inference_interval", type=int, default=4)
    # sampling / misc
    p.add_argument("--temperature", "-t", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.99)
    p.add_argument("--omit_top_p", action="store_true",
                   help="do not send top_p to APIs that reject the parameter")
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--top_logprobs", type=int, default=0,
                   help="request and record the top-K token logprobs from vLLM; 0 disables")
    p.add_argument("--record_sampled_logprobs", action="store_true",
                   help="record the sampled token logprob without requesting top-K alternatives")
    p.add_argument("--gui_think_prefilled", action="store_true",
                   help="the checkpoint's chat_template.jinja ends its generation prompt "
                        "with '<think>\\n', so the response starts with bare reasoning and "
                        "only closes with </think>. Accepts that shape in addition to the "
                        "strict <think>...</think> wrapper. Check the template before "
                        "setting it: grep \"<think>\" <ckpt>/chat_template.jinja")
    p.add_argument("--gui_allow_missing_think", action="store_true",
                   help="--gui_reasoning_mode optional: also accept an action with NO think "
                        "tags at all. Our checkpoints always emit the wrapper (empty when they "
                        "choose not to reason), so leave this off for them — a missing wrapper "
                        "is a real protocol mismatch. Commercial models follow the prose prompt "
                        "and omit the empty tag, which otherwise makes every step a no_op.")
    p.add_argument("--gui_key_map", default="wasd", choices=["wasd", "arrows", "both"],
                   help="movement binding accepted by the native GUI decoder. 'arrows' "
                        "moves on Up/Down/Left/Right and DROPS W/S/A/D (prompt-following "
                        "probe; pair with --system_message_tag gui_native_optional_en_arrows); "
                        "'both' accepts either binding.")
    p.add_argument("--env_pacing", default="stop", choices=["stop", "streaming"],
                   help="stop = the world freezes while the model thinks (default). streaming = "
                        "each call's wall-clock latency is charged to the game clock: the env "
                        "advances ceil(latency/stream_tick_ms) ticks per call, playing the previous "
                        "plan then idling, so slow models miss the window like a human would. "
                        "Independent of --vlm_client_mode streaming (that is the /step backend).")
    p.add_argument("--stream_tick_ms", type=float, default=50.0,
                   help="env_pacing=streaming: ms of game time per Minecraft tick (20 Hz).")
    p.add_argument("--stream_max_latency_ms", type=float, default=15000.0,
                   help="env_pacing=streaming: cap on charged latency per call; 0 disables.")
    p.add_argument("--stream_min_latency_ms", type=float, default=0.0,
                   help="env_pacing=streaming: floor on charged latency per call; a reply faster "
                        "than this still advances the world this many ms so the previous chunk "
                        "plays out (200 = the 4-tick native window). 0 disables.")
    p.add_argument("--stream_idle_action", default="noop", choices=["noop", "hold"],
                   help="env_pacing=streaming: action on ticks with no pending plan. noop releases "
                        "everything; hold keeps the last key state with the mouse still.")
    p.add_argument("--stream_capture", default="on_arrival", choices=["on_arrival", "after_chunk"],
                   help="env_pacing=streaming: on_arrival = capture the instant a reply lands and "
                        "play it during the next call (pipelined; default). after_chunk = execute "
                        "the whole reply, then capture the next frame (model sees its own result).")
    p.add_argument("--extra_body_json", default="",
                   help="JSON dict of backend-specific sampling params passed through as "
                        "the OpenAI client's extra_body. Empty (default) changes nothing. "
                        "Checkpoints that roll out a whole trajectory instead of one step "
                        "need '{\"stop_token_ids\": [151670], "
                        "\"include_stop_str_in_output\": true}' to stop at the first "
                        "<|action_end|> and keep the marker.")
    p.add_argument("--enforce_format", action="store_true")
    p.add_argument("--enforce_prefix", default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no_render", action="store_true",
                   help="skip MP4 video rendering at end of each rollout to speed up eval")
    p.add_argument("--render_width", type=int, default=None,
                   help="Minecraft native render width (default 640; normally do not override)")
    p.add_argument("--render_height", type=int, default=None,
                   help="Minecraft native render height (default 360; normally do not override)")
    p.add_argument("--model_input_width", type=int, default=None,
                   help="resize native POV to this width only before VLM encoding")
    p.add_argument("--model_input_height", type=int, default=None,
                   help="resize native POV to this height only before VLM encoding")
    p.add_argument("--instruction_lang", type=str, default="zh", choices=["en", "zh"],
                   help="task instruction language: 'en' = English task_description (reference "
                        "model); 'zh' = Chinese conversion (our trained model). Default zh.")
    p.add_argument("--gpu_per_rollout", type=float, default=0.5)
    p.add_argument("--launch_stagger", type=float, default=2.0,
                   help="seconds to space out rollout process starts. Guards against two "
                        "Minecraft instances racing for the same Malmo port (which kills "
                        "one of them). Costs only parallel*stagger of one-time ramp-up.")
    p.add_argument("--rollout_timeout", type=float, default=1200,
                   help="hard wall-clock limit (seconds) per rollout. Each rollout runs in "
                        "its own process group and is killed (with its Minecraft JVM) if it "
                        "hangs past this, so the sweep never stalls. 0 disables the watchdog "
                        "(runs in-process, no protection). Default 1200 (20 min).")
    a = p.parse_args()

    if a.gui_reasoning_mode != "direct" and a.raw_action_type != "gui":
        p.error("--gui_reasoning_mode optional|forced requires --raw_action_type gui")

    global _URL_SLOT_POOL
    if a.base_url and "," in a.base_url:
        urls = [u.strip() for u in a.base_url.split(",") if u.strip()]
        _URL_SLOT_POOL = queue.Queue()
        for i in range(max(1, a.parallel_tasks)):
            _URL_SLOT_POOL.put(urls[i % len(urls)])
        logger.info("Initialized 1-to-1 URL slot pool (%d slots across %d URLs): %s",
                    max(1, a.parallel_tasks), len(urls), urls)

    if a.vlm_client_mode == "streaming" and a.parallel_tasks > 1:
        # Per-rollout unique session_id (vlm_client._init_streaming_backend) makes this
        # safe — every rollout gets its own server-side KV session. But: (1) the streaming
        # server serializes inference (_inference_lock), so model calls still queue, and
        # (2) each rollout spawns its own Minecraft JVM, so N in flight = N heavy MC instances.
        logger.warning("streaming + parallel_tasks=%d: sessions are isolated, but the server "
                       "serializes inference and each rollout launches its own Minecraft "
                       "instance. Make sure the box has the CPU/RAM for %d MC JVMs.",
                       a.parallel_tasks, a.parallel_tasks)

    os.makedirs(a.record_path, exist_ok=True)
    tasks = load_tasks(a.task_file, a.categories, a.tasks)
    work = build_work_list(a, tasks)
    logger.info("Evaluating %d tasks x %d rollouts = %d rollouts to run, %d concurrent "
                "(mode=%s, raw_action=%s, gui_reasoning_mode=%s, send_instruction=%s)",
                len(tasks), a.rollouts_per_task, len(work), a.parallel_tasks,
                a.vlm_client_mode, a.raw_action_type, a.gui_reasoning_mode,
                a.streaming_send_instruction)

    t0 = time.time()
    if a.parallel_tasks > 1 and len(work) > 1:
        with ThreadPoolExecutor(max_workers=a.parallel_tasks) as ex:
            futs = [ex.submit(run_one_rollout, a, t, lbl) for t, lbl in work]
            for n, fut in enumerate(as_completed(futs), 1):
                fut.result()  # run_one_rollout swallows its own errors
                if n % 10 == 0 or n == len(futs):
                    logger.info("progress: %d/%d rollouts done (%.1f min elapsed)",
                                n, len(futs), (time.time() - t0) / 60)
    else:
        for t, lbl in work:
            run_one_rollout(a, t, lbl)

    # Counts are read back off disk, so they include rollouts from earlier resumed runs.
    results = {t: existing_counts(a, t) for t in tasks}
    report = summarize(a, results, time.time() - t0)
    out = os.path.join(a.record_path,
                       f"eval_summary_{a.model_id}_{datetime.now().strftime('%y%m%d-%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON report -> %s", out)


def summarize(a, results: dict, elapsed: float) -> dict:
    by_cat = {}
    for task, (s, total) in results.items():
        by_cat.setdefault(category_of(task), []).append((task, s, total))

    print("\n" + "=" * 62)
    print(f"  EVALUATION SUMMARY  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  model={a.model_id}  mode={a.vlm_client_mode}  raw_action={a.raw_action_type}"
          f"  gui_reasoning_mode={a.gui_reasoning_mode}"
          f"  gui_think_prefilled={a.gui_think_prefilled}"
          f"  gui_allow_missing_think={a.gui_allow_missing_think}"
          f"  gui_key_map={a.gui_key_map}"
          f"  env_pacing={a.env_pacing}"
          f"  send_instruction={a.streaming_send_instruction}")
    print("=" * 62)

    cat_summary, all_s, all_t = {}, 0, 0
    for cat in [c for c in CATEGORY_ORDER if c in by_cat] + \
               [c for c in by_cat if c not in CATEGORY_ORDER]:
        rows = by_cat[cat]
        cs = sum(s for _, s, _ in rows)
        ct = sum(t for _, _, t in rows)
        all_s += cs
        all_t += ct
        cat_summary[cat] = {"success": cs, "total": ct, "sr": (cs / ct if ct else 0.0)}
        print(f"\n  [{cat.upper()}]")
        print("  " + "-" * 58)
        print(f"  {'Task':<40}{'Success':>9}{'SR':>9}")
        print("  " + "-" * 58)
        for task, s, t in rows:
            sr = f"{s / t * 100:.1f}%" if t else "n/a"
            print(f"  {task:<40}{f'{s}/{t}':>9}{sr:>9}")
        print("  " + "-" * 58)
        print(f"  {'Category total':<40}{f'{cs}/{ct}':>9}"
              f"{(f'{cs / ct * 100:.1f}%' if ct else 'n/a'):>9}")

    print("\n" + "=" * 62)
    print(f"  {'OVERALL':<40}{f'{all_s}/{all_t}':>9}"
          f"{(f'{all_s / all_t * 100:.1f}%' if all_t else 'n/a'):>9}")
    print(f"  Elapsed: {elapsed / 60:.1f} min")
    print("=" * 62 + "\n")

    return {
        "model_id": a.model_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "vlm_client_mode": a.vlm_client_mode,
            "base_url": a.base_url or f"http://{a.model_ips}:{a.model_ports}/v1",
            "raw_action_type": a.raw_action_type,
            "gui_reasoning_mode": a.gui_reasoning_mode,
            "gui_think_prefilled": a.gui_think_prefilled,
            "gui_allow_missing_think": a.gui_allow_missing_think,
            "gui_key_map": a.gui_key_map,
            "gui_mouse_mapping": os.environ.get("GUI_MOUSE_MAPPING", ""),
            "gui_yaw_scale": os.environ.get("GUI_YAW_SCALE", ""),
            "gui_pitch_axis_scale": os.environ.get("GUI_PITCH_AXIS_SCALE", ""),
            "gui_resample_substeps": os.environ.get("GUI_RESAMPLE_SUBSTEPS", ""),
            "gui_state_aware_scale": os.environ.get("GUI_STATE_AWARE_SCALE", "0"),
            "gui_open_yaw_scale": os.environ.get("GUI_OPEN_YAW_SCALE", ""),
            "gui_open_pitch_axis_scale": os.environ.get("GUI_OPEN_PITCH_AXIS_SCALE", ""),
            "system_message_tag": a.system_message_tag,
            "env_pacing": a.env_pacing,
            "stream_tick_ms": a.stream_tick_ms,
            "stream_max_latency_ms": a.stream_max_latency_ms,
            "stream_min_latency_ms": a.stream_min_latency_ms,
            "stream_idle_action": a.stream_idle_action,
            "stream_capture": a.stream_capture,
            "streaming_send_instruction": a.streaming_send_instruction,
            "rollouts_per_task": a.rollouts_per_task,
            "max_steps_num": a.max_steps_num,
            "difficulty": a.difficulty,
            "parallel_tasks": a.parallel_tasks,
        },
        "tasks": {t: {"success": s, "total": tot, "sr": (s / tot if tot else 0.0)}
                  for t, (s, tot) in results.items()},
        "categories": cat_summary,
        "overall": {"success": all_s, "total": all_t, "sr": (all_s / all_t if all_t else 0.0)},
        "elapsed_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    main()
