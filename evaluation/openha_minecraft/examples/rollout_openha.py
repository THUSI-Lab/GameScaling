import argparse 
import ray
from tqdm import tqdm
import random 
import os 
import json 
from datetime import datetime
import logging
import torch
import uuid 
import time
from openagents.utils.render import render_video
from openagents.utils.file_op import save_render_videos, clip_action_info
from openagents.envs.tasks.task_manager import choose_available_task
from openagents.envs import DEFAULT_MAXIMUM_BUFFER_SIZE
from openagents.envs.env import env_init
from openagents.envs.streaming_pacing import run_streaming_episode
from openagents.agents.openha import OpenHA

# Configure logging format and level
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')

def instruction_from_task(task:str) -> str:
    """Convert a task string into a natural instruction for the agent."""
    if ':' in task:
        obj_name = task.split(":")[-1]
        return f"Pinpoint the {obj_name} and track it."
    else:
        return task

def rollout_wrapper(args):
    """Wrapper for rollout function with error handling (useful for Ray parallel execution)."""
    try:
        rollout(args)
    except Exception as e:
        print(e)
    return args.task

# @ray.remote(num_gpus=0.5)
def rollout(args):
    """Main rollout loop for running an OpenHA agent in the Minecraft environment."""
    
    # Task setup and environment configuration
    tasks = args.task.split(",")
    task = random.choice([p.strip() for p in tasks])
    task_config = choose_available_task(task, difficulty=args.difficulty)
    task_name = task_config['task_name']
    model_id = args.model_id + "-" + args.output_mode
    max_buffer_size = DEFAULT_MAXIMUM_BUFFER_SIZE if args.num_rollouts > 10 else 1

    # Save directory setup
    rollout_path = os.path.join(
        args.record_path, 
        model_id, 
        task_name,
        datetime.now().strftime('%y-%m-%d-%H-%M-%S')+'-'+task_name.replace(":", "_")+'-'+str(uuid.uuid4()).replace('-', '')[:8])
    os.makedirs(rollout_path, exist_ok=True)
    raw_action_file_path = os.path.join(rollout_path, "raw_action.jsonl")
    
    # Determine segment type based on task name
    if "mine_block" in task_name:
        segment_type = "Mine"
    elif "kill_entity" in task_name:
        segment_type = "Hunt"
    elif "craft_item" in task_name:
        segment_type = "Craft"
    else:
        segment_type = "Explore"

    # Agent initialization with configuration.
    # --base_url (e.g. https://api.openai.com/v1) takes precedence over the
    # self-hosted ip:port form; getattr so an older args payload still works.
    model_url = getattr(args, "base_url", None) or \
        f"http://{random.choice(args.model_ips.split(','))}:{random.choice(args.model_ports.split(','))}/v1"
    agent = OpenHA(
        output_mode = args.output_mode,
        raw_action_type = args.raw_action_type,
        gui_reasoning_mode = args.gui_reasoning_mode,
        gui_think_prefilled = getattr(args, "gui_think_prefilled", False),
        gui_allow_missing_think = getattr(args, "gui_allow_missing_think", False),
        gui_key_map = getattr(args, "gui_key_map", "wasd"),
        vlm_client_mode = args.vlm_client_mode,
        streaming_send_instruction = args.streaming_send_instruction,
        instruction_prefix = args.instruction_prefix,
        per_turn_user_text = args.per_turn_user_text,
        native_gui_chat = args.native_gui_chat,
        native_mouse_pos = args.native_mouse_pos,
        model_input_width = getattr(args, "model_input_width", None),
        model_input_height = getattr(args, "model_input_height", None),
        model_path = args.model_path,
        grounding_policy_path = args.grounding_policy_path,
        motion_policy_path = args.motion_policy_path,
        sam_path = args.sam_path,
        segment_type = segment_type,
        model_id = args.model_id, 
        model_url = model_url, 
        api_key = getattr(args, "api_key", None) or "EMPTY",
        maximum_history_length=args.maximum_history_length,
        gui_history_content=getattr(args, "gui_history_content", "full"),
        action_chunk_len=args.action_chunk_len,
        instruction_type=args.instruction_type,
        LLM_backbone=args.LLM_backbone,
        VLM_backbone=args.VLM_backbone,
        tokenizer_path=args.tokenizer_path,
        
        system_message=args.system_message,
        system_message_tag = args.system_message_tag,
        enforce_format= args.enforce_format,
        enforce_prefix = args.enforce_prefix,
        temperature=args.temperature,
        top_k = args.top_k,
        top_p = args.top_p,
        top_logprobs = getattr(args, "top_logprobs", 0) or None,
        get_sampled_logprobs = getattr(args, "get_sampled_logprobs", False),
        # Backend-specific sampling params (run_eval.py --extra_body_json). getattr so
        # an args payload written before that flag existed still works.
        extra_body = getattr(args, "extra_body", None),
        grounding_inference_interval = args.grounding_inference_interval,
        motion_inference_interval = args.motion_inference_interval,
    )
    
    # Sleep randomly to avoid contention when launching multiple rollouts in parallel
    time.sleep(random.random())
    
    # Initialize environment
    env = env_init(task_config, rollout_path, args,)
    noop_action = env.noop_action()
    obs, reward, terminated, truncated, info = env.step(noop_action)

    # Reset the agent with task description.
    # Chinese task instructions for the 30-task benchmark (aligns the goal with the
    # model's Chinese training). Falls back to the English task_description if missing.
    ZH_TASK_INSTRUCTION = {
        # mine_block: break the block
        "mine_block:crafting_table": "面前有一个工作台，你必须把它破坏掉才算任务完成",
        "mine_block:sugar_cane": "面前有甘蔗，你必须把它破坏掉才算任务完成",
        "mine_block:hay_block": "面前有一个干草块，你必须把它破坏掉才算任务完成",
        "mine_block:oak_log": "面前有橡木原木，你必须把它破坏掉才算任务完成",
        "mine_block:grass_block": "面前有草方块，你必须把它破坏掉才算任务完成",
        "mine_block:sand": "面前有沙子，你必须把它破坏掉才算任务完成",
        "mine_block:dirt": "面前有泥土方块，你必须把它破坏掉才算任务完成",
        "mine_block:oak_leaves": "面前有橡树树叶，你必须把它破坏掉才算任务完成",
        "mine_block:birch_log": "面前有白桦木原木，你必须把它破坏掉才算任务完成",
        "mine_block:chest": "面前有一个箱子，你必须把它破坏掉才算任务完成",
        # craft_item: craft the item
        "craft_item:honeycomb_block": "你必须制作出一个蜜脾块才算任务完成",
        "craft_item:sea_lantern": "你必须制作出一个海晶灯才算任务完成",
        "craft_item:netherite_scrap": "你必须制作出下界合金碎片才算任务完成",
        "craft_item:cooked_salmon": "你必须制作出熟鲑鱼才算任务完成",
        "craft_item:paper": "你必须制作出纸才算任务完成",
        "craft_item:enchanting_table": "你必须制作出一个附魔台才算任务完成",
        "craft_item:cooked_cod": "你必须制作出熟鳕鱼才算任务完成",
        "craft_item:emerald": "你必须制作出绿宝石才算任务完成",
        "craft_item:arrow": "你必须制作出箭才算任务完成",
        "craft_item:shulker_box": "你必须制作出一个潜影盒才算任务完成",
        # kill_entity: kill the mob
        "kill_entity:salmon": "面前有一条鲑鱼，你必须把它杀死才算任务完成",
        "kill_entity:parrot": "面前有一只鹦鹉，你必须把它杀死才算任务完成",
        "kill_entity:sheep": "面前有一只羊，你必须把它杀死才算任务完成",
        "kill_entity:zombie": "面前有一只僵尸，你必须把它杀死才算任务完成",
        "kill_entity:cod": "面前有一条鳕鱼，你必须把它杀死才算任务完成",
        "kill_entity:chicken": "面前有一只鸡，你必须把它杀死才算任务完成",
        "kill_entity:iron_golem": "面前有一个铁傀儡，你必须把它杀死才算任务完成",
        "kill_entity:slime": "面前有一只史莱姆，你必须把它杀死才算任务完成",
        "kill_entity:villager": "面前有一个村民，你必须把它杀死才算任务完成",
        "kill_entity:cow": "面前有一头牛，你必须把它杀死才算任务完成",
    }
    # Instruction language:
    #   en -> original English benchmark task_description (for the English-trained
    #         reference model minecraft-openha-qwen2vl-7b-2509 / run_eval_openha.sh)
    #   zh -> Chinese conversion above (for our Chinese-trained model)
    if getattr(args, "instruction_lang", "zh") == "en":
        instruction = task_config['task_description']
    else:
        instruction = ZH_TASK_INSTRUCTION.get(task_name, task_config['task_description'])
    agent.reset(instruction=instruction, task_name = task_name, )

    # Rollout loop variables
    success_state = False
    reward = 0
    start_time = time.time()
    run_frame_idx = 0

    # Buffer for raw actions to save periodically
    processed_raw_actions = [{"raw_action": "", "model_generated": False, "response_id": None}]
    # A native GUI response contains several substeps and is replayed over several
    # Minecraft ticks.  Keep raw_action.jsonl frame-aligned, but store the (often
    # long) model text only once instead of duplicating it on every replay tick.
    model_response_id = -1
    record_logprobs = bool(
        getattr(args, "top_logprobs", 0)
        or getattr(args, "get_sampled_logprobs", False)
    )
    logprobs_path = os.path.join(rollout_path, "logprobs.jsonl")
    env_pacing = getattr(args, "env_pacing", "stop")
    stream_stats = None

    try:
        if env_pacing == "streaming":
            # Real-time pacing: model latency is charged to the game clock. See
            # openagents/envs/streaming_pacing.py for the exact simulation.
            result = run_streaming_episode(
                agent, env, obs, info,
                max_steps=args.max_steps_num,
                tick_ms=getattr(args, "stream_tick_ms", 50.0),
                max_latency_ms=getattr(args, "stream_max_latency_ms", 15000.0),
                min_latency_ms=getattr(args, "stream_min_latency_ms", 0.0),
                idle_mode=getattr(args, "stream_idle_action", "noop"),
                capture_mode=getattr(args, "stream_capture", "on_arrival"),
                rows=processed_raw_actions,
                verbose=args.verbose,
                logprobs_path=logprobs_path,
                record_logprobs=record_logprobs,
            )
            success_state = result["success"]
            run_frame_idx = result["frames"]
            obs, info = result["obs"], result["info"]
            stream_stats = result["stats"]
            model_response_id = stream_stats["calls"] - 1
        # Main environment-agent interaction loop (stop pacing)
        for run_frame_idx in (tqdm(range(args.max_steps_num)) if env_pacing != "streaming" else ()):
            # Query agent for next action
            action = agent.get_action(obs=obs, info=info,verbose=args.verbose)
            if agent.action_type == 'agent':
                action = env.agent_action_to_env_action(action)
            if action == None:
                action = env.action_space.sample()
            elif action == 'no_op':
                action = env.action_space.no_op()

            # Save action details for logging. On cached GUI-substep ticks,
            # agent._response is still the previous model response; writing it again
            # made raw_action.jsonl look as though the model generated the same text
            # 3-4 times. Preserve one row per env frame but write the response only
            # on the tick that actually queried the model.
            if agent.generated_this_step:
                model_response_id += 1
            processed_raw_action = {
                "points": agent._points,
                "raw_action": agent._response if agent.generated_this_step else "",
                "action_type": agent._policy_type,
                "model_generated": agent.generated_this_step,
                "response_id": model_response_id if model_response_id >= 0 else None,
            }
            if agent.generated_this_step:
                processed_raw_action["thought"] = agent.thought or ""
                if agent.reasoning_content:
                    processed_raw_action["reasoning_content"] = agent.reasoning_content
            processed_raw_actions.append(processed_raw_action)

            # One model response is replayed over several cached GUI substeps. Log
            # probabilities only on the tick that actually queried the model.
            if record_logprobs and agent.generated_this_step:
                with open(logprobs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "frame": run_frame_idx,
                        "response": agent.response,
                        "response_tokens": agent.response_tokens,
                        "top_logprobs": agent.logprobs,
                        "sampled_logprobs": agent.sampled_logprobs,
                    }, ensure_ascii=False) + "\n")

            # Write actions to file when buffer is full
            if len(processed_raw_actions) % max_buffer_size == 0:
                with open(raw_action_file_path, 'a', encoding='utf-8') as f:
                    for processed_raw_action in processed_raw_actions:
                        f.write(json.dumps(processed_raw_action, ensure_ascii=False) + '\n')
                        processed_raw_actions = []

            # Step environment with chosen action
            obs, reward, terminated, truncated, info = env.step(action)
            if reward > 0:
                success_state = True
                break

        # Report FPS
        print(f"FPS: {run_frame_idx/(time.time()-start_time)}")

        # Pause after success (to render final states more clearly)
        end_pause = 1
        if success_state:
            end_pause = random.randint(5, 20)
        for _ in range(end_pause):
            time.sleep(0.1)
            processed_raw_actions.append({
                "raw_action": "",
                "model_generated": False,
                "response_id": model_response_id if model_response_id >= 0 else None,
            })
            obs, reward, terminated, truncated, info = env.step(noop_action)

        # Save remaining actions
        with open(raw_action_file_path, 'a', encoding='utf-8') as f:
            for processed_raw_action in processed_raw_actions:
                f.write(json.dumps(processed_raw_action, ensure_ascii=False) + '\n')
    finally:
        # Always close the environment to release the Minecraft JVM process,
        # preventing zombie processes from filling the resetting_list slot pool.
        env.close()
        if getattr(args, "vlm_client_mode", "") == "streaming":
            try:
                agent.reset(task_name="")
            except Exception:
                pass

    # Write success/failure metadata
    meta = {"frames": run_frame_idx, "env_pacing": env_pacing, "args": vars(args)}
    if stream_stats is not None:
        meta["stream"] = stream_stats
    if success_state:
        with open(os.path.join(rollout_path, 'success.json'), 'w') as f:
            json.dump({"success": True, **meta}, f, indent=2, ensure_ascii=False)
    else:
        with open(os.path.join(rollout_path, 'loss.json'), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # Clip actions and render video (skippable to speed up eval — success/loss
    # judgement uses success.json/loss.json above, not the MP4).
    if getattr(args, "no_render", False):
        print(f"skipping video render for {rollout_path} (--no_render)")
    else:
        clip_action_info(rollout_path, assert_video_action_num_same=True)
        renderred_frames = render_video(
            rollout_path,
            enable_task_name=True, enable_rawaction=True, enable_thought=False, enable_envaction=True, enable_point_cot=False,
            raw_action_max_length = 5,
        )
        print(f"saving render video in {rollout_path} with {len(renderred_frames)} frames")
        save_render_videos(rollout_path, renderred_frames, fps=args.fps)
    return rollout_path

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_mode', type=str, default="greedy", choices=["greedy", "grounding", "motion", "text_action"],)
    parser.add_argument('--vlm_client_mode', type=str, default="online", choices=["online", "openai", "anthropic", "vllm", "lmdeploy", "hf", "streaming", ],)
    parser.add_argument('--streaming_send_instruction', action="store_true",
                        help="streaming mode: forward the task instruction (goal) to the /step server, "
                             "aligning the instruction with OpenHA. Off by default.")
    parser.add_argument('--instruction_prefix', action="store_true",
                        help="non-streaming modes: wrap the per-task goal as "
                             "'你当前要完成的任务是：{instruction}' (\"The task you need to complete now is: "
                             "{instruction}\") before appending it to the system "
                             "prompt (matches the streaming server's phrasing). Off by default.")
    parser.add_argument('--per_turn_user_text', type=str, default="",
                        help="non-streaming modes: text put on EVERY user turn (history + current), "
                             "e.g. 'current game screen'. Realigns online multi-image history with a "
                             "streaming-trained model. Empty by default (OpenHA-native behavior).")
    parser.add_argument('--native_gui_chat', action="store_true",
                        help="reproduce the training-pipeline chat format exactly: system role + "
                             "'current game screen\\n<image>\\ncurrent mouse position\\n{pos}' per turn, "
                             "no task instruction. Fixes no-op collapse with history>0.")
    parser.add_argument('--native_mouse_pos', type=str, default="[500, 500]",
                        help="native_gui_chat: mouse-position line value (default [500, 500]).")
    parser.add_argument("--system_message_tag", type=str, default="text_action")
    parser.add_argument("--system_message", type=str, default = None)
    parser.add_argument("--model_ips", type=str, default="localhost")
    parser.add_argument("--model_ports", type=str, default="11000")
    parser.add_argument("--base_url", type=str, default=None,
                        help="full OpenAI-compatible endpoint (e.g. https://api.openai.com/v1). "
                             "Overrides --model_ips/--model_ports. Pair with --vlm_client_mode openai "
                             "and --api_key (or OPENAI_API_KEY) for commercial APIs.")
    parser.add_argument('--raw_action_type', type=str,default="text", choices=["reserved", "text", "gui"],)
    parser.add_argument('--gui_reasoning_mode', default="direct",
                        choices=["direct", "optional", "forced"],
                        help="native GUI output contract: direct, optional think, or forced think.")
    parser.add_argument('--gui_allow_missing_think', action="store_true",
                        help="gui_reasoning_mode=optional: also accept an action with NO think "
                             "tags at all. For commercial models that follow the prose prompt but "
                             "omit the empty <think></think> our checkpoints always emit.")
    parser.add_argument('--gui_key_map', default="wasd", choices=["wasd", "arrows", "both"],
                        help="movement binding the native GUI decoder accepts.")
    parser.add_argument('--env_pacing', default="stop", choices=["stop", "streaming"],
                        help="stop = world freezes while the model thinks (default); streaming = "
                             "model wall-clock latency is charged to the game clock, the previous "
                             "plan keeps playing and then the agent idles until the reply lands.")
    parser.add_argument('--stream_tick_ms', type=float, default=50.0,
                        help="streaming: game ms per Minecraft tick used to convert latency to ticks (20 Hz).")
    parser.add_argument('--stream_max_latency_ms', type=float, default=15000.0,
                        help="streaming: cap on charged latency per call (guards against a single "
                             "timed-out request consuming the whole episode). 0 = no cap.")
    parser.add_argument('--stream_min_latency_ms', type=float, default=0.0,
                        help="streaming: floor on charged latency per call so a fast reply cannot "
                             "pre-empt the previous chunk (200 = one full 4-tick window). 0 = off.")
    parser.add_argument('--stream_idle_action', default="noop", choices=["noop", "hold"],
                        help="streaming: what to do on ticks with no pending plan.")
    parser.add_argument('--stream_capture', default="on_arrival", choices=["on_arrival", "after_chunk"],
                        help="streaming: when the next frame is sent. on_arrival = look the instant a "
                             "reply lands and play its plan during the next call (pipelined, default); "
                             "after_chunk = play the whole plan first, then look.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="CraftJarvis/minecraft-openha-qwen2vl-7b-2509")
    parser.add_argument("--sam_path", type=str, default="facebook/sam2-hiera-base-plus")
    parser.add_argument("--grounding_policy_path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--motion_policy_path", type=str, default="CraftJarvis/Minecraft-Motion_policy-2509")
    parser.add_argument("--grounding_inference_interval", type=int, default=4)
    parser.add_argument("--motion_inference_interval", type=int, default = 4)
    parser.add_argument('--action-chunk-len',type=int, default = 1)
    parser.add_argument('--maximum_history_length',type=int, default = 15)
    parser.add_argument('--gui_history_content', choices=['full', 'action_only'], default='full')
    
    parser.add_argument('--LLM_backbone', type=str, default="")
    parser.add_argument('--VLM_backbone', type=str, default="")
    parser.add_argument('--tokenizer_path', type=str, default="")
    
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--record_path", type=str, required=True)
    
    parser.add_argument("--max_steps_num", type=int, default=400)
    parser.add_argument("--num_rollouts", type=int, default=1)
    
    parser.add_argument("--task", type=str, default="kill_entity:sheep") 
    parser.add_argument('--difficulty', type=str,default="zero")
    
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--fps',type=int,default=20)
    parser.add_argument('--no_render', action="store_true",
                        help="skip clip_action_info + MP4 render at end of each rollout to speed up eval")
    parser.add_argument('--render_width', type=int, default=None,
                        help="POV render width sent to the VLM (default 640; set 1280 for 720p)")
    parser.add_argument('--render_height', type=int, default=None,
                        help="POV render height sent to the VLM (default 360; set 720 for 720p)")
    parser.add_argument('--instruction_lang', type=str, default="zh", choices=["en", "zh"],
                        help="task instruction language: 'en' = original English task_description "
                             "(reference model); 'zh' = Chinese conversion (our trained model)")
    
    parser.add_argument("--enforce_format", type=bool, default = False)
    parser.add_argument("--enforce_prefix", type=str, default = "")
    parser.add_argument('--instruction-type',type=str, default='final')
    parser.add_argument('--temperature','-t',type=float, default=0.8)
    parser.add_argument('--top_p',type=float, default=0.99)
    parser.add_argument('--top_k',type=int, default=-1)
    parser.add_argument('--gpu_per_rollout',type=float, default=0.5)
    args = parser.parse_args()
    
    # Multi-rollout mode with Ray
    if args.num_rollouts > 1:
        args.num_rollouts = args.num_rollouts
        num_gpus = torch.cuda.device_count()
        logging.info(f"Available GPUs: {num_gpus}")

        # Initialize Ray with available GPUs
        ray.init(num_gpus=num_gpus, log_to_driver=True)
        # ray.init(address='auto')

        # Launch rollout tasks in parallel
        futures = [ray.remote(num_gpus=args.gpu_per_rollout)(rollout_wrapper).remote(args) for _ in range(args.num_rollouts)]
        print(f"Rollout futures: {len(futures)}")

        # Gather results
        results = ray.get(futures)
        for result in results:
            logging.info(result)

        # Shutdown Ray
        ray.shutdown()
    else:
        # Single rollout mode
        rollout(args)
