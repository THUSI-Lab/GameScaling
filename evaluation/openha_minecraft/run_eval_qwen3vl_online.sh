#!/usr/bin/env bash
# Common STANDARD-vLLM evaluator for our Qwen3.5 CPT/SFT checkpoints.
# Prefer the wrapper:
#   bash run_eval_qwen35_cpt.sh
#
# It uses the training-pipeline native chat format:
#   messages[0] = {role: system, content: <the training system prompt>}
#   first user  = task goal + "current game screen\n<image>"
#   later users =             "current game screen\n<image>"
#                 + assistant "<|action_start|>...<|action_end|>"
# REASONING_MODE + PROMPT_LANG select the output protocol and system prompt:
#   direct   -> no <think> output
#   optional -> SFT and special-CPT contract: model chooses empty or non-empty <think>
#   forced   -> legacy experiment contract: every response must contain non-empty <think>
#
# online mode has no server-side KV session, so --maximum_history_length is the
# ONLY source of history. Default 15 (as run_eval_openha.sh); the model trained on
# 20-frame windows so 15 is in-distribution. Keep serve_qwen3vl.sh MAX_IMAGES >= MAXHIST+1.
#
# Direct usage:
#   MODEL_KIND=cpt REASONING_MODE=optional MODEL_IP=<MODEL_IP> MODEL_PORT=8002 bash run_eval_qwen3vl_online.sh
set -euo pipefail

# MineStudio keeps a diskcache sqlite DB under MINESTUDIO_DIR, and sqlite cannot take
# locks on Ceph/NFS, so this must be a LOCAL filesystem — not the repo dir. Provision the
# engine once per host by copying it out of the checkout.
export MINESTUDIO_DIR=${MINESTUDIO_DIR:-/tmp/minestudio_local}
mkdir -p "$MINESTUDIO_DIR"
_SRC_MS="$(cd "$(dirname "$0")" && pwd)/MineStudio"
if [ ! -d "$MINESTUDIO_DIR/engine" ] && [ -d "$_SRC_MS/engine" ]; then
    echo "[run_eval] provisioning local MineStudio engine -> $MINESTUDIO_DIR (one-time per host)"
    for _sub in engine tmp .cache; do
        [ -d "$_SRC_MS/$_sub" ] && cp -a "$_SRC_MS/$_sub" "$MINESTUDIO_DIR/" || true
    done
fi

cd "$(dirname "$0")"
# shellcheck source=openha_eval_env.sh
source "$PWD/openha_eval_env.sh"
# Force THIS checkout onto sys.path so the editable-installed openagents from an
# older checkout can't shadow it (see openha-editable-install-shadowing memory).
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL_ID=${MODEL_ID:-qwen3.5-9b}         # matches serve_qwen3vl.sh's default --served-model-name
MODEL_IP=${MODEL_IP:-localhost}          # the vllm>=0.11 serve box
MODEL_PORT=${MODEL_PORT:-8002}
# Backend selection:
#   online -> self-hosted vLLM at http://MODEL_IP:MODEL_PORT/v1 (default; vLLM-only
#             sampling extras like top_k / skip_special_tokens / stop_token_ids are sent)
#   openai -> a commercial OpenAI-compatible API at BASE_URL with API_KEY (or
#             $OPENAI_API_KEY). Nothing vLLM-specific is sent; STOP_MODE defaults to
#             none because include_stop_str_in_output is not supported there and a
#             plain "stop" would strip the <|action_end|> marker the parser needs.
#             Same prompt / parser as the CPT run: the model is free to emit an empty
#             <think></think> or real reasoning before the action.
#   Example: VLM_CLIENT_MODE=openai BASE_URL=https://api.openai.com/v1 \
#            API_KEY=sk-... MODEL_ID=gpt-4o MODEL_PATH=gpt-4o bash run_eval_qwen35_cpt.sh
VLM_CLIENT_MODE=${VLM_CLIENT_MODE:-online}
BASE_URL=${BASE_URL:-}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-EMPTY}}
STREAMING_SEND_INSTR_ARG=()
case "$VLM_CLIENT_MODE" in
    online) ;;
    streaming)
        if [ -z "$BASE_URL" ]; then
            BASE_URL="http://$MODEL_IP:$MODEL_PORT"
        fi
        STOP_MODE=${STOP_MODE:-none}
        STREAMING_SEND_INSTR_ARG=(--streaming_send_instruction)
        ;;
    openai)
        if [ -z "$BASE_URL" ]; then
            echo "VLM_CLIENT_MODE=openai requires BASE_URL (e.g. https://api.openai.com/v1)" >&2
            exit 2
        fi
        if [ "$API_KEY" = "EMPTY" ]; then
            echo "VLM_CLIENT_MODE=openai requires API_KEY or OPENAI_API_KEY" >&2
            exit 2
        fi
        # Hand the key over via env (vlm_client falls back to it in openai mode) so it
        # never shows up in `ps` / logs as a --api_key argument.
        export OPENAI_API_KEY="$API_KEY"
        STOP_MODE=${STOP_MODE:-none}
        ;;
    *)
        echo "VLM_CLIENT_MODE must be online|openai|streaming (got '$VLM_CLIENT_MODE')" >&2
        exit 2
        ;;
esac
# model_path is NOT loaded in online mode (weights live on the serve box);
# load_visual_model() only string-matches it to pick the backbone. Keep "qwen3" in
# it so VLM_backbone resolves to "" -> raw frame sent, server sizes via mm-processor.
MODEL_PATH=${MODEL_PATH:-qwen3.5-9b}

# Online-history ablation:
#   full -> re-feed up to MAXHIST previous image/assistant turns; HISTORY_CONTENT
#           below controls whether assistant history is action-only or think+action
#   none -> send no previous turns; the request contains only the current image
#           (plus the system prompt and current task/user text)
HISTORY_MODE=${HISTORY_MODE:-full}
case "$HISTORY_MODE" in
    full)
        MAXHIST=${MAXHIST:-15}
        if [ "$MAXHIST" -le 0 ]; then
            echo "HISTORY_MODE=full requires MAXHIST > 0 (got $MAXHIST)" >&2
            exit 2
        fi
        HISTORY_TAG="history-full-${MAXHIST}"
        ;;
    none)
        MAXHIST=0
        HISTORY_TAG="history-none"
        ;;
    *)
        echo "HISTORY_MODE must be full|none (got '$HISTORY_MODE')" >&2
        exit 2
        ;;
esac

# Retained GUI turns include their image plus either the exact assistant response
# or only its action block. Default to action-only so thinking is not re-fed.
HISTORY_CONTENT=${HISTORY_CONTENT:-action_only}
case "$HISTORY_CONTENT" in
    action_only|full) ;;
    *)
        echo "HISTORY_CONTENT must be action_only|full (got '$HISTORY_CONTENT')" >&2
        exit 2
        ;;
esac

MODEL_KIND=${MODEL_KIND:-}
PROMPT_LANG=${PROMPT_LANG:-en}
REASONING_MODE=${REASONING_MODE:-}
# 1 only for checkpoints whose chat template itself appends the opening
# '<think>\n' to the assistant generation prompt.  Their raw completion starts
# with bare reasoning and contains only the closing </think>.  Keep this 0 for
# models that generate the complete <think>...</think> wrapper themselves.
THINK_TAG_PREFILLED=${THINK_TAG_PREFILLED:-0}
# Backward compatibility for commands written during the short-lived binary
# THINKING=0/1 interface. Both SFT and special-CPT thinking are optional.
if [ -z "$REASONING_MODE" ] && [ -n "${THINKING:-}" ]; then
    case "$THINKING:$MODEL_KIND" in
        0:*) REASONING_MODE=direct ;;
        1:sft) REASONING_MODE=optional ;;
        1:cpt) REASONING_MODE=optional ;;
    esac
fi
case "$REASONING_MODE:$PROMPT_LANG" in
    direct:zh) SYS_TAG=gui_native_direct_zh ;;
    direct:en) SYS_TAG=gui_native_direct_en ;;
    optional:zh) SYS_TAG=gui_native_optional_zh ;;
    optional:en) SYS_TAG=gui_native_optional_en ;;
    forced:zh) SYS_TAG=gui_native_forced_zh ;;
    forced:en) SYS_TAG=gui_native_forced_en ;;
    *)
        echo "REASONING_MODE must be direct|optional|forced and PROMPT_LANG must be zh|en (got '$REASONING_MODE:$PROMPT_LANG')" >&2
        exit 2
        ;;
esac
REASONING_TAG=$REASONING_MODE
# Movement-binding probe. arrows/both swap in the arrow-key system prompt (only
# authored for the optional:en protocol) and tell the decoder which keys move.
#   wasd   -> production (default)
#   arrows -> Up/Down/Left/Right move, W/S/A/D dropped
#   both   -> either binding moves; adoption rate is read off raw_action.jsonl
GUI_KEY_MAP=${GUI_KEY_MAP:-wasd}
case "$GUI_KEY_MAP" in
    wasd) KEYMAP_TAG="" ;;
    arrows|both)
        if [ "$SYS_TAG" != "gui_native_optional_en" ]; then
            echo "GUI_KEY_MAP=$GUI_KEY_MAP requires REASONING_MODE=optional PROMPT_LANG=en (got $SYS_TAG)" >&2
            exit 2
        fi
        SYS_TAG=gui_native_optional_en_arrows
        KEYMAP_TAG="_keymap-${GUI_KEY_MAP}"
        ;;
    *)
        echo "GUI_KEY_MAP must be wasd|arrows|both (got '$GUI_KEY_MAP')" >&2
        exit 2
        ;;
esac
# System-prompt wording. Same action space and think contract either way.
#   default -> gui_native_sft_en, the prose prompt our checkpoints were trained on
#   strict  -> explicit grammar + worked wrong-examples, for untrained commercial
#              models that otherwise repeat the X Y Z triple inside every key group
# Default to strict in openai mode: the training-matched wording measured 0 usable
# key presses there, so it is not the meaningful baseline for those models.
case "$VLM_CLIENT_MODE" in
    openai) PROMPT_VARIANT=${PROMPT_VARIANT:-strict} ;;
    *)      PROMPT_VARIANT=${PROMPT_VARIANT:-default} ;;
esac
case "$PROMPT_VARIANT" in
    default) PROMPT_VARIANT_TAG="" ;;
    strict)
        if [ "$SYS_TAG" != "gui_native_optional_en" ]; then
            echo "PROMPT_VARIANT=strict requires REASONING_MODE=optional PROMPT_LANG=en and GUI_KEY_MAP=wasd (got $SYS_TAG)" >&2
            exit 2
        fi
        SYS_TAG=gui_native_optional_en_strict
        PROMPT_VARIANT_TAG="_prompt-strict"
        ;;
    *)
        echo "PROMPT_VARIANT must be default|strict (got '$PROMPT_VARIANT')" >&2
        exit 2
        ;;
esac
case "$THINK_TAG_PREFILLED" in
    0) THINK_PREFILL_ARG=(); THINK_FORMAT_TAG=think-complete ;;
    1)
        if [ "$REASONING_MODE" = "direct" ]; then
            echo "THINK_TAG_PREFILLED=1 requires REASONING_MODE=optional|forced" >&2
            exit 2
        fi
        THINK_PREFILL_ARG=(--gui_think_prefilled)
        THINK_FORMAT_TAG=think-prefilled
        ;;
    *)
        echo "THINK_TAG_PREFILLED must be 0 or 1 (got '$THINK_TAG_PREFILLED')" >&2
        exit 2
        ;;
esac

# Our checkpoints always emit the <think> wrapper (empty when they choose not to
# reason), so a missing wrapper is a real protocol mismatch and must stay a no_op.
# Commercial models follow the prose prompt and just omit the empty tag, which would
# otherwise score every one of their steps as a no_op — default this on for them.
case "$VLM_CLIENT_MODE" in
    openai) ALLOW_MISSING_THINK=${ALLOW_MISSING_THINK:-1} ;;
    *)      ALLOW_MISSING_THINK=${ALLOW_MISSING_THINK:-0} ;;
esac
case "$ALLOW_MISSING_THINK" in
    0) MISSING_THINK_ARG=() ;;
    1)
        if [ "$REASONING_MODE" != "optional" ]; then
            echo "ALLOW_MISSING_THINK=1 requires REASONING_MODE=optional (got '$REASONING_MODE')" >&2
            exit 2
        fi
        MISSING_THINK_ARG=(--gui_allow_missing_think)
        THINK_FORMAT_TAG="${THINK_FORMAT_TAG}-nothink-ok"
        ;;
    *)
        echo "ALLOW_MISSING_THINK must be 0 or 1 (got '$ALLOW_MISSING_THINK')" >&2
        exit 2
        ;;
esac

# 0.9/0.99 reproduces the sampling our checkpoints were evaluated under. Format
# adherence matters more than diversity for untrained models, so greedy is often the
# better setting there (OmniGameArena runs every commercial model at temperature 0).
TEMPERATURE=${TEMPERATURE:-0.9}
TOP_P=${TOP_P:-0.99}
case "${OMIT_TOP_P:-0}" in
    0) OMIT_TOP_P_ARG=(); TOP_P_TAG="$TOP_P" ;;
    1) OMIT_TOP_P_ARG=(--omit_top_p); TOP_P_TAG="omitted" ;;
    *) echo "OMIT_TOP_P must be 0 or 1 (got '$OMIT_TOP_P')" >&2; exit 2 ;;
esac

# Env pacing (independent of VLM_CLIENT_MODE):
#   stop      -> world freezes while the model thinks; latency is free (default,
#                what every result so far was measured under)
#   streaming -> each call's wall-clock latency is charged to the game clock at
#                STREAM_TICK_MS per tick; the previous 200 ms plan keeps playing,
#                then the agent idles (STREAM_IDLE_ACTION) until the reply lands.
ENV_PACING=${ENV_PACING:-stop}
STREAM_TICK_MS=${STREAM_TICK_MS:-50}
STREAM_MAX_LATENCY_MS=${STREAM_MAX_LATENCY_MS:-15000}
# Floor on charged latency: a reply faster than this still advances the world by
# this much so the previous 4-tick chunk plays out instead of being pre-empted.
STREAM_MIN_LATENCY_MS=${STREAM_MIN_LATENCY_MS:-0}
STREAM_IDLE_ACTION=${STREAM_IDLE_ACTION:-noop}
# on_arrival  -> capture the instant a reply lands; its plan plays during the next call (default)
# after_chunk -> execute the whole reply, then capture the next frame
STREAM_CAPTURE=${STREAM_CAPTURE:-on_arrival}
case "$ENV_PACING" in
    stop) PACING_TAG="" ;;
    streaming)
        PACING_TAG="_pacing-streaming-${STREAM_IDLE_ACTION}-${STREAM_CAPTURE}"
        [ "$STREAM_MIN_LATENCY_MS" != "0" ] && PACING_TAG="${PACING_TAG}-min${STREAM_MIN_LATENCY_MS}ms"
        ;;
    *) echo "ENV_PACING must be stop|streaming (got '$ENV_PACING')" >&2; exit 2 ;;
esac
case "$STREAM_IDLE_ACTION" in
    noop|hold) ;;
    *) echo "STREAM_IDLE_ACTION must be noop|hold (got '$STREAM_IDLE_ACTION')" >&2; exit 2 ;;
esac
case "$STREAM_CAPTURE" in
    on_arrival|after_chunk) ;;
    *) echo "STREAM_CAPTURE must be on_arrival|after_chunk (got '$STREAM_CAPTURE')" >&2; exit 2 ;;
esac

ROLLOUTS=${ROLLOUTS:-5}
MAX_STEPS=${MAX_STEPS:-600}
CATEGORIES=${CATEGORIES:-all}
TASKS=${TASKS:-}
# Default is the 30-task OpenHA slice. The 800-task table is
# CrossAgent/STRL/data_processor/utils/task_list_full.json.
TASK_FILE=${TASK_FILE:-}
PARALLEL=${PARALLEL:-30}
VERBOSE=${VERBOSE:-}
# Keep the OpenHA task goal in the same language as the selected system prompt.
INSTR_LANG=${INSTR_LANG:-$PROMPT_LANG}
# Model-input resolution. Minecraft itself remains native 640x360; resizing happens
# only at the VLM boundary, matching the GUI-game training data path.
RES=${RES:-480p}
case "$RES" in
    720p) MODEL_INPUT_W=${MODEL_INPUT_W:-1280}; MODEL_INPUT_H=${MODEL_INPUT_H:-720} ;;
    # 480p (854x480 = 410k px) sits inside the server's [MIN_PIXELS, MAX_PIXELS] window,
    # so unlike 720p it is NOT downscaled again by the mm-processor.
    480p) MODEL_INPUT_W=${MODEL_INPUT_W:-854};  MODEL_INPUT_H=${MODEL_INPUT_H:-480} ;;
    360p) MODEL_INPUT_W=${MODEL_INPUT_W:-640};  MODEL_INPUT_H=${MODEL_INPUT_H:-360} ;;
    *)
        echo "RES must be 720p, 480p or 360p (got '$RES')" >&2
        exit 2
        ;;
esac
ENGINE_RENDER_W=${ENGINE_RENDER_W:-}
ENGINE_RENDER_H=${ENGINE_RENDER_H:-}
NO_RENDER=${NO_RENDER:-}
TOP_LOGPROBS=${TOP_LOGPROBS:-}
RECORD_SAMPLED_LOGPROBS=${RECORD_SAMPLED_LOGPROBS:-}
# Stop each request after the first native GUI action.  A server-level generation
# override is not reliably retained by every vLLM Chat Completions path, so send
# this explicitly on every request.  Keep the stop token in the returned text;
# GuiActionTokenizer requires the closing <|action_end|> marker.
# Checkpoint-specific stopping: token for legacy CPT, string for SFT exports.
STOP_MODE=${STOP_MODE:-token}
case "$STOP_MODE" in
    token) EXTRA_BODY_JSON=${EXTRA_BODY_JSON:-'{"stop_token_ids":[151670],"include_stop_str_in_output":true}'} ;;
    string) EXTRA_BODY_JSON=${EXTRA_BODY_JSON:-'{"stop":["<|action_end|>"],"include_stop_str_in_output":true}'} ;;
    none) EXTRA_BODY_JSON=${EXTRA_BODY_JSON:-} ;;
    *) echo "STOP_MODE must be token|string|none (got '$STOP_MODE')" >&2; exit 2 ;;
esac
# Mouse/camera mapping ablation. Select one mode instead of manually keeping two
# axis scales in sync:
#   calibrated -> reconstruct the checkpoint's 1280x720 normalized labels, then
#                 apply the VPT/MineStudio 0.15 camera conversion
#   openha     -> original OpenHA 0.15 degrees/unit on both axes
#   direct     -> treat model X/Y as degrees (usually much too aggressive)
#   off        -> discard mouse/camera movement while retaining keyboard actions
# Explicit GUI_YAW_SCALE / GUI_PITCH_AXIS_SCALE values still override the selected
# mode, which is useful for fine-grained sweeps.
GUI_MOUSE_MAPPING=${GUI_MOUSE_MAPPING:-calibrated}
case "$GUI_MOUSE_MAPPING" in
    calibrated) _DEFAULT_YAW_SCALE=0.192; _DEFAULT_PITCH_SCALE=0.108 ;;
    openha)     _DEFAULT_YAW_SCALE=0.15;  _DEFAULT_PITCH_SCALE=0.15 ;;
    direct)     _DEFAULT_YAW_SCALE=1.0;   _DEFAULT_PITCH_SCALE=1.0 ;;
    off)        _DEFAULT_YAW_SCALE=0.0;   _DEFAULT_PITCH_SCALE=0.0 ;;
    *)
        echo "GUI_MOUSE_MAPPING must be calibrated|openha|direct|off (got '$GUI_MOUSE_MAPPING')" >&2
        exit 2
        ;;
esac
GUI_YAW_SCALE=${GUI_YAW_SCALE:-$_DEFAULT_YAW_SCALE}
GUI_PITCH_AXIS_SCALE=${GUI_PITCH_AXIS_SCALE:-$_DEFAULT_PITCH_SCALE}
GUI_RESAMPLE_SUBSTEPS=${GUI_RESAMPLE_SUBSTEPS:-1}
GUI_DOWN_PITCH_GAIN=${GUI_DOWN_PITCH_GAIN:-1.0}
# GUI-state-aware ablation: when a GUI is open, decode X/Y with the 640x360 cursor
# space scales (0.096 / 0.054) instead of the free-camera ones. Off by default.
GUI_STATE_AWARE_SCALE=${GUI_STATE_AWARE_SCALE:-0}
export GUI_MOUSE_MAPPING GUI_YAW_SCALE GUI_PITCH_AXIS_SCALE GUI_RESAMPLE_SUBSTEPS GUI_DOWN_PITCH_GAIN GUI_STATE_AWARE_SCALE
RECORD_PATH=${RECORD_PATH:-"$OPENHA_EVAL_OUTPUT/eval_runs_qwen35_${MODEL_KIND}_${PROMPT_LANG}_${REASONING_TAG}_vllm_${HISTORY_TAG}${KEYMAP_TAG}${PROMPT_VARIANT_TAG}${PACING_TAG}"}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-2000}
export MINESTUDIO_MAX_WAIT=${MINESTUDIO_MAX_WAIT:-120}

echo "[run_eval] GUI mapping: mode=$GUI_MOUSE_MAPPING yaw=$GUI_YAW_SCALE pitch=$GUI_PITCH_AXIS_SCALE "\
"resample_30hz_to_20hz=$GUI_RESAMPLE_SUBSTEPS down_pitch_gain=$GUI_DOWN_PITCH_GAIN "\
"gui_state_aware=$GUI_STATE_AWARE_SCALE gui_open_yaw=${GUI_OPEN_YAW_SCALE:-0.096} gui_open_pitch=${GUI_OPEN_PITCH_AXIS_SCALE:-0.054}"
echo "[run_eval] engine POV: ${ENGINE_RENDER_W:-640}x${ENGINE_RENDER_H:-360}; model input: $RES (${MODEL_INPUT_W}x${MODEL_INPUT_H})"
echo "[run_eval] history: mode=$HISTORY_MODE max=$MAXHIST assistant_content=$HISTORY_CONTENT"
echo "[run_eval] protocol: system_tag=$SYS_TAG reasoning=$REASONING_MODE key_map=$GUI_KEY_MAP prompt_variant=$PROMPT_VARIANT"
echo "[run_eval] backend: mode=$VLM_CLIENT_MODE endpoint=${BASE_URL:-http://$MODEL_IP:$MODEL_PORT/v1} model_id=$MODEL_ID stop_mode=$STOP_MODE"
echo "[run_eval] think: reasoning=$REASONING_MODE prefilled=$THINK_TAG_PREFILLED allow_missing=$ALLOW_MISSING_THINK"
echo "[run_eval] sampling: temperature=$TEMPERATURE top_p=$TOP_P_TAG"
echo "[run_eval] pacing: env_pacing=$ENV_PACING tick_ms=$STREAM_TICK_MS max_latency_ms=$STREAM_MAX_LATENCY_MS min_latency_ms=$STREAM_MIN_LATENCY_MS idle=$STREAM_IDLE_ACTION capture=$STREAM_CAPTURE"

python3 scripts/run_eval.py \
    --rollout_timeout "$ROLLOUT_TIMEOUT" \
    ${VERBOSE:+--verbose} \
    ${NO_RENDER:+--no_render} \
    ${TOP_LOGPROBS:+--top_logprobs "$TOP_LOGPROBS"} \
    ${RECORD_SAMPLED_LOGPROBS:+--record_sampled_logprobs} \
    --extra_body_json "$EXTRA_BODY_JSON" \
    ${ENGINE_RENDER_W:+--render_width "$ENGINE_RENDER_W"} \
    ${ENGINE_RENDER_H:+--render_height "$ENGINE_RENDER_H"} \
    --model_input_width "$MODEL_INPUT_W" \
    --model_input_height "$MODEL_INPUT_H" \
    --instruction_lang "$INSTR_LANG" \
    --model_id "$MODEL_ID" \
    --model_path "$MODEL_PATH" \
    --model_ips "$MODEL_IP" --model_ports "$MODEL_PORT" \
    ${BASE_URL:+--base_url "$BASE_URL"} \
    --vlm_client_mode "$VLM_CLIENT_MODE" \
    "${STREAMING_SEND_INSTR_ARG[@]}" \
    --output_mode text_action \
    --raw_action_type gui \
    --gui_reasoning_mode "$REASONING_MODE" \
    --gui_key_map "$GUI_KEY_MAP" \
    --env_pacing "$ENV_PACING" \
    --stream_tick_ms "$STREAM_TICK_MS" \
    --stream_max_latency_ms "$STREAM_MAX_LATENCY_MS" \
    --stream_min_latency_ms "$STREAM_MIN_LATENCY_MS" \
    --stream_idle_action "$STREAM_IDLE_ACTION" \
    --stream_capture "$STREAM_CAPTURE" \
    "${THINK_PREFILL_ARG[@]}" \
    "${MISSING_THINK_ARG[@]}" \
    --native_gui_chat \
    --native_mouse_pos "" \
    --system_message_tag "$SYS_TAG" \
    --instruction_prefix \
    --maximum_history_length "$MAXHIST" \
    --gui_history_content "$HISTORY_CONTENT" \
    --parallel_tasks "$PARALLEL" \
    --rollouts_per_task "$ROLLOUTS" \
    --max_steps_num "$MAX_STEPS" \
    --categories "$CATEGORIES" \
    ${TASK_FILE:+--task_file "$TASK_FILE"} \
    ${TASKS:+--tasks "$TASKS"} \
    "${OMIT_TOP_P_ARG[@]}" \
    --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --record_path "$RECORD_PATH"
