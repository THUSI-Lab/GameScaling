#!/usr/bin/env bash
# Serve Qwen3.5 (official model or a merged training-pipeline checkpoint) with a
# STANDARD vLLM OpenAI-compatible server — the same shape as
# run_openha-qwen2vl-7b.sh, but for Qwen3.5.
#
# Why this exists: the model is normally deployed via inference-server (the
# nano-vllm "streaming" /step engine). This script instead exposes a plain
# `vllm serve` /v1 endpoint so OpenHA can eval it in `online` mode — a simpler
# path, and a way to isolate whether
# poor SR comes from the model or from the custom streaming engine.
#
# IMPORTANT: Qwen3.5 needs a current vLLM release. Upgrade vLLM before running
# this script; old environments such as vLLM 0.8.5 cannot load Qwen3.5.
# The eval then points --model_ips/--model_ports here.
#
# Usage:
#   bash serve_qwen3vl.sh  # defaults to Qwen/Qwen3.5-9B
#   MODEL=/abs/path/to/merged-qwen3.5-hf-ckpt bash serve_qwen3vl.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 DP=1 PORT=8002 MODEL=/... bash serve_qwen3vl.sh
set -euo pipefail
# vLLM's prebuilt kernels embed PTX newer than some hosts' driver; without a matching
# libcuda the ViT attention kernel dies with cudaErrorUnsupportedPtxVersion during the
# startup profile run. Which forward-compat package a host ships differs (13.0 on some
# boxes, 12.9 on others), and a hardcoded path that is absent gets silently ignored by
# the loader, so probe for the newest one and use it only when it beats the host driver.
_drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]')
for _lib in $(printf '%s\n' /usr/local/cuda-*/compat/libcuda.so.*.* | sort -Vr); do
    [ -e "$_lib" ] || continue
    _cver=${_lib##*/libcuda.so.}
    if [ -z "$_drv" ] || [ "$(printf '%s\n%s\n' "$_drv" "$_cver" | sort -Vr | head -n1)" = "$_cver" ]; then
        export LD_LIBRARY_PATH="$(dirname "$_lib")":${LD_LIBRARY_PATH:-}
        echo "CUDA forward-compat: libcuda $_cver from $(dirname "$_lib") (host driver ${_drv:-unknown})"
    fi
    break
done
# Merged HF checkpoint (config.json + *.safetensors + tokenizer WITH the added
# <|action_start|>/<|action_end|> tokens), or an official Hugging Face model ID.
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
# served-model-name is what the eval passes as --model_id. Keep "qwen3.5" in it,
# and do NOT put "qwen2_vl" in it: OpenHA's vlm_client only client-side-resizes
# frames when the model name contains "qwen2_vl"; for Qwen3.5 we want the raw
# frame sent and image sizing handled server-side by --mm-processor-kwargs below.
SERVED_NAME=${SERVED_NAME:-qwen3.5-9b}
PORT=${PORT:-8002}
TP=${TP:-1}
DP=${DP:-8}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
# Max images per request. online eval re-feeds up to maximum_history_length+1
# frames; with MAXHIST=15 that is 16. Bump if you raise MAXHIST.
MAX_IMAGES=${MAX_IMAGES:-16}
# Match the training / streaming-server pixel budget (256*32*32 .. 576*32*32)
# so image tokenization lines up with how the model was trained.
MIN_PIXELS=${MIN_PIXELS:-262144}
MAX_PIXELS=${MAX_PIXELS:-589824}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
# CPU-side image preprocessing (HF Qwen2VL image processor) runs in the API server
# process; vLLM pins torch to 1 thread unless OMP_NUM_THREADS is set. Measured on
# H200: 4x720p frames 222ms @1 thread -> 113ms @8 threads (no gain beyond 8).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# Multimodal processor cache. Frames re-fed as history hit the cache and skip the
# HF preprocessing (sliding 3-of-4 hits: ~250ms -> ~165ms on 2B). 0 disables.
# With DP>1 (one API server per replica) use MM_CACHE_TYPE=shm so all API servers
# share one cache; "lru" is per-process.
MM_CACHE_GB=${MM_CACHE_GB:-4}
MM_CACHE_TYPE=${MM_CACHE_TYPE:-lru}
MM_CACHE_ARGS=(--mm-processor-cache-gb "$MM_CACHE_GB")
if [ "$MM_CACHE_GB" != "0" ]; then
    MM_CACHE_ARGS+=(--mm-processor-cache-type "$MM_CACHE_TYPE")
fi
# Optional chat-template override (path to a .jinja). Used to serve a checkpoint
# under a template other than the one shipped in its HF dir, e.g.
# openagents/assets/chat_templates/qwen35_cpt_think_prefill.jinja puts
# "<think>\n" in the generation prompt so the model always starts inside a think
# block (pair the eval with THINK_TAG_PREFILLED=1). Unset -> checkpoint's own template.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-}
CHAT_TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE" ]; then
    [ -f "$CHAT_TEMPLATE" ] || { echo "CHAT_TEMPLATE not found: $CHAT_TEMPLATE" >&2; exit 2; }
    CHAT_TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
fi
# Optional weight dtype override (vLLM --dtype). Unset -> vLLM "auto", i.e. the dtype
# named in config.json. Some exports carry the trainer's fp32 config next to bf16
# safetensors; DTYPE=bfloat16 keeps those from being served in fp32.
DTYPE=${DTYPE:-}
DTYPE_ARGS=()
if [ -n "$DTYPE" ]; then
    DTYPE_ARGS=(--dtype "$DTYPE")
fi

# DEBUG=1 -> verbose vLLM + NCCL logging so the real worker-process exception
# (OOM / NCCL handshake / unsupported arch) surfaces instead of the generic
# "WorkerProc initialization failed" message.
if [ "${DEBUG:-0}" = "1" ]; then
    export VLLM_LOGGING_LEVEL=DEBUG
    export NCCL_DEBUG=INFO
fi

echo "Serving $MODEL as '$SERVED_NAME' on port $PORT (TP=$TP, DP=$DP, GPU_MEM_UTIL=$GPU_MEM_UTIL)"
echo "  images/req<=$MAX_IMAGES  pixels=[$MIN_PIXELS,$MAX_PIXELS]  max_model_len=$MAX_MODEL_LEN max_num_seqs=${MAX_NUM_SEQS:-16}"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS  mm_cache=${MM_CACHE_GB}GB/${MM_CACHE_TYPE}  chat_template=${CHAT_TEMPLATE:-<checkpoint>}"
# Ignore sampling defaults embedded by a training checkpoint. The online evaluator
# supplies its own temperature/top_p, so those must be the effective request values.
exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --generation-config vllm \
    --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}" \
    --mm-processor-kwargs "{\"min_pixels\": $MIN_PIXELS, \"max_pixels\": $MAX_PIXELS}" \
    --mm-encoder-tp-mode data \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    "${MM_CACHE_ARGS[@]}" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${DTYPE_ARGS[@]}" \
    --max-num-seqs "${MAX_NUM_SEQS:-16}" \
    --max-model-len "$MAX_MODEL_LEN"
