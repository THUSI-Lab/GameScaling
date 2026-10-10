#!/usr/bin/env bash
# Serve a GameScaling checkpoint with vLLM's OpenAI-compatible server.
#   bash scripts/serve_vllm.sh /path/to/checkpoint [port] [tensor_parallel]
# The clients in this repository expect --served-model-name gamescaling.
set -euo pipefail
CKPT="${1:?usage: serve_vllm.sh CKPT [PORT] [TP]}"
PORT="${2:-8000}"
TP="${3:-1}"

# 32k context holds the system prompt + 5 history frames + the reasoning budget;
# 8 images = 5 history frames + the current one, with headroom.
exec vllm serve "$CKPT" \
  --served-model-name gamescaling \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 8}' \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --trust-remote-code
