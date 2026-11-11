#!/usr/bin/env bash
# Evaluate a self-trained Qwen3.5 CPT checkpoint through a standard vLLM /v1
# endpoint. Defaults to the English direct-action protocol and English OpenHA goal.
# Special CPT checkpoints use the same optional-thinking contract as SFT.  There
# are two tokenizer/template layouts; select one explicitly for thinking runs:
#
#   THINK_TAG_PREFILLED=1  template already appends the opening <think> tag
#   THINK_TAG_PREFILLED=0  model emits the complete <think>...</think> wrapper
#
# The prefilled case must be served with the think-prefill chat template
# (openagents/assets/chat_templates/qwen35_cpt_think_prefill.jinja), keeping the raw
# completion and NOT using a vLLM reasoning parser. The complete-tag case (the
# default protocol) uses serve_qwen3vl.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_KIND=cpt
if [ -z "${REASONING_MODE:-}" ]; then
    case "${THINKING:-}" in
        1) REASONING_MODE=optional ;;
        *) REASONING_MODE=direct ;;
    esac
fi
export REASONING_MODE
export THINK_TAG_PREFILLED=${THINK_TAG_PREFILLED:-0}
exec bash "$HERE/run_eval_qwen3vl_online.sh" "$@"
