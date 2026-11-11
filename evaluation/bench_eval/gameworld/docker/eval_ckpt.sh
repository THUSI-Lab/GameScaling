#!/usr/bin/env bash
# ============================================================================
# Per-checkpoint GameWorld eval (root pod, cu129-root vLLM image).
#
# GameWorld = browser games (Playwright + Chromium, CPU). No UE5/Vulkan.
# The checkpoint is served by vLLM (GPU); the harness drives headless Chromium
# and talks to vLLM over 127.0.0.1:8088..8091 (per the SFT catalog endpoints).
#
# env CKPT   (REQUIRED) checkpoint dir under $HF_EXPORT (or $WS/hf_export)
#                       (e.g. gui_game_cpt_step5160).
# env SHARDS (optional) space-separated shards to run. default "g1 g2 g3 g4".
# env MAXP   (optional) run_suite --max-parallel per shard. default 5.
# env DTYPE  (optional) vLLM dtype. default bfloat16 (weights are bf16 though
#                       config.json declares float32).
# env MAXLEN (optional) vLLM --max-model-len. default 16384.
#
# The cu129-root image already has torch/vLLM/python; we runtime-install the
# Playwright/Chromium + GameWorld python deps (no dedicated image needed).
# ============================================================================
set -uo pipefail

# BENCH_WORKSPACE_BEGIN -- filesystem checks only, before any serving/game code.
_bench_hint="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd -P)"
WS=""
for cand in "${WS_OVERRIDE:-}" "${BENCH_CODE_HINT:-}" "$_bench_hint"; do
  [ -n "$cand" ] || continue
  [ "$(cat "$cand/.bench-workspace-id" 2>/dev/null)" = "bench-workspace-v1" ] || continue
  [ -f "$cand/evaluation/gameworld/run_suite.py" ] || continue
  _bench_physical="$(readlink -f "$cand")"
  [ "$(readlink -f "$cand/evaluation/gameworld")" = "$_bench_physical/evaluation/gameworld" ] || continue
  WS="$_bench_physical"; break
done
[ -n "$WS" ] || { echo '[BENCH][FATAL] own code workspace not mounted; refusing another workspace' >&2; exit 20; }
export WS_OVERRIDE="$WS" BENCH_CODE_HINT="$WS"
# BENCH_WORKSPACE_END

CKPT="${CKPT:?set CKPT to a hf_export checkpoint dir name}"
SHARDS="${SHARDS:-g1 g2 g3 g4}"
MAXP="${MAXP:-5}"
DTYPE="${DTYPE:-bfloat16}"
MAXLEN="${MAXLEN:-16384}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
CLOCK_MODE="${CLOCK_MODE:-}"
FINAL_RT_STRICT="${FINAL_RT_STRICT:-0}"
echo "[EVAL] host=$(hostname) uid=$(id -u) CKPT=$CKPT shards='$SHARDS' maxp=$MAXP dtype=$DTYPE tp=$VLLM_TP clock=${CLOCK_MODE:-catalog} date=$(date)"
nvidia-smi -L 2>&1 | head -8

# 0. Locate the project root (mount path varies by platform).
# Workspace already selected by the identity-checked prologue.

GW="$WS/evaluation/gameworld"
# MODEL_PATH overrides the default hf_export/<CKPT> location. CKPT is then just
# a label for RESULTS_ROOT/<CKPT>. Absolute paths are accepted; relative paths
# resolve under WS.
if [ -n "${MODEL_PATH:-}" ]; then
  if [[ "$MODEL_PATH" = /* ]]; then
    MODEL_DIR="$MODEL_PATH"
  else
    MODEL_DIR="$WS/$MODEL_PATH"
  fi
elif [ -n "${MODEL_PATH_ROOT:-}" ]; then
  MODEL_DIR="${MODEL_PATH_ROOT%/}/$CKPT"
else
  MODEL_DIR=""
  for root in "${HF_EXPORT:-}" "$WS/hf_export"; do
    if [ -f "$root/$CKPT/config.json" ]; then MODEL_DIR="$root/$CKPT"; break; fi
  done
  [ -n "$MODEL_DIR" ] || { echo "[EVAL][FATAL] HF export not found for $CKPT"; exit 21; }
fi

# Optional preparation for raw FSDP checkpoints. The merge runs
# once per unique target directory and leaves an explicit completion marker so
# an interrupted save is never mistaken for a loadable model.
if [ -n "${FSDP_MERGE_SOURCE:-}" ]; then
  echo '[EVAL][FATAL] bench accepts exported HF weights; run checkpoint export separately.' >&2
  exit 21
fi

echo "[EVAL] WS=$WS  GW=$GW  MODEL_DIR=$MODEL_DIR"
[ -f "$MODEL_DIR/config.json" ] || { echo "[EVAL][FATAL] no HF model at $MODEL_DIR"; exit 21; }

# Qwen3-VL and Qwen3.5 use different action token IDs. Resolve from the actual
# export; never stop a 35B generation on the old model's token 151670.
_action_end_id=$(python3 - "$MODEL_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = root / 'model.safetensors.index.json'
if index.is_file():
    files = set(json.loads(index.read_text())['weight_map'].values())
    if not files or any(not (root / name).is_file() for name in files):
        raise SystemExit('Incomplete indexed HF export')
elif not (root / 'model.safetensors').is_file():
    raise SystemExit('No HF weights')
tokenizer = json.loads((root / 'tokenizer.json').read_text())
ids = [t['id'] for t in tokenizer.get('added_tokens', []) if t['content'] == '<|action_end|>']
if len(ids) > 1:
    raise SystemExit('Ambiguous action_end token')
print(ids[0] if ids else '')
PY
) || { echo "[EVAL][FATAL] HF/tokenizer preflight failed"; exit 21; }
if [ -n "$_action_end_id" ]; then
  if [ -n "${SFT_ACTION_END_TOKEN_ID:-}" ] && [ "$SFT_ACTION_END_TOKEN_ID" != "$_action_end_id" ]; then
    echo "[EVAL][FATAL] configured action_end token disagrees with tokenizer"; exit 21
  fi
  export SFT_ACTION_END_TOKEN_ID="$_action_end_id"
  echo "[EVAL] SFT_ACTION_END_TOKEN_ID=$SFT_ACTION_END_TOKEN_ID"
elif [ -n "${SFT_ACTION_END_TOKEN_ID:-}" ]; then
  echo "[EVAL][FATAL] explicit action_end token is absent from tokenizer"; exit 21
else
  echo "[EVAL] no added action_end token; keeping the selected agent stop config"
fi
export PYTHONDONTWRITEBYTECODE=1

cd "$GW" || { echo "[EVAL][FATAL] cannot cd $GW"; exit 22; }

# 1. Runtime-provision the harness deps into the system python (root).
if ! python3 -c "import colorlog, playwright" >/dev/null 2>&1; then
  echo "[EVAL] installing GameWorld python deps ..."
  pip install --no-cache-dir -r requirements.txt 2>&1 | tail -8 || { echo "[EVAL][FATAL] pip install failed"; exit 23; }
else
  echo "[EVAL] GameWorld python deps already installed"
fi
if [ ! -x /root/.cache/ms-playwright/chromium-1124/chrome-linux/chrome ]; then
  echo "[EVAL] installing Chromium (+ system deps) for Playwright ..."
  python -m playwright install --with-deps chromium 2>&1 | tail -8 || { echo "[EVAL][FATAL] playwright install failed"; exit 24; }
else
  echo "[EVAL] Playwright Chromium already installed"
fi
python -m playwright --version 2>&1 | sed 's/^/[EVAL] /' || true
python3 -c 'import vllm, transformers; print("[EVAL] vllm=" + vllm.__version__ + " transformers=" + transformers.__version__)' || exit 23

# Prompt / decode knobs: export so run_suite children (SFT agent) inherit them.
if [ -n "${SFT_MERGED_PROMPT:-}" ]; then
  export SFT_MERGED_PROMPT
  echo "[EVAL] SFT_MERGED_PROMPT=$SFT_MERGED_PROMPT (merged prompt ON)"
else
  echo "[EVAL] using merged game/task prompt by default (SFT_LEGACY_PROMPT selects legacy)"
fi
if [ -n "${SFT_THINK:-}" ]; then
  export SFT_THINK
  case "${SFT_THINK}" in
    closed|close|empty|2) echo "[EVAL] SFT_THINK=$SFT_THINK (prefill closed empty <think></think>)" ;;
    *) echo "[EVAL] SFT_THINK=$SFT_THINK (prefill open <think>)" ;;
  esac
fi
if [ -n "${SFT_TEMPERATURE:-}" ]; then
  export SFT_TEMPERATURE
  echo "[EVAL] SFT_TEMPERATURE=$SFT_TEMPERATURE (overrides catalog temperature)"
fi
if [ -n "${SFT_KEY_TIMELINE:-}" ]; then export SFT_KEY_TIMELINE; echo "[EVAL] SFT_KEY_TIMELINE=$SFT_KEY_TIMELINE"; fi
if [ -n "${SFT_WASD_REMAP:-}" ]; then export SFT_WASD_REMAP; echo "[EVAL] SFT_WASD_REMAP=$SFT_WASD_REMAP"; fi
if [ -n "${GW_FROZEN_CAPTURE:-}" ]; then export GW_FROZEN_CAPTURE; echo "[EVAL] GW_FROZEN_CAPTURE=$GW_FROZEN_CAPTURE"; fi
if [ -n "${SFT_STOP:-}" ]; then export SFT_STOP; echo "[EVAL] SFT_STOP=$SFT_STOP"; fi
if [ -n "${SFT_STOP_TOKEN_IDS:-}" ]; then export SFT_STOP_TOKEN_IDS; echo "[EVAL] SFT_STOP_TOKEN_IDS=$SFT_STOP_TOKEN_IDS"; fi
if [ -n "${SFT_MEMORY_ROUNDS:-}" ]; then export SFT_MEMORY_ROUNDS; echo "[EVAL] SFT_MEMORY_ROUNDS=$SFT_MEMORY_ROUNDS (overrides catalog memory_rounds)"; fi
if [ -n "${SFT_RESIZE_HEIGHT:-}" ]; then export SFT_RESIZE_HEIGHT; echo "[EVAL] SFT_RESIZE_HEIGHT=$SFT_RESIZE_HEIGHT"; fi
if [ -n "${SFT_MEMORY_KEEP_THINK:-}" ]; then export SFT_MEMORY_KEEP_THINK; echo "[EVAL] SFT_MEMORY_KEEP_THINK=$SFT_MEMORY_KEEP_THINK (keep <think> in multi-turn memory)"; fi
if [ -n "${RUN_PRESET:-}" ]; then export RUN_PRESET; echo "[EVAL] RUN_PRESET=$RUN_PRESET (single-game, skip shards)"; fi
# Default: do NOT skip special tokens (see qwen_3_vl_sft.py). Export so children
# inherit; override with SFT_SKIP_SPECIAL_TOKENS=1 to strip markers on decode.
export SFT_SKIP_SPECIAL_TOKENS="${SFT_SKIP_SPECIAL_TOKENS:-0}"
echo "[EVAL] SFT_SKIP_SPECIAL_TOKENS=$SFT_SKIP_SPECIAL_TOKENS"

# Results dir on AFS (per checkpoint). RESULTS_ROOT lets the merged-prompt run
# write to a separate tree (e.g. results/merged) so it doesn't overwrite the
# baseline. Defined early so the cleanup trap can snapshot vLLM logs here.
RESULTS="${RESULTS_ROOT:-results/eval}/$CKPT"
[[ "$RESULTS" = /* ]] || RESULTS="$GW/$RESULTS"
mkdir -p "$RESULTS"

# 2. Start one vLLM per catalog port on the checkpoint. served-model-name MUST
#    be /model to match catalog `model: "/model"`. By default serve all four
#    ports (8088..8091) so every shard finds its endpoint regardless of the
#    g->port mapping in the catalog yamls. VLLM_PORTS can narrow this (each
#    listed port maps to GPU 0,1,2,... by position) e.g. for a 1-GPU smoke.
VLLM_PORTS="${VLLM_PORTS:-8088 8089 8090 8091}"
read -ra _VLLM_PORT_ARRAY <<< "$VLLM_PORTS"
if [ -n "${VLLM_GPU_GROUPS:-}" ]; then
  read -ra _VLLM_GPU_GROUP_ARRAY <<< "$VLLM_GPU_GROUPS"
else
  _VLLM_GPU_GROUP_ARRAY=()
  for ((i=0; i<${#_VLLM_PORT_ARRAY[@]}; i++)); do
    _VLLM_GPU_GROUP_ARRAY+=("$i")
  done
fi
[ "${#_VLLM_PORT_ARRAY[@]}" -eq "${#_VLLM_GPU_GROUP_ARRAY[@]}" ] || {
  echo "[EVAL][FATAL] VLLM_PORTS count must equal VLLM_GPU_GROUPS count"; exit 25;
}
if [ "$FINAL_RT_STRICT" = 1 ]; then
  [ "$CLOCK_MODE" = "final-rt" ] || {
    echo "[EVAL][FATAL] FINAL_RT_STRICT=1 requires CLOCK_MODE=final-rt"; exit 26;
  }
  [ "$VLLM_MAX_NUM_SEQS" = 1 ] || {
    echo "[EVAL][FATAL] strict final-rt requires VLLM_MAX_NUM_SEQS=1"; exit 27;
  }
  [ "$MAXP" = 1 ] || {
    echo "[EVAL][FATAL] strict final-rt requires MAXP=1 per endpoint lane"; exit 28;
  }
  [ -n "${EXTERNAL_STEP_CONFIG:-}" ] && _VLLM_GPU_GROUP_ARRAY_CHECK=() || _VLLM_GPU_GROUP_ARRAY_CHECK=("${_VLLM_GPU_GROUP_ARRAY[@]}")
  for group in "${_VLLM_GPU_GROUP_ARRAY_CHECK[@]}"; do
    IFS=',' read -ra _group_gpus <<< "$group"
    [ "${#_group_gpus[@]}" -eq "$VLLM_TP" ] || {
      echo "[EVAL][FATAL] GPU group '$group' has ${#_group_gpus[@]} GPUs, expected TP=$VLLM_TP"; exit 29;
    }
  done
  if [ -n "${EXPECTED_GPUS:-}" ] && [ -z "${EXTERNAL_STEP_CONFIG:-}" ]; then
    _visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
    [ "$_visible_gpus" -eq "$EXPECTED_GPUS" ] || {
      echo "[EVAL][FATAL] visible GPUs=$_visible_gpus expected=$EXPECTED_GPUS"; exit 29;
    }
  fi
fi
# CHAT_TEMPLATE_NAME (optional): serve with a specific chat template FILE from the
# model dir instead of the model's built-in one (e.g. chat_template.jinja.orig to
# use the STOCK Qwen3.5 template = open <think> at gen + no empty-think in history).
CHAT_TPL_ARG=""
if [ -n "${CHAT_TEMPLATE_NAME:-}" ]; then
  CT="$MODEL_DIR/$CHAT_TEMPLATE_NAME"
  [ -f "$CT" ] && { CHAT_TPL_ARG="--chat-template $CT"; echo "[EVAL] chat-template override: $CT"; } || echo "[EVAL][WARN] CHAT_TEMPLATE_NAME set but $CT missing"
fi
VLLM_PIDS=()
if [ -n "${EXTERNAL_STEP_CONFIG:-}" ]; then
  # An owned streaming /step service replaces the local vLLM. Each lane keeps its
  # logical 127.0.0.1:<port> URL; rt_external maps port 8089 to the second engine
  # (joint two-player tasks) and every other lane to the first.
  export BENCH_EXTERNAL_RUNTIME=1 BENCH_EXTERNAL_CONFIG="$WS/$EXTERNAL_STEP_CONFIG" \
         BENCH_EXTERNAL_MODEL="${EXTERNAL_STEP_MODEL:?}" BENCH_EXTERNAL_TIMING_LOG="$RESULTS/api_timing.jsonl"
  export PYTHONPATH="$WS/rt_external/bootstrap:$WS:${PYTHONPATH:-}"
  echo "[EVAL] external /step service $BENCH_EXTERNAL_MODEL; no local vLLM"
  # Warmup is not scored; the shared gateway occasionally answers with a
  # non-JSON page, so allow three attempts before giving up.
  _wok=0; for _wt in 1 2 3; do
    python3 -m rt_external.step_warmup "$BENCH_EXTERNAL_CONFIG" "$BENCH_EXTERNAL_MODEL" "$RESULTS/warmup_timing.jsonl" && { _wok=1; break; }
    sleep 20
  done
  [ "$_wok" = 1 ] || { echo "[EVAL][FATAL] external /step warmup failed"; exit 31; }
  _VLLM_PORT_ARRAY_LAUNCH=()
else
  _VLLM_PORT_ARRAY_LAUNCH=("${_VLLM_PORT_ARRAY[@]}")
fi
MAX_SEQS_ARG=()
if [ -n "$VLLM_MAX_NUM_SEQS" ]; then
  MAX_SEQS_ARG=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
for ((i=0; i<${#_VLLM_PORT_ARRAY_LAUNCH[@]}; i++)); do
  port="${_VLLM_PORT_ARRAY_LAUNCH[$i]}"
  gpu_group="${_VLLM_GPU_GROUP_ARRAY[$i]}"
  echo "[EVAL] launching vLLM port=$port gpu_group=$gpu_group tp=$VLLM_TP max_num_seqs=${VLLM_MAX_NUM_SEQS:-default}"
  CUDA_VISIBLE_DEVICES="$gpu_group" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name /model \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$VLLM_TP" --dtype "$DTYPE" --trust-remote-code \
    --max-model-len "$MAXLEN" --limit-mm-per-prompt '{"image": '"${IMG_LIMIT:-8}"'}' \
    --gpu-memory-utilization 0.85 "${MAX_SEQS_ARG[@]}" $CHAT_TPL_ARG \
    > "/tmp/vllm_${port}.log" 2>&1 &
  VLLM_PIDS+=($!)
  # Stagger launches: 4 simultaneous inits can race on the auto-picked internal
  # torch.distributed port (EADDRINUSE). A short gap lets each bind before the
  # next selects its port.
  sleep 12
done

cleanup() {
  echo "[EVAL] snapshot vLLM logs -> $RESULTS ; killing vLLM ${VLLM_PIDS[*]}"
  cp -f /tmp/vllm_*.log "$RESULTS/" 2>/dev/null || true
  kill "${VLLM_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# 3. Wait for readiness on every port (up to ~20 min: vLLM load + graph capture).
for port in "${_VLLM_PORT_ARRAY_LAUNCH[@]}"; do
  ok=0
  for i in $(seq 1 120); do
    if [ $((i % 3)) -eq 1 ]; then
      cp -f /tmp/vllm_*.log "$RESULTS/" 2>/dev/null || true
    fi
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "[EVAL] vLLM $port READY (after ${i}0s)"; ok=1; break
    fi
    for pid in "${VLLM_PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "[EVAL][FATAL] a vLLM process exited before readiness"
        tail -40 /tmp/vllm_*.log
        exit 30
      fi
    done
    sleep 10
  done
  if [ "$ok" != 1 ]; then
    echo "[EVAL][FATAL] vLLM $port did not become ready; tail log:"; tail -40 "/tmp/vllm_${port}.log"; exit 30
  fi
done

if [ "${VLLM_WARMUP:-0}" = 1 ] && [ -z "${EXTERNAL_STEP_CONFIG:-}" ]; then
  for port in "${_VLLM_PORT_ARRAY[@]}"; do
    python3 docker/warmup_vllm_gameworld.py \
      --endpoint "http://127.0.0.1:$port" \
      --width "${SFT_RESIZE_WIDTH:-853}" \
      --height "${SFT_RESIZE_HEIGHT:-480}" \
      --history "${SFT_MEMORY_ROUNDS:-5}" || {
        echo "[EVAL][FATAL] vLLM shape warmup failed on port $port"; exit 31;
      }
  done
fi

# 4. Run either a single preset (RUN_PRESET) or the requested shards.
RC=0
CLOCK_ARGS=()
if [ -n "$CLOCK_MODE" ]; then CLOCK_ARGS=(--clock "$CLOCK_MODE"); fi
if [ -n "${RETRY_SUITE:-}" ]; then
  _retry_parallel="${RETRY_EVAL_PARALLEL:-1}"
  echo "[EVAL] retry suite=$RETRY_SUITE parallel=$_retry_parallel"
  python run_suite.py --suite "$RETRY_SUITE" \
    --max-parallel "$_retry_parallel" --port 8101 --results-dir "$RESULTS" \
    "${CLOCK_ARGS[@]}" > "$RESULTS/run_retry.log" 2>&1 || RC=$?
elif [ -n "${FINAL_RT_SMOKE_SUITE:-}" ]; then
  _smoke_parallel="${FINAL_RT_EVAL_PARALLEL:-${#_VLLM_PORT_ARRAY[@]}}"
  [ "$_smoke_parallel" -le "${#_VLLM_PORT_ARRAY[@]}" ] || {
    echo "[EVAL][FATAL] smoke parallel=$_smoke_parallel exceeds replicas=${#_VLLM_PORT_ARRAY[@]}"; exit 32;
  }
  echo "[EVAL] final-rt smoke suite=$FINAL_RT_SMOKE_SUITE parallel=$_smoke_parallel"
  python run_suite.py --suite "$FINAL_RT_SMOKE_SUITE" \
    --max-parallel "$_smoke_parallel" --port 8101 --results-dir "$RESULTS" \
    "${CLOCK_ARGS[@]}" > "$RESULTS/run_final_rt_smoke.log" 2>&1 || RC=$?
elif [ -n "${FINAL_RT_LANES:-}" ]; then
  IFS='|' read -ra _lane_specs <<< "$FINAL_RT_LANES"
  [ "${#_lane_specs[@]}" -eq "${#_VLLM_PORT_ARRAY[@]}" ] || {
    echo "[EVAL][FATAL] FINAL_RT_LANES count must equal replica count"; exit 33;
  }
  SHARD_PIDS=()
  for ((lane=0; lane<${#_lane_specs[@]}; lane++)); do
    (
      lane_rc=0
      endpoint="http://127.0.0.1:${_VLLM_PORT_ARRAY[$lane]}/v1/chat/completions"
      if [ "${_lane_specs[$lane]}" = "-" ]; then
        echo "[EVAL] final-rt lane=$lane reserved for joint tasks"
        exit 0
      fi
      IFS=',' read -ra _lane_shards <<< "${_lane_specs[$lane]}"
      for sh in "${_lane_shards[@]}"; do
        gport=$((8101 + lane * 100))
        suite="benchmark/suites/by_model/${SUITE_PREFIX:-qwen3-vl-8b-sft8}-$sh.yaml"
        echo "[EVAL] final-rt lane=$lane endpoint=$endpoint shard=$sh suite=$suite"
        SFT_ENDPOINT="$endpoint" python run_suite.py --suite "$suite" \
          --max-parallel 1 --port "$gport" --results-dir "$RESULTS" \
          --exclude-game 12_fireboy-and-watergirl \
          "${CLOCK_ARGS[@]}" > "$RESULTS/run_${sh}.log" 2>&1 || lane_rc=1
      done
      exit "$lane_rc"
    ) &
    SHARD_PIDS+=($!)
  done
  for pid in "${SHARD_PIDS[@]}"; do wait "$pid" || RC=1; done
  if [ "$RC" -eq 0 ] && [ -n "${FINAL_RT_MULTI_AGENT_SUITE:-}" ]; then
    echo "[EVAL] final-rt joint suite=$FINAL_RT_MULTI_AGENT_SUITE"
    python run_suite.py --suite "$FINAL_RT_MULTI_AGENT_SUITE" \
      --max-parallel 1 --port 8101 --results-dir "$RESULTS" \
      "${CLOCK_ARGS[@]}" > "$RESULTS/run_final_rt_joint.log" 2>&1 || RC=$?
  fi
  echo "[EVAL] all final-rt lanes done rc=$RC"
elif [ -n "${RUN_PRESET:-}" ]; then
  echo "[EVAL] single preset=$RUN_PRESET port=8101 results=$RESULTS"
  python main.py --config "$RUN_PRESET" --port 8101 --log-root "$RESULTS/single" --headless \
    "${CLOCK_ARGS[@]}" \
    > "$RESULTS/run_single.log" 2>&1 || RC=$?
  echo "[EVAL] single preset done rc=$RC"
else
  declare -A SHARD_PORT=( [g1]=8101 [g2]=8201 [g3]=8301 [g4]=8401 [g5]=8501 [g6]=8601 [g7]=8701 [g8]=8801 )
  SHARD_PIDS=()
  for sh in $SHARDS; do
    gport=${SHARD_PORT[$sh]:-8101}
    suite="benchmark/suites/by_model/${SUITE_PREFIX:-qwen3-vl-8b-sft}-$sh.yaml"
    [ -f "$suite" ] || { echo "[EVAL][FATAL] missing suite $suite"; exit 32; }
    echo "[EVAL] shard $sh -> suite=$suite game_port=$gport results=$RESULTS"
    python run_suite.py --suite "$suite" --max-parallel "$MAXP" \
      --port "$gport" --results-dir "$RESULTS" "${CLOCK_ARGS[@]}" \
      > "$RESULTS/run_${sh}.log" 2>&1 &
    SHARD_PIDS+=($!)
  done
  for pid in "${SHARD_PIDS[@]}"; do wait "$pid" || RC=1; done
  echo "[EVAL] all shards done rc=$RC"
fi

# 5. Aggregate shard summaries (skip for single-preset probes).
if [ -n "${RUN_PRESET:-}" ]; then
  echo "[EVAL] skip shard aggregate (RUN_PRESET)"
else
python3 - "$RESULTS" "$CKPT" <<'PY'
import csv, glob, json, os, sys
results_root, ckpt = sys.argv[1], sys.argv[2]
summ = {"checkpoint": ckpt, "shards": {}}
tot = suc = fai = err = 0
prog_sum = prog_n = 0.0
# combine each shard's summary.json overall block
for sj in sorted(glob.glob(os.path.join(results_root, "*", "summary.json"))):
    try:
        d = json.load(open(sj))
    except Exception as e:
        print("  (skip %s: %s)" % (sj, e)); continue
    shard = os.path.basename(os.path.dirname(sj))
    ov = d.get("overall") or {}
    summ["shards"][shard] = ov
    tot += ov.get("total_runs") or 0
    suc += ov.get("success_runs") or 0
    fai += ov.get("fail_runs") or 0
    err += ov.get("error_runs") or 0
    p = ov.get("task_avg_progress")
    n = ov.get("total_runs") or 0
    if isinstance(p, (int, float)) and n:
        prog_sum += p * n; prog_n += n
# per-game success from runs.csv
games = {}
for cf in sorted(glob.glob(os.path.join(results_root, "*", "runs.csv"))):
    for row in csv.DictReader(open(cf)):
        g = row.get("game_id") or "?"
        st = row.get("final_status")
        d = games.setdefault(g, {"n": 0, "success": 0})
        d["n"] += 1
        if st == "success":
            d["success"] += 1
summ["overall"] = {
    "total_runs": tot, "success_runs": suc, "fail_runs": fai, "error_runs": err,
    "task_avg_success_rate": (suc / tot) if tot else 0.0,
    "task_avg_progress": (prog_sum / prog_n) if prog_n else None,
}
summ["by_game"] = {g: {**v, "success_rate": (v["success"] / v["n"]) if v["n"] else 0.0}
                   for g, v in sorted(games.items())}
print("\n============ EVAL SUMMARY (%s) ============" % ckpt)
for g, v in summ["by_game"].items():
    print("  %-32s n=%d success=%d rate=%.3f" % (g, v["n"], v["success"], v["success_rate"]))
o = summ["overall"]
print("  OVERALL success=%d/%d rate=%.4f avg_progress=%s" %
      (suc, tot, o["task_avg_success_rate"], o["task_avg_progress"]))
out = os.path.join(results_root, "eval_summary.json")
json.dump(summ, open(out, "w"), indent=2)
print("  summary -> %s" % out)
print("===========================================\n")
PY
fi

if [ -n "${FINAL_RT_SMOKE_SUITE:-}" ] && [ "$RC" -eq 0 ]; then
  for port in "${_VLLM_PORT_ARRAY[@]}"; do
    cp -f "/tmp/vllm_${port}.log" "$RESULTS/vllm_${port}.log" 2>/dev/null || true
  done
  python3 docker/check_final_rt_smoke.py "$RESULTS" || {
    echo "[EVAL][FATAL] final-rt smoke acceptance gate failed"
    RC=34
  }
fi

if [ -n "${EXPECTED_TASKS:-}" ] && [ "$RC" -eq 0 ]; then
  python3 - "$RESULTS" "$EXPECTED_TASKS" <<'PY' || RC=35
import csv, json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
rows = [row for path in root.glob('*/runs.csv') for row in csv.DictReader(path.open())]
keys = {(row['game_id'], row['task_id'], row['repeat_index']) for row in rows}
errors = [row for row in rows if row['final_status'] not in ('success', 'fail')]
report = {'expected': expected, 'completed': len(rows), 'unique': len(keys),
          'errors': len(errors), 'passed': len(rows) == len(keys) == expected and not errors}
(root / 'completion_gate.json').write_text(json.dumps(report, indent=2) + '\n')
print('[EVAL] completion gate', report)
if not report['passed']:
    raise SystemExit(1)
PY
fi
echo "[EVAL] DONE CKPT=$CKPT rc=$RC"
exit "$RC"
