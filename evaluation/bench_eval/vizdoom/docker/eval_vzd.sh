#!/usr/bin/env bash
# ============================================================================
# Single-node ViZDoom eval: N vLLM replicas + M harness workers in one pod.
#
# Why this is simpler than the OmniGameArena driver: ViZDoom renders in software
# on the CPU. There is no Vulkan, no /dev/dri, no port shim, no one-instance-per-
# GPU rule and no render gate -- a Doom instance is ~40 MB of RAM and a fraction
# of a core, and a 350-step episode costs ~2 s of engine time. Every GPU in the
# job therefore goes to vLLM, and the only scaling question is how many replicas
# the prefill load wants.
#
# Doom also only advances when we ask it to, so the world is frozen while the
# model thinks (OGA's `pdq`). Extra workers per replica cost latency but not
# score -- which is why PAR may exceed VLLM_REPLICAS here, unlike under `lcrt`.
#
# Required env: SERVED_NAME plus one of CKPT / MODEL_REL / MODEL_DIR.
# Optional env: MAPS EPISODES PAR VLLM_REPLICAS VLLM_TP VLLM_BASE_PORT GPU_UTIL
#               MAX_LEN HF_EXPORT CONFIG PROMPT_STYLE HISTORY HISTORY_REASONING
#               TEMP MAX_TOKENS REASONING REASONING_STYLE REASONING_MAX_TOKENS
#               CLOCK MAX_STEPS RESIZE MOUSE_MODE TURN_DEG LOOK_DEG STOP
#               RECORD_VIDEO KEEP_GOING OBJECTIVE_SET OBJECTIVE_POS RECAP_LAST
#               CARRY_THINK IMAGE_ONLY_USER
# ============================================================================
set -uo pipefail

# BENCH_WORKSPACE_BEGIN -- filesystem checks only, before any serving/game code.
_bench_hint="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd -P)"
WS=""
for cand in "${WS_OVERRIDE:-}" "${BENCH_CODE_HINT:-}" "$_bench_hint"; do
  [ -n "$cand" ] || continue
  [ "$(cat "$cand/.bench-workspace-id" 2>/dev/null)" = "bench-workspace-v1" ] || continue
  [ -f "$cand/evaluation/vizdoom/scripts/run_benchmark.py" ] || continue
  _bench_physical="$(readlink -f "$cand")"
  [ "$(readlink -f "$cand/evaluation/vizdoom")" = "$_bench_physical/evaluation/vizdoom" ] || continue
  WS="$_bench_physical"; break
done
[ -n "$WS" ] || { echo '[BENCH][FATAL] own code workspace not mounted; refusing another workspace' >&2; exit 20; }
export WS_OVERRIDE="$WS" BENCH_CODE_HINT="$WS"
# BENCH_WORKSPACE_END


# Pin the code for this job's lifetime. bash reads a running script off disk by
# byte offset, so editing these files on shared storage mid-run makes a live job
# resume mid-line (OGA lost three shards' aggregation to exactly this).
if [ "${VZD_SNAPSHOT:-0}" != "1" ]; then
  _src="$(cd "$(dirname "$0")" && pwd)"
  _snap="/tmp/vzd_scripts_$$"
  if mkdir -p "$_snap" && cp "$_src"/*.sh "$_src"/*.py "$_snap"/ 2>/dev/null \
     && [ -f "$_snap/$(basename "$0")" ]; then
    export VZD_SNAPSHOT=1 VZD_SCRIPT_DIR="$_snap"
    echo "[VZD] running from snapshot $_snap (source $_src)"
    exec bash "$_snap/$(basename "$0")" "$@"
  fi
  echo "[VZD][WARN] could not snapshot scripts; running directly from $_src"
fi
SDIR="${VZD_SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

SERVED_NAME="${SERVED_NAME:?set SERVED_NAME (the vLLM served-model-name / result key)}"
EPISODES="${EPISODES:-3}"
LOGD="/tmp/vzd_eval"; mkdir -p "$LOGD"

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[VZD][WARN] %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m[VZD][FATAL] %s\033[0m\n' "$*" >&2; exit "${2:-1}"; }

echo "[VZD] host=$(hostname) node=${NODE_NAME:-?} uid=$(id -u) date=$(date)"
echo "[VZD] SERVED_NAME=$SERVED_NAME CKPT=${CKPT:-<none>} MODEL_REL=${MODEL_REL:-<none>}"

# -- 0. locate the workspace + the weights ---------------------------------
# Workspace already selected by the identity-checked prologue.

echo "[VZD] WS=$WS"

ROOT="$WS/evaluation/vizdoom"
CONFIG="${CONFIG:-$ROOT/configs/doom_suite.yaml}"
# The helpers in $SDIR run from a /tmp snapshot, so a path relative to their own
# file cannot find the harness package. Hand them the real root explicitly.
export VZD_ROOT="$ROOT"

# Respect an explicit HF_EXPORT; otherwise locate this checkpoint, not just a root.
if [ -z "${HF_EXPORT:-}" ]; then
  for cand in "$WS/hf_export"; do
    if [ -n "${CKPT:-}" ] && [ -f "$cand/$CKPT/config.json" ]; then
      HF_EXPORT="$cand"; break
    fi
  done
  HF_EXPORT="${HF_EXPORT:-$WS/hf_export}"
fi
if [ -n "${MODEL_REL:-}" ]; then
  case "$MODEL_REL" in
    /*) MODEL_DIR="${MODEL_DIR:-$MODEL_REL}" ;;
    *) MODEL_DIR="${MODEL_DIR:-${BENCH_MODEL_WORKSPACE:-$WS}/$MODEL_REL}" ;;
  esac
fi
[ -n "${CKPT:-}" ] && MODEL_DIR="${MODEL_DIR:-$HF_EXPORT/$CKPT}"
MODEL_DIR="${MODEL_DIR:-}"
[ -n "$MODEL_DIR" ] || die "give one of CKPT / MODEL_REL / MODEL_DIR" 21
if [ ! -f "$MODEL_DIR/config.json" ]; then
  # The model mount is only visible inside the pod, so the submitter cannot check
  # the step number. Print what does exist, otherwise the only way to find out
  # which checkpoints are there is to guess and burn another job.
  warn "no HF model at $MODEL_DIR -- checkpoints that do exist under $HF_EXPORT:"
  # Two layouts coexist: one nests as <run>/global_step_N, the other puts
  # <size>_s<stage>_<hours>_gs<step> directly under the root. Find both by
  # looking for config.json rather than by name.
  for d in "$HF_EXPORT"/*/ "$HF_EXPORT"/*/*/; do
    [ -f "$d/config.json" ] && echo "  ${d#"$HF_EXPORT"/}"
  done 2>/dev/null | head -120
  # A training checkpoint that has not been converted yet has no config.json, so
  # the list above would call it missing when it is merely un-exported. Print the
  # raw entries of $HF_EXPORT and its parent too, otherwise "is this stage
  # trained but un-exported, or not trained at all?" costs another job to answer.
  warn "raw entries under $HF_EXPORT and its parent:"
  ls -1 "$HF_EXPORT" 2>/dev/null | sed "s|^|  $(basename "$HF_EXPORT")/|" | head -80
  ls -1 "$(dirname "$HF_EXPORT")" 2>/dev/null | sed 's|^|  ../|' | head -40
  die "no HF model at $MODEL_DIR" 21
fi
echo "[VZD] MODEL_DIR=$MODEL_DIR"

# -- 1. harness deps -------------------------------------------------------
# Vendored wheels first so a pod with no egress still works; the ViZDoom wheel
# is self-contained (it bundles SDL2 + boost + freedoom2.wad), so nothing has to
# be apt-installed and no X server is involved.
say "Provisioning harness deps"
WHEELS="$ROOT/vendor/wheels"
python3 -c "import vizdoom" 2>/dev/null || \
  pip install -q --no-index --no-deps --find-links "$WHEELS" vizdoom 2>&1 | tail -3
python3 -c "import vizdoom" 2>/dev/null || {
  warn "vendored ViZDoom wheel did not match this interpreter; falling back to PyPI"
  pip install -q --no-deps vizdoom==1.2.4 2>&1 | tail -3
}
python3 -c "import vizdoom, sys; print('[VZD] vizdoom', vizdoom.__version__, 'py', sys.version.split()[0])" \
  || die "ViZDoom is not importable; the benchmark cannot run" 22

RECORD_VIDEO="${RECORD_VIDEO:-1}"
if [ "$RECORD_VIDEO" != "0" ]; then
  python3 -c "import imageio, imageio_ffmpeg" 2>/dev/null || \
    pip install -q --no-index --no-deps --find-links "$WHEELS" imageio imageio_ffmpeg 2>&1 | tail -3
  # The MP4 is the only way to check afterwards that the agent was really
  # playing, and the writer fails loudly rather than emitting empty files, so a
  # missing backend has to be settled here and not mid-episode.
  python3 -c "import imageio, imageio_ffmpeg" 2>/dev/null || {
    warn "imageio/imageio_ffmpeg unavailable; continuing WITHOUT video"
    RECORD_VIDEO=0
  }
fi
python3 -c "import yaml" 2>/dev/null || pip install -q pyyaml 2>&1 | tail -1

# -- 2. GPU / worker layout ------------------------------------------------
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "${NGPU:-0}" -ge 1 ] || die "no visible GPU" 25
VLLM_TP="${VLLM_TP:-1}"
VLLM_REPLICAS="${VLLM_REPLICAS:-$((NGPU / VLLM_TP))}"
[ "$VLLM_REPLICAS" -ge 1 ] || VLLM_REPLICAS=1
VLLM_GPUS=$((VLLM_TP * VLLM_REPLICAS))
[ "$VLLM_GPUS" -le "$NGPU" ] || die "vLLM wants $VLLM_GPUS GPUs but only $NGPU are visible" 25
VLLM_BASE_PORT="${VLLM_BASE_PORT:-30000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
HISTORY="${HISTORY:-20}"
CLOCK="${CLOCK:-pdq}"

# ~950 image tokens per 1280x720 frame, so a 20-frame window is ~19k on its own.
if [ "$HISTORY" -gt 20 ]; then
  MAX_LEN="${MAX_LEN:-$(( (HISTORY + 1) * 1000 + 8192 ))}"
elif [ -n "${HISTORY_REASONING:-}" ] && [ "${HISTORY_REASONING:-0}" != "0" ]; then
  MAX_LEN="${MAX_LEN:-32768}"
else
  MAX_LEN="${MAX_LEN:-24576}"
fi

MAPS="${MAPS:-$(python3 -c "
import sys, yaml
print(' '.join((yaml.safe_load(open(sys.argv[1])) or {}).get('maps') or []))" "$CONFIG")}"
[ -n "$MAPS" ] || die "no maps to run" 23
MAP_ARR=($MAPS)
NUNITS=$(( ${#MAP_ARR[@]} * EPISODES ))
PAR="${PAR:-$(( VLLM_REPLICAS < NUNITS ? VLLM_REPLICAS : NUNITS ))}"
[ "$PAR" -gt "$NUNITS" ] && PAR="$NUNITS"
[ "$PAR" -lt 1 ] && PAR=1

if [ "$CLOCK" = "lcrt" ] && [ "$PAR" -gt "$VLLM_REPLICAS" ] \
   && [ "${ALLOW_LCRT_CONTENTION:-0}" != "1" ]; then
  # lcrt fast-forwards the measured per-step latency into the game, so a replica
  # shared by two workers halves the game time the agent controls and the score
  # then describes the serving layout instead of the model.
  die "CLOCK=lcrt with PAR=$PAR on $VLLM_REPLICAS replica(s) measures the serving
       stack, not the model. Use PAR<=VLLM_REPLICAS or ALLOW_LCRT_CONTENTION=1" 26
fi

echo "[VZD] maps(${#MAP_ARR[@]})=$MAPS episodes=$EPISODES units=$NUNITS"
echo "[VZD] layout: $NGPU GPU | vLLM x$VLLM_REPLICAS (tp=$VLLM_TP) on GPU 0..$((VLLM_GPUS-1)) | $PAR CPU workers"
echo "[VZD] knobs: mouse=${MOUSE_MODE:-<config>} clock=$CLOCK history=$HISTORY temp=${TEMP:-<config>} reasoning=${REASONING:-<config>}/${REASONING_STYLE:-<config>} prompt=${PROMPT_STYLE:-<config>} max_len=$MAX_LEN video=$RECORD_VIDEO"

# -- 3. vLLM replicas ------------------------------------------------------
# Qwen3.5 vision needs the image's vllmenv (qwen3_5 linear attention); putting it
# first is a no-op when both interpreters carry the same vLLM.
if [ -x /root/vllmenv/bin/vllm ]; then
  export PATH="/root/vllmenv/bin:$PATH"
  echo "[VZD] vllm=$(command -v vllm)"
fi

cleanup() {
  for pf in "$LOGD"/vllm_*.pid; do
    [ -f "$pf" ] && kill "$(cat "$pf")" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# All replicas load the same tens of GB of safetensors off AFS. Started together
# they read it N times in parallel and throttle each other (OGA measured 554 s
# just for "Loading weights" with 3 replicas). One sequential read first puts the
# files in page cache; the node has ~2 TB of RAM.
if [ "$VLLM_REPLICAS" -gt 1 ] && [ "${PREWARM_WEIGHTS:-1}" != "0" ]; then
  t0=$(date +%s)
  cat "$MODEL_DIR"/*.safetensors > /dev/null 2>&1 || true
  echo "[VZD] weights prewarmed into page cache in $(( $(date +%s) - t0 ))s"
fi

declare -a ENDPOINTS=()
# EXTERNAL_STEP_CONFIG: an owned streaming /step service replaces the local vLLM.
# The copied client still speaks chat/completions to 127.0.0.1; the rt_external
# transport (activated by BENCH_EXTERNAL_RUNTIME in sitecustomize) maps each call
# onto the service. Everything after this block is unchanged.
if [ -n "${EXTERNAL_STEP_CONFIG:-}" ]; then
  export BENCH_EXTERNAL_RUNTIME=1 BENCH_EXTERNAL_CONFIG="$WS/$EXTERNAL_STEP_CONFIG" \
         BENCH_EXTERNAL_MODEL="${EXTERNAL_STEP_MODEL:?}" BENCH_EXTERNAL_TIMING_LOG="$LOGD/api_timing.jsonl"
  export PYTHONPATH="$WS/rt_external/bootstrap:$WS:${PYTHONPATH:-}"
  say "External /step service ($BENCH_EXTERNAL_MODEL); no local vLLM"
  ENDPOINTS+=("http://127.0.0.1:$VLLM_BASE_PORT/v1")
  # Warmup is not scored; the shared gateway occasionally answers with a
  # non-JSON page, so allow three attempts before giving up.
  _wok=0; for _wt in 1 2 3; do
    python3 -m rt_external.step_warmup "$BENCH_EXTERNAL_CONFIG" "$BENCH_EXTERNAL_MODEL" "$LOGD/warmup_timing.jsonl" && { _wok=1; break; }
    sleep 20
  done
  [ "$_wok" = 1 ] || die "external /step warmup failed" 27
else
say "Serving $SERVED_NAME ($VLLM_REPLICAS replica(s), tp=$VLLM_TP)"
for r in $(seq 0 $((VLLM_REPLICAS-1))); do
  port=$((VLLM_BASE_PORT + r))
  ENDPOINTS+=("http://127.0.0.1:$port/v1")
  gpus="$(seq -s, $((r*VLLM_TP)) $(((r+1)*VLLM_TP - 1)))"
  log="$LOGD/vllm_$port.log"
  echo "[VZD] replica $r -> :$port on GPU $gpus ($log)"
  TP_FLAG=""; [ "$VLLM_TP" -gt 1 ] && TP_FLAG="--tensor-parallel-size $VLLM_TP"
  TRUST_FLAG=""
  if [ -n "${HISTORY_REASONING:-}" ] && [ "${HISTORY_REASONING:-0}" != "0" ]; then
    # The agent sends a patched chat template per request to keep <think> in
    # history; vLLM answers 400 to request templates unless they are trusted.
    TRUST_FLAG="--trust-request-chat-template"
  fi
  CUDA_VISIBLE_DEVICES="$gpus" nohup vllm serve "$MODEL_DIR" \
      --served-model-name "$SERVED_NAME" \
      --port "$port" \
      --dtype bfloat16 --trust-remote-code \
      --max-model-len "$MAX_LEN" \
      --limit-mm-per-prompt "{\"image\": $((HISTORY+1))}" \
      --gpu-memory-utilization "$GPU_UTIL" $TP_FLAG $TRUST_FLAG \
      > "$log" 2>&1 &
  echo $! > "$LOGD/vllm_$port.pid"
  [ "$r" -lt $((VLLM_REPLICAS-1)) ] && sleep "${VLLM_LAUNCH_STAGGER:-10}"
done

for r in $(seq 0 $((VLLM_REPLICAS-1))); do
  port=$((VLLM_BASE_PORT + r)); pid="$(cat "$LOGD/vllm_$port.pid")"
  tries="${VLLM_READY_TRIES:-240}"          # x10s; AFS reads alone can take ~10 min
  for i in $(seq 1 "$tries"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      # `tail -40` only catches the APIServer re-raising "Engine core
      # initialization failed"; the EngineCore child's real exception is
      # hundreds of lines earlier, and the job log API truncates to ~60 lines,
      # so copy the whole thing to AFS where it outlives the pod.
      keep="$ROOT/runs/_vllm_fail"; mkdir -p "$keep"
      cp "$LOGD/vllm_$port.log" "$keep/${SERVED_NAME}_$port.log" 2>/dev/null \
        && warn "full vLLM log kept at runs/_vllm_fail/${SERVED_NAME}_$port.log"
      grep -nE "Error|Exception|Traceback|No such file|out of memory" \
        "$LOGD/vllm_$port.log" | head -20
      tail -40 "$LOGD/vllm_$port.log"
      die "vLLM :$port died at startup" 27
    fi
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      # The submitter's wait-ready poll greps for exactly this string.
      echo "vLLM ready (~${i}0s) :$port"
      break
    fi
    sleep 10
    [ "$i" = "$tries" ] && { tail -40 "$LOGD/vllm_$port.log"; die "vLLM :$port not ready after $((tries/6))min" 27; }
  done
done
fi

# -- 3b. latency breakdown probe (VZD_PROBE=1) ------------------------------
# Not an eval: runs one instrumented episode against replica 0 and exits. Kept
# here rather than in its own launcher because it needs exactly the serving
# layout an eval gets (same MODEL_DIR, same vLLM flags), and a second launcher
# would drift from this one.
if [ "${VZD_PROBE:-0}" = "1" ]; then
  cd "$ROOT"
  out="$ROOT/runs/_latency_probe/${SERVED_NAME}.json"
  python3 -u "$SDIR/vzd_latency_probe.py" \
    --base-url "${ENDPOINTS[0]}" --served-name "$SERVED_NAME" \
    --game "${PROBE_GAME:-battle1}" --steps "${PROBE_STEPS:-120}" \
    --history "$HISTORY" --resize "$RESIZE" --out "$out" \
    --sweep-reps "${PROBE_SWEEP_REPS:-0}" --mm-cache-reps "${PROBE_MM_REPS:-0}"
  rc=$?
  echo "[VZD] probe done rc=$rc -> $out"
  exit "$rc"
fi

# -- 4. plan the units over workers, longest first --------------------------
PLAN="$LOGD/worker_plan.tsv"
UNIT_ARGS=()
for m in "${MAP_ARR[@]}"; do UNIT_ARGS+=("$m:$EPISODES"); done
python3 "$SDIR/vzd_plan_workers.py" "$PAR" "${UNIT_ARGS[@]}" > "$PLAN" \
  || die "unit planning failed" 24
echo "[VZD] worker plan (worker / units / est min):"
cat "$PLAN"

SMI_LOG="$LOGD/smi_$(hostname).log"
( while true; do echo "== $(date +%H:%M:%S) =="; \
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 30; done ) \
  > "$SMI_LOG" 2>&1 &
SMI_PID=$!

# -- 5. run ----------------------------------------------------------------
say "Running $NUNITS units on $PAR workers"
cd "$ROOT"
OPT=()
[ -n "${PROMPT_STYLE:-}" ]        && OPT+=(--prompt-style "$PROMPT_STYLE")
[ -n "${HISTORY:-}" ]             && OPT+=(--history "$HISTORY")
[ -n "${HISTORY_REASONING:-}" ]   && OPT+=(--history-reasoning "$HISTORY_REASONING")
[ -n "${TEMP:-}" ]                && OPT+=(--temp "$TEMP")
[ -n "${MAX_TOKENS:-}" ]          && OPT+=(--max-tokens "$MAX_TOKENS")
[ -n "${REASONING:-}" ]           && OPT+=(--reasoning "$REASONING")
[ -n "${REASONING_STYLE:-}" ]     && OPT+=(--reasoning-style "$REASONING_STYLE")
[ -n "${REASONING_MAX_TOKENS:-}" ] && OPT+=(--reasoning-max-tokens "$REASONING_MAX_TOKENS")
[ -n "${MAX_STEPS:-}" ]           && OPT+=(--max-steps "$MAX_STEPS")
[ -n "${RESIZE:-}" ]              && OPT+=(--resize "$RESIZE")
[ -n "${MOUSE_MODE:-}" ]          && OPT+=(--mouse-mode "$MOUSE_MODE")
[ -n "${TURN_DEG:-}" ]            && OPT+=(--turn-deg "$TURN_DEG")
[ -n "${LOOK_DEG:-}" ]            && OPT+=(--look-deg "$LOOK_DEG")
[ -n "${STOP:-}" ]                && OPT+=(--stop "$STOP")
[ -n "${OBJECTIVE_SET:-}" ]       && OPT+=(--objective-set "$OBJECTIVE_SET")
[ -n "${OBJECTIVE_POS:-}" ]       && OPT+=(--objective-pos "$OBJECTIVE_POS")
[ "${RECAP_LAST:-0}" != "0" ]     && OPT+=(--recap-last)
[ "${CARRY_THINK:-0}" != "0" ]    && OPT+=(--carry-think)
[ "${IMAGE_ONLY_USER:-0}" != "0" ] && OPT+=(--image-only-user)
OPT+=(--clock "$CLOCK")
if [ "$RECORD_VIDEO" != "0" ]; then OPT+=(--record-video); else OPT+=(--no-record-video); fi
[ "${KEEP_GOING:-1}" != "0" ] && OPT+=(--keep-going)

declare -a WPIDS=()
for w in $(seq 0 $((PAR-1))); do
  units="$(awk -F'\t' -v w="$w" '$1==w{print $2}' "$PLAN")"
  [ -n "$units" ] || continue
  replica=$((w % VLLM_REPLICAS))
  endpoint="${ENDPOINTS[$replica]}"
  wlog="$LOGD/worker_$w.log"
  echo "[VZD] worker $w -> $endpoint : $units  ($wlog)"
  (
    python3 scripts/run_benchmark.py \
      --config "$CONFIG" \
      --served-name "$SERVED_NAME" \
      --base-url "$endpoint" \
      --units $units \
      "${OPT[@]}" > "$wlog" 2>&1
    echo "$?" > "$LOGD/rc_w$w"
  ) &
  WPIDS+=("$!")
done
for p in "${WPIDS[@]}"; do wait "$p"; done
kill "$SMI_PID" 2>/dev/null || true

RC=0
for w in $(seq 0 $((PAR-1))); do
  [ -f "$LOGD/rc_w$w" ] || continue
  rc="$(cat "$LOGD/rc_w$w")"
  if [ "$rc" -ne 0 ]; then
    warn "worker $w rc=$rc"; tail -25 "$LOGD/worker_$w.log"; RC=1
  fi
done
grep -h '^\[episode end\]' "$LOGD"/worker_*.log 2>/dev/null | sort || true

# -- 6. aggregate + validity gate ------------------------------------------
OUT_ROOT="$(python3 -c "
import sys, yaml, os
cfg = yaml.safe_load(open(sys.argv[1])) or {}
root = cfg.get('output_root', 'runs/doom')
print(root if os.path.isabs(root) else os.path.join(sys.argv[2], root))" "$CONFIG" "$ROOT")"
python3 "$SDIR/vzd_aggregate.py" "$OUT_ROOT" "$SERVED_NAME"
# A run whose turns never parsed, or that never aimed, still prints four map
# means. Check before anyone reads the table above.
python3 "$SDIR/vzd_check_sanity.py" "$OUT_ROOT" "$SERVED_NAME" \
  --episodes "$EPISODES" --maps "$MAPS"
GATE=$?

ARCH="$ROOT/runs/_launch_logs/${SERVED_NAME}_$(hostname)_$(date +%m%d_%H%M%S)"
mkdir -p "$ARCH" && cp "$LOGD"/*.log "$LOGD"/*.tsv "$ARCH"/ 2>/dev/null || true
echo "[VZD] logs -> $ARCH"

echo "[VZD] DONE served=$SERVED_NAME workers=$PAR rc=$RC gate=$GATE"
[ "$GATE" -ne 0 ] && warn "validity gate flagged this run -- do NOT register these scores"
exit "$RC"
