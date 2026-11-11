#!/usr/bin/env bash
# ============================================================================
# Single-node PARALLEL OmniGameArena eval: one job, one node, every game at
# once against ONE shared vLLM.
#
# Why this replaces the earlier sharded one-game-per-job path:
#   the sharded path ran one game per JOB because UE5 hardcodes RemoteInput port
#   12345 and a k8s pod cannot create network namespaces (no CAP_NET_ADMIN).
#   That cost 2 GPUs per game -- 14 for the 7-game suite, spread over up to 7
#   nodes -- to do work that measured batch=1 on vLLM (KV cache ~2.6%) and ~3 GB
#   of VRAM per UE5. The LD_PRELOAD shim (ue5_port_shim.so) rewrites that bind()
#   to a per-instance port, so N UE5 now coexist in ONE pod with no capabilities
#   at all, and the whole suite fits on a single node.
#
# Layout on an 8-GPU node (defaults):
#   GPU 0                  vLLM replica 0 (all drivers share it -> real batching)
#   GPU 1..7               one UE5 Vulkan adapter each, ports 12401..12407
#   games -> workers       round-robin over the UE5 instances that warmed OK
#
# A UE5 instance that never survives its first render only loses its own slot:
# its games are redistributed over the healthy workers instead of failing the job.
#
# env: CKPT (required, path under hf_export) plus the same knobs as
#      eval_ckpt.sh: SERVED_NAME GAMES GAMES_CONFIG EPISODES TEMP REASONING
#      REASONING_STYLE REASONING_MAX_TOKENS MAP_WAIT INVERT_Y STOP HISTORY
#      MODEL_DIR MODEL_REL CLOCK
# parallel-specific: PAR VLLM_REPLICAS VLLM_TP UE5_BASE_PORT VLLM_BASE_PORT
#                    GPU_UTIL UE5_TRIES
# ============================================================================
set -uo pipefail

# BENCH_WORKSPACE_BEGIN -- filesystem checks only, before any serving/game code.
_bench_hint="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." 2>/dev/null && pwd -P)"
WS=""
for cand in "${WS_OVERRIDE:-}" "${BENCH_CODE_HINT:-}" "$_bench_hint"; do
  [ -n "$cand" ] || continue
  [ "$(cat "$cand/.bench-workspace-id" 2>/dev/null)" = "bench-workspace-v1" ] || continue
  [ -f "$cand/evaluation/omnigamearena/omnigamearena/scripts/run_benchmark.py" ] || continue
  _bench_physical="$(readlink -f "$cand")"
  [ "$(readlink -f "$cand/evaluation/omnigamearena")" = "$_bench_physical/evaluation/omnigamearena" ] || continue
  WS="$_bench_physical"; break
done
[ -n "$WS" ] || { echo '[BENCH][FATAL] own code workspace not mounted; refusing another workspace' >&2; exit 20; }
export WS_OVERRIDE="$WS" BENCH_CODE_HINT="$WS"
# BENCH_WORKSPACE_END


. "$(cd "$(dirname "$0")" && pwd)/lib_tz.sh"

# Pin the code for this job's lifetime: bash reads a running script off disk by
# byte offset, so editing these files on shared storage mid-run makes live jobs
# resume mid-line. (Observed 2026-08-03: three shards lost their aggregation.)
if [ "${OGA_SNAPSHOT:-0}" != "1" ]; then
  _src="$(cd "$(dirname "$0")" && pwd)"
  _snap="/tmp/oga_scripts_$$"
  if mkdir -p "$_snap" && cp "$_src"/*.sh "$_src"/*.py "$_snap"/ 2>/dev/null \
     && cp "$_src"/ue5_port_shim.so "$_src"/ue5_port_shim.c "$_snap"/ 2>/dev/null \
     && [ -f "$_snap/$(basename "$0")" ]; then
    export OGA_SNAPSHOT=1 OGA_SCRIPT_DIR="$_snap"
    echo "[PAR] running from snapshot $_snap (source $_src)"
    exec bash "$_snap/$(basename "$0")" "$@"
  fi
  echo "[PAR][WARN] could not snapshot scripts; running directly from $_src"
fi
SDIR="${OGA_SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

CKPT="${CKPT:?set CKPT to a path under hf_export (e.g. run/global_step_800)}"
GAMES_CONFIG="${GAMES_CONFIG:-solo_suite}"
EPISODES="${EPISODES:-3}"
CKPT_ID="${SERVED_NAME:-$(echo "$CKPT" | tr '/' '__')}"
CLOCK="${CLOCK:-pdq}"
LOGD="/tmp/oga_parallel"; mkdir -p "$LOGD"

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[PAR][WARN] %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m[PAR][FATAL] %s\033[0m\n' "$*" >&2; exit "${2:-1}"; }

echo "[PAR] host=$(hostname) node=${NODE_NAME:-?} uid=$(id -u) CKPT=$CKPT CKPT_ID=$CKPT_ID date=$(date)"

# -- 0. locate workspace + weights -----------------------------------------
# Workspace already selected by the identity-checked prologue.

echo "[PAR] WS=$WS"
# This copy uses WS directly; no legacy user-directory symlink is needed.

ROOT_EVAL="$WS/evaluation/omnigamearena"
ARENA="$ROOT_EVAL/omnigamearena"
UE5_DIR="${BENCH_UE5_DIR:-$ROOT_EVAL/omnigamearena_ue5/Linux}"
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
MODEL_DIR="${MODEL_DIR:-$HF_EXPORT/$CKPT}"
[ -f "$MODEL_DIR/config.json" ] || die "no HF model at $MODEL_DIR" 21
echo "[PAR] MODEL_DIR=$MODEL_DIR"

SHIM="$SDIR/ue5_port_shim.so"
[ -f "$SHIM" ] || SHIM="$ROOT_EVAL/docker/ue5_port_shim.so"
[ -f "$SHIM" ] || die "port shim not found; build it with
  gcc -shared -fPIC -O2 -o ue5_port_shim.so ue5_port_shim.c -ldl" 24

# The prebuilt .so is compiled outside this image, so verify it actually loads
# and rewrites a bind here before trusting the whole run to it. dlsym moved into
# libc in glibc 2.34, so a container older than the build host would silently
# fail to preload -- in that case rebuild from source.
_shim_works() {
  UE5_PORT_OVERRIDE=12999 UE5_PORT_SHIM_QUIET=1 LD_PRELOAD="$1" python3 - <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
s.bind(("127.0.0.1", 12345))
sys.exit(0 if s.getsockname()[1] == 12999 else 1)
PY
}
if _shim_works "$SHIM"; then
  echo "[PAR] port shim OK: $SHIM"
elif command -v gcc >/dev/null && [ -f "$SDIR/ue5_port_shim.c" ] \
     && gcc -shared -fPIC -O2 -o /tmp/ue5_port_shim.so "$SDIR/ue5_port_shim.c" -ldl 2>&1 \
     && _shim_works /tmp/ue5_port_shim.so; then
  SHIM=/tmp/ue5_port_shim.so
  echo "[PAR] port shim rebuilt in-container: $SHIM"
else
  die "port shim does not work in this image and could not be rebuilt" 24
fi
chmod o+rx "$SHIM" 2>/dev/null || true

# -- 1. Vulkan needs DRM render nodes; the pod starts without /dev/dri -------
if [ ! -e /dev/dri/renderD128 ]; then
  echo "[PAR] creating /dev/dri render/card nodes (major 226)"
  mkdir -p /dev/dri
  for m in $(seq 0 7);     do mknod -m 0666 "/dev/dri/card$m"     c 226 "$m" 2>/dev/null || true; done
  for m in $(seq 128 135); do mknod -m 0666 "/dev/dri/renderD$m" c 226 "$m" 2>/dev/null || true; done
fi
vulkaninfo --summary 2>&1 | grep -iE "deviceName|ERROR_INCOMPATIBLE|Cannot create" | head -3

# -- 2. GPU / worker layout -------------------------------------------------
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"
[ "${NGPU:-0}" -ge 2 ] || die "need >=2 visible GPUs, saw ${NGPU:-0}" 25
VLLM_TP="${VLLM_TP:-1}"
VLLM_REPLICAS="${VLLM_REPLICAS:-1}"
VLLM_GPUS=$((VLLM_TP * VLLM_REPLICAS))
# EXTERNAL_STEP_CONFIG: an owned streaming /step service replaces the local vLLM;
# no GPU is reserved for inference, UE5 starts on GPU 0.
[ -n "${EXTERNAL_STEP_CONFIG:-}" ] && VLLM_GPUS=0
[ "$VLLM_GPUS" -lt "$NGPU" ] || die "vLLM would claim all $NGPU GPUs; leave at least one for UE5" 25
UE5_BASE_PORT="${UE5_BASE_PORT:-12401}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-30000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
HISTORY="${HISTORY:-20}"
# A 20-frame prompt is already ~19k image tokens against the 24576 default, and
# keeping reasoning in history adds ~100 tokens per turn (measured: non-empty
# traces are 323 chars median, 478 max). That is only ~2k for 20 turns, but it
# lands on top of the tightest case, so give those runs room instead of letting
# vLLM truncate mid-experiment.
# A window longer than the inherited 20 needs its own ceiling: ~950 image tokens
# per 1280x720 frame means the 40-frame training window is ~38k on its own, and a
# 24576 default would only surface as a runtime failure. The 20-frame values are
# left exactly as every past batch ran them.
if [ "$HISTORY" -gt 20 ]; then
  MAX_LEN="${MAX_LEN:-$(( (HISTORY + 1) * 1000 + 8192 ))}"
elif [ -n "${HISTORY_REASONING:-}" ] && [ "${HISTORY_REASONING:-0}" != "0" ]; then
  MAX_LEN="${MAX_LEN:-32768}"
else
  MAX_LEN="${MAX_LEN:-24576}"
fi

SRC_CFG="$ARENA/configs/yourvla/${GAMES_CONFIG}.yaml"
[ -f "$SRC_CFG" ] || die "config missing: $SRC_CFG" 22
if [ -n "${GAMES:-}" ]; then
  GAME_LIST=($GAMES)
else
  GAME_LIST=($(python3 -c "
import sys, yaml
print(' '.join(yaml.safe_load(open(sys.argv[1]))['games']))" "$SRC_CFG"))
fi
NGAMES=${#GAME_LIST[@]}
[ "$NGAMES" -ge 1 ] || die "no games to run" 23

# PvP/Coop maps serve player 2 from a SECOND listener on 12346 inside the same
# UE5 process (verified: both ports bind at boot, and on a two-player map they
# render two different views). So a match still costs one GPU, but the instance
# owns a port PAIR and the shim has to move the pair as a block -- otherwise
# every concurrent match keeps its own 12401.. while all of them fight over the
# untranslated 12346. Mixing modes in one suite would need two port strides, so
# require the suite to be homogeneous.
MODES="$(PYTHONPATH="$ARENA" python3 - "${GAME_LIST[@]}" <<'PY'
import sys
from omni_game_arena.benchmark.games import get_game
print(" ".join(sorted({getattr(get_game(g), "mode", "solo") for g in sys.argv[1:]})))
PY
)" || die "could not read game modes from the registry" 23
case "$MODES" in
  solo)      PORT_SPAN=1 ;;
  pvp|coop|"coop pvp") PORT_SPAN=2 ;;
  *)         die "mixed solo and two-player games in one job ($MODES); run them as separate jobs" 23 ;;
esac
echo "[PAR] game modes: $MODES (port span $PORT_SPAN per UE5)"

# A shim built before two-player support only rewrites 12345, which would leave
# every instance's player-2 listener on the shared 12346: the second match to
# start would take whatever the first one bound. Fail loudly instead.
if [ "$PORT_SPAN" -gt 1 ]; then
  UE5_PORT_OVERRIDE=12990 UE5_PORT_SPAN=2 UE5_PORT_SHIM_QUIET=1 LD_PRELOAD="$SHIM" python3 - <<'PY' \
    || die "the port shim cannot move a 2-port span; rebuild it from ue5_port_shim.c" 24
import socket, sys
got = []
for p in (12345, 12346):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", p)); got.append(s.getsockname()[1])
sys.exit(0 if got == [12990, 12991] else 1)
PY
  echo "[PAR] port shim moves the player pair as a block"
fi

# One UE5 per GPU, hard requirement. Sharing a GPU between UE5 instances starves
# the render thread, and because an action chunk is held for a fixed wall-clock
# 6x33ms, a slow frame rate means the game barely advances while the keys are
# down: measured 2.4x (cue_chase) to 63x (obstacle_run_3d) less movement per
# identical command with 3.5 instances/GPU, plus 12% of steps not moving at all.
# Scores collapse (OVERALL 0.17 -> 0.08 on the same checkpoint), so a packed run
# is not a measurement of the model. Never pack; run games in sequence instead.
UE5_GPUS=$((NGPU - VLLM_GPUS))
PAR_EXPLICIT="${PAR:+yes}"
PAR="${PAR:-$NGAMES}"
[ "$PAR" -gt "$NGAMES" ] && PAR="$NGAMES"
[ "$PAR" -lt 1 ] && PAR=1

if [ "$PAR" -gt "$UE5_GPUS" ]; then
  if [ "${ALLOW_UE5_PACKING:-0}" = "1" ]; then
    echo "[PAR] !! WARNING: $PAR UE5 on $UE5_GPUS GPU(s) -- render-starved, SCORES ARE INVALID"
    echo "[PAR] !! only use this for render-contention experiments, never for evaluation"
  elif [ -n "$PAR_EXPLICIT" ]; then
    die "PAR=$PAR needs $PAR GPUs for UE5 but only $UE5_GPUS free ($NGPU total - $VLLM_GPUS vLLM). Raise --gpus, lower PAR, or drop --vllm-replicas; ALLOW_UE5_PACKING=1 is for render experiments only" 26
  else
    PAR="$UE5_GPUS"
    echo "[PAR] $NGAMES games over $PAR UE5 (one per free GPU); workers run their games in sequence"
  fi
fi
echo "[PAR] games($NGAMES)=${GAME_LIST[*]}"
echo "[PAR] layout: ${NGPU} GPUs | vLLM x$VLLM_REPLICAS (tp=$VLLM_TP) on GPU 0..$((VLLM_GPUS-1)) | $PAR UE5 on GPU $VLLM_GPUS..$((NGPU-1))"

# -- 3. per-game run configs, one set per vLLM replica ----------------------
# The endpoint has to be baked into the yaml: factory.py hands the config's
# base_url straight to the agent, so it wins over YOURVLA_BASE_URL and every
# driver would otherwise pile onto replica 0.
declare -a ENDPOINTS=()
for r in $(seq 0 $((VLLM_REPLICAS-1))); do
  ENDPOINTS+=("http://127.0.0.1:$((VLLM_BASE_PORT + r))/v1")
done

say "Generating $NGAMES run configs x $VLLM_REPLICAS replica(s)"
for r in $(seq 0 $((VLLM_REPLICAS-1))); do
  BASE_URL="${ENDPOINTS[$r]}" \
  MAP_WAIT="${MAP_WAIT:-60}" TEMP="${TEMP:-}" REASONING="${REASONING:-0}" \
  REASONING_MAX_TOKENS="${REASONING_MAX_TOKENS:-2048}" REASONING_STYLE="${REASONING_STYLE:-two_stage}" \
  HISTORY_EVICT_CHUNK="${HISTORY_EVICT_CHUNK:-1}" INVERT_Y="${INVERT_Y:-1}" STOP="${STOP:-}" \
  HISTORY="$HISTORY" HISTORY_REASONING="${HISTORY_REASONING:-}" \
  RESIZE="${RESIZE:-}" \
  PYTHONPATH="$ARENA:${PYTHONPATH:-}" \
    python3 "$SDIR/oga_make_configs.py" "$SRC_CFG" "$LOGD/configs/r$r" "$CKPT_ID" "${GAME_LIST[@]}" \
    > "$LOGD/configs_r$r.tsv" || die "config generation failed (replica $r)" 22
  cat "$LOGD/configs_r$r.tsv"
done
echo "[PAR] knobs: episodes=$EPISODES clock=$CLOCK temp=${TEMP:-<suite>} reasoning=${REASONING:-0}/${REASONING_STYLE:-two_stage} invert_y=${INVERT_Y:-1} stop=${STOP:-<default action_end>} resize=${RESIZE:-1280x720}"
echo "[PAR] context: history=$HISTORY frames, reasoning kept in history=${HISTORY_REASONING:-0} turns, max_len=$MAX_LEN"

# -- 4. UE5 instances (before vLLM: a loaded GPU pushes the first render past
#       the ~10s VulkanRHI fence, which is what kills half of all launches) ---
ue5_launch() {          # $1 = port, $2 = adapter; blocks until warm, ~45s
  PORT="$1" ADAPTER="$2" UE5_DIR="$UE5_DIR" SHIM="$SHIM" LOGD="$LOGD" \
  PORT_SPAN="$PORT_SPAN" \
  TRIES="${UE5_TRIES:-4}" WARMUP_SLEEP="${WARMUP_SLEEP:-30}" SHOT_PY="$SDIR/ue5_shot.py" \
    bash "$SDIR/ue5_instance.sh" >> "$LOGD/warm_$1.log" 2>&1
}
ue5_kill() {            # $1 = port
  local pg; pg="$(cat "$LOGD/ue5_$1.pgid" 2>/dev/null || true)"
  [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null || true
  rm -f "$LOGD/ue5_$1.pgid"
  sleep 2               # let the port clear before the replacement binds it
}

say "Warming $PAR UE5 instances"
declare -a UE5_PORTS=() UE5_ADAPTERS=() UE5_PIDS=()
for i in $(seq 0 $((PAR-1))); do
  port=$((UE5_BASE_PORT + i * PORT_SPAN))
  adapter=$((VLLM_GPUS + (i % (NGPU - VLLM_GPUS))))
  UE5_PORTS+=("$port"); UE5_ADAPTERS+=("$adapter")
  ue5_launch "$port" "$adapter" &
  UE5_PIDS+=("$!")
  sleep "${UE5_LAUNCH_STAGGER:-8}"      # spread the CPU-heavy boots
done

declare -a WORKERS=() WORKER_ADAPTERS=()
for i in $(seq 0 $((PAR-1))); do
  if wait "${UE5_PIDS[$i]}"; then
    WORKERS+=("${UE5_PORTS[$i]}"); WORKER_ADAPTERS+=("${UE5_ADAPTERS[$i]}")
  else
    warn "UE5 on port ${UE5_PORTS[$i]} never warmed; dropping that worker"
    tail -6 "$LOGD/warm_${UE5_PORTS[$i]}.log"
  fi
  grep -hE "^\[ue5:|^\[shot:" "$LOGD/warm_${UE5_PORTS[$i]}.log" | tail -4
done
NWORK=${#WORKERS[@]}
echo "[PAR] UE5 ready: $NWORK/$PAR  ports=${WORKERS[*]:-none}"

# Best-effort local run log (TSV: time, node, outcome, run), created on first
# write at runtime; not shipped with the repo. Write failures are ignored.
REG="$ROOT_EVAL/docker/node_render_registry.tsv"
record_outcome() {   # $1 = outcome tag; keeps a dead-render node out of future jobs
  printf "%s\t%s\t%s\t%s\n" "$(date +%F_%T)" "${NODE_NAME:-?}" "$1" \
    "$CKPT_ID/$GAMES_CONFIG/parallel" >> "$REG" 2>/dev/null || true
  echo "[PAR] node=${NODE_NAME:-?} outcome=$1 recorded -> $REG"
}

# Exit 17 (not 1) so the submitter can tell a dead-render NODE apart from our own
# config bugs; the phrasing matches what wait_stack_ready() greps for.
if [ "$NWORK" -lt 1 ]; then
  ARCH="$ARENA/runs/_launch_logs/parallel_${CKPT_ID}_$(hostname)_$(date +%m%d_%H%M%S)"
  mkdir -p "$ARCH" && cp "$LOGD"/*.log "$LOGD"/configs_r*.tsv "$ARCH"/ 2>/dev/null || true
  echo "[PAR] warm failure logs -> $ARCH"
  for _port in "${UE5_PORTS[@]}"; do
    echo "[PAR] UE5 :$_port last engine output"
    tail -60 "$LOGD/ue5_$_port.log" 2>/dev/null || true
  done
  printf '\n\033[1;31mERROR: UE5 failed to warm after %s attempts on all %s instances\033[0m\n' \
    "${UE5_TRIES:-4}" "$PAR" >&2
  record_outcome "FAIL_render_ue5warm"
  exit 17
fi

cleanup() {
  for p in "${WORKERS[@]:-}"; do
    pg="$(cat "$LOGD/ue5_$p.pgid" 2>/dev/null || true)"
    [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null || true
  done
  for pf in "$LOGD"/vllm_*.pid; do
    [ -f "$pf" ] && kill "$(cat "$pf")" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# -- 5. shared vLLM replica(s) ---------------------------------------------
# Qwen3.5 vision lives in the image's vllmenv (see q35_infer_probe.sh). The
# default PATH vLLM is what the Qwen3-VL 8B batches used; putting vllmenv first
# is a no-op when both are 0.17, and is required for qwen3_5 linear-attn.
if [ -x /root/vllmenv/bin/vllm ]; then
  export PATH="/root/vllmenv/bin:$PATH"
  echo "[PAR] vllm=$(command -v vllm)"
fi

if [ -n "${EXTERNAL_STEP_CONFIG:-}" ]; then
  export BENCH_EXTERNAL_RUNTIME=1 BENCH_EXTERNAL_CONFIG="$WS/$EXTERNAL_STEP_CONFIG" \
         BENCH_EXTERNAL_MODEL="${EXTERNAL_STEP_MODEL:?}" BENCH_EXTERNAL_TIMING_LOG="$LOGD/api_timing.jsonl"
  export PYTHONPATH="$WS/rt_external/bootstrap:$WS:${PYTHONPATH:-}"
  say "External /step service ($BENCH_EXTERNAL_MODEL); no local vLLM"
  # Warmup is not scored; the shared gateway occasionally answers with a
  # non-JSON page, so allow three attempts before giving up.
  _wok=0; for _wt in 1 2 3; do
    python3 -m rt_external.step_warmup "$BENCH_EXTERNAL_CONFIG" "$BENCH_EXTERNAL_MODEL" "$LOGD/warmup_timing.jsonl" && { _wok=1; break; }
    sleep 20
  done
  [ "$_wok" = 1 ] || die "external /step warmup failed" 26
else
say "Serving $CKPT_ID ($VLLM_REPLICAS replica(s), tp=$VLLM_TP)"

# All replicas load the same ~17 GB of safetensors off AFS. Started together they
# read it N times in parallel and throttle each other: a 3-replica run measured
# 554 s just for "Loading weights" and blew the readiness timeout. One sequential
# read first puts the files in page cache (the node has ~2 TB of RAM), so the
# replicas then load from memory.
if [ "$VLLM_REPLICAS" -gt 1 ] && [ "${PREWARM_WEIGHTS:-1}" != "0" ]; then
  t0=$(date +%s)
  cat "$MODEL_DIR"/*.safetensors > /dev/null 2>&1 || true
  echo "[PAR] weights prewarmed into page cache in $(( $(date +%s) - t0 ))s"
fi

for r in $(seq 0 $((VLLM_REPLICAS-1))); do
  port=$((VLLM_BASE_PORT + r))
  gpus="$(seq -s, $((r*VLLM_TP)) $(((r+1)*VLLM_TP - 1)))"
  log="$LOGD/vllm_$port.log"
  echo "[PAR] replica $r -> :$port on GPU $gpus ($log)"
  TP_FLAG=""; [ "$VLLM_TP" -gt 1 ] && TP_FLAG="--tensor-parallel-size $VLLM_TP"
  # Keeping reasoning in history needs a patched chat template, which the agent
  # sends per request; vLLM refuses request templates unless trusted (0.17 answers
  # 400 to both /tokenize and /v1/chat/completions). Only pass the flag for those
  # runs so a standard run's server config stays identical to every past batch.
  TRUST_FLAG=""
  if [ -n "${HISTORY_REASONING:-}" ] && [ "${HISTORY_REASONING:-0}" != "0" ]; then
    TRUST_FLAG="--trust-request-chat-template"
  fi
  CUDA_VISIBLE_DEVICES="$gpus" nohup vllm serve "$MODEL_DIR" \
      --served-model-name "$CKPT_ID" \
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
  tries="${VLLM_READY_TRIES:-240}"          # x10s; AFS reads alone can take ~10min
  for i in $(seq 1 "$tries"); do
    kill -0 "$pid" 2>/dev/null || { tail -30 "$LOGD/vllm_$port.log"; die "vLLM :$port died at startup"; }
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "vLLM ready (~${i}0s) :$port"; break; }
    sleep 10
    [ "$i" = "$tries" ] && { tail -30 "$LOGD/vllm_$port.log"; die "vLLM :$port not ready after $((tries/6))min"; }
  done
done

# First real VL generate on a cold vLLM compiles the image-encoder + decode
# graph. 0.8B S3 realtime smoke (job-iw81vr768b): /health was up, then the first
# /v1/chat/completions took ~4.5 min (HTTP 200) while YourVLA's 120s read
# timeout fired, the episode ended on an empty action, and oga_check_render
# flagged DEAD EPISODES. pdq hides that stall because the world is frozen
# and does not charge it back; realtime/lcrt/final-rt all put latency on the
# game clock (realtime = keys-up while the world runs; lcrt = keys-up
# fast-forward; final-rt = hold previous keys for max(0, L-chunk)). Pay the
# compile cost here, with a dummy frame matching --resize, before any episode.
# 35B-A3B cold VL generate is even slower than 0.8B's 4.5 min, so pdq also
# warms: otherwise the first step of every worker is a 180s client timeout
# (empty action) even though the world is frozen.
_rw=1280; _rh=720
  if [ -n "${RESIZE:-}" ]; then
    _rw="${RESIZE%%x*}"; _rh="${RESIZE##*x}"
  fi
  say "Warming VL generate on $VLLM_REPLICAS replica(s) (${_rw}x${_rh}, clock=$CLOCK)"
  python3 - "$VLLM_BASE_PORT" "$VLLM_REPLICAS" "$CKPT_ID" "$_rw" "$_rh" <<'PY' || die "vLLM VL warmup failed; first realtime step would hit the 120s client timeout" 26
import base64, io, json, sys, time, urllib.error, urllib.request
from PIL import Image
base_port, nrep, model, w, h = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
buf = io.BytesIO()
Image.new("RGB", (w, h), (32, 64, 96)).save(buf, format="JPEG", quality=85)
uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
body = json.dumps({
    "model": model,
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "warmup"},
        ]},
        {"role": "assistant", "content": "<think>"},
    ],
    "continue_final_message": True,
    "add_generation_prompt": False,
    "max_tokens": 8,
    "temperature": 0.0,
    "skip_special_tokens": False,
}).encode()
ok = 0
for r in range(nrep):
    port = base_port + r
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        print(f"[PAR] warmup :{port} {time.time()-t0:.1f}s finish={((payload.get('choices') or [{}])[0].get('finish_reason'))} text={text[:60]!r}", flush=True)
        ok += 1
    except Exception as exc:
        print(f"[PAR] warmup :{port} FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
if ok != nrep:
    sys.exit(1)
PY

fi

# -- 6. run the games: longest job first over healthy workers ---------------
# Games that need a fresh UE5 for every episode, not just for every game.
# solo_craft is the known case: on a reused instance its episodes 2-3 open on the
# previous episode's GAME OVER frame, end in 1 step without a single model call,
# and report episode 1's score again, so its n=3 was really n=1 (EVAL_FLOW
# 7b-bis). Restarting costs ~45s per episode and solo_craft sits on the idlest
# worker, so this is free in wall clock. Only listed games are affected, which
# keeps the other six byte-identical to previous runs. "all" / "none" also work.
RPE="${UE5_RESTART_PER_EPISODE:-solo_craft}"
case "$RPE" in
  1|all) RPE_ALL=1; RPE="" ;;
  0|none) RPE_ALL=0; RPE="" ;;
  *)     RPE_ALL=0 ;;
esac
echo "[PAR] fresh UE5 per episode: ${RPE:-$([ "$RPE_ALL" = 1 ] && echo 'all games' || echo none)}"

say "Running $NGAMES games on $NWORK workers"
# Balance by measured game length, not list order: with fewer workers than games
# a round-robin slice could land two 30-minute games on one worker while another
# finished a 30-second game and idled.
PLAN="$LOGD/worker_plan.tsv"
python3 "$SDIR/oga_plan_workers.py" "$NWORK" "${GAME_LIST[@]}" > "$PLAN" \
  || die "worker planning failed" 24
echo "[PAR] worker plan (worker / games / est min per episode, uncontended):"
cat "$PLAN"
SMI_LOG="$LOGD/smi_$(hostname).log"
( while true; do echo "== $(date +%H:%M:%S) =="; \
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 15; done ) > "$SMI_LOG" 2>&1 &
SMI_PID=$!

cd "$ARENA"
export PYTHONPATH="$ARENA:${PYTHONPATH:-}"

# MP4 recording needs imageio_ffmpeg (bundled ffmpeg), which the pod image does
# not carry -- run_arena_eval.sh installed it and this orchestrator inherited the
# --record-video flag without the install, which is why the first parallel batches
# have no episode.mp4. The recorder only reports the missing backend at debug
# level, so every frame fails silently and the run still looks clean; treat the
# dependency as a precondition instead (EVAL_FLOW 4b makes the MP4 mandatory).
python3 -c "import imageio_ffmpeg" 2>/dev/null || pip install -q imageio_ffmpeg 2>&1 | tail -1
python3 -c "import imageio_ffmpeg" 2>/dev/null \
  || die "imageio_ffmpeg missing and pip install failed; every episode.mp4 would be silently empty" 27

declare -a WPIDS=()
for w in $(seq 0 $((NWORK-1))); do
  port="${WORKERS[$w]}"
  adapter="${WORKER_ADAPTERS[$w]}"
  replica=$((w % VLLM_REPLICAS))
  endpoint="${ENDPOINTS[$replica]}"
  CFG_MAP="$LOGD/configs_r$replica.tsv"
  mine=()   # keep defined for `set -u` when a worker drew no games
  read -r -a mine <<< "$(awk -F'\t' -v w="$w" '$1==w{print $2}' "$PLAN")"
  [ ${#mine[@]} -gt 0 ] || continue
  echo "[PAR] worker $w (UE5 :$port adapter=$adapter -> $endpoint): ${mine[*]}"
  (
    first=1
    for g in "${mine[@]}"; do
      # A game inherits whatever the previous one left in this instance. Reusing
      # it is how solo_craft's episodes 2-3 ended up starting on a stale GAME
      # OVER frame (EVAL_FLOW 7b-bis) and scoring a leftover value, so hand every
      # game a fresh process. Costs ~45s, only on workers that hold >1 game, and
      # makes each game byte-identical in setup to the one-UE5-per-game layout.
      if [ "$first" != 1 ]; then
        echo "[PAR] worker $w: restarting UE5 :$port before $g"
        ue5_kill "$port"
        if ! ue5_launch "$port" "$adapter"; then
          warn "UE5 :$port did not come back; skipping ${mine[*]} from $g on"
          tail -6 "$LOGD/warm_$port.log"
          echo 18 > "$LOGD/rc_$g"
          break
        fi
        tail -1 "$LOGD/warm_$port.log"
      fi
      first=0
      cfg="$(awk -F'\t' -v g="$g" '$1==g{print $2}' "$CFG_MAP")"
      glog="$LOGD/game_$g.log"
      echo "[PAR] START $g on :$port -> $glog"
      per_ep=$RPE_ALL
      case " $RPE " in *" $g "*) per_ep=1 ;; esac
      if [ "$per_ep" = 1 ] && [ "$EPISODES" -gt 1 ]; then
        echo "[PAR] $g: one fresh UE5 per episode"
        : > "$glog"      # the episode count below greps this log
        rc=0
        for ep in $(seq 1 "$EPISODES"); do
          if [ "$ep" != 1 ]; then
            echo "[PAR] worker $w: restarting UE5 :$port before $g episode $ep"
            ue5_kill "$port"
            ue5_launch "$port" "$adapter" || { rc=18; break; }
            tail -1 "$LOGD/warm_$port.log"   # the READY line only lands in the file
          fi
          YOURVLA_BASE_URL="$endpoint" python scripts/run_benchmark.py \
              --config "$cfg" \
              --host 127.0.0.1 --port "$port" \
              --clock-mode "$CLOCK" \
              ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
              --episodes 1 \
              --record-video --video-with-thinking >> "$glog" 2>&1
          rc=$?
          [ "$rc" -eq 0 ] || break
        done
      else
        YOURVLA_BASE_URL="$endpoint" python scripts/run_benchmark.py \
            --config "$cfg" \
            --host 127.0.0.1 --port "$port" \
            --clock-mode "$CLOCK" \
            ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
            --episodes "$EPISODES" \
            --record-video --video-with-thinking > "$glog" 2>&1
        rc=$?
      fi
      echo "$rc" > "$LOGD/rc_$g"
      echo "[PAR] END   $g rc=$rc  $(grep -c '\[episode end\]' "$glog" 2>/dev/null || echo 0) episodes"
      [ "$rc" -eq 0 ] || tail -15 "$glog"
    done
  ) &
  WPIDS+=("$!")
done
for p in "${WPIDS[@]}"; do wait "$p"; done
kill "$SMI_PID" 2>/dev/null || true

RC=0
for g in "${GAME_LIST[@]}"; do
  rc="$(cat "$LOGD/rc_$g" 2>/dev/null || echo 99)"
  [ "$rc" -eq 0 ] || { warn "game $g rc=$rc"; RC=1; }
done

# -- 7. aggregate + validity gate -------------------------------------------
if [ "$PORT_SPAN" -gt 1 ]; then
  python3 "$SDIR/oga_aggregate_2p.py" "$ARENA/runs/yourvla" "$CKPT_ID"
else
  python3 "$SDIR/oga_aggregate.py" "$ARENA/runs/yourvla" "$CKPT_ID"
fi
# A render-starved engine still produces a full set of plausible-looking scores,
# so check the movement traces before anyone reads the table above.
python3 "$SDIR/oga_check_render.py" "$ARENA/runs/yourvla" "$CKPT_ID" || true

# The per-instance UE5 / driver logs live in the pod's /tmp and are the only
# record of why an instance died, so keep a copy next to the results.
ARCH="$ARENA/runs/_launch_logs/parallel_${CKPT_ID}_$(hostname)_$(date +%m%d_%H%M%S)"
mkdir -p "$ARCH" && cp "$LOGD"/*.log "$LOGD"/configs_r*.tsv "$ARCH"/ 2>/dev/null || true
echo "[PAR] logs -> $ARCH"

if   [ "$RC" -ne 0 ];             then OUT="FAIL_cfg_rc${RC}"
elif [ "$NWORK" -lt "$PAR" ];     then OUT="OK_partial_ue5_${NWORK}of${PAR}"
else                                   OUT="OK"; fi
record_outcome "$OUT"
echo "[PAR] DONE CKPT=$CKPT CKPT_ID=$CKPT_ID workers=$NWORK/$PAR rc=$RC"
exit "$RC"
