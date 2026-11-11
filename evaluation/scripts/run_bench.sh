#!/usr/bin/env bash
# Run one benchmark on a single 8-GPU node against one merged HF checkpoint.
# Invokes the per-benchmark drivers (bench_eval/*/docker/eval_*.sh and
# openha_minecraft/run_eval_qwen35_cpt.sh) with the fixed evaluation protocol
# described in README.md.
#
# Usage:
#   MODEL_DIR=/path/to/hf_ckpt TAG=my_ckpt bash scripts/run_bench.sh <openha|vizdoom|oga|gameworld>
#
# env:
#   MODEL_DIR  (required) merged HF checkpoint dir (has config.json)
#   TAG        (required) short label, becomes the served-name / result key
#   WS         bench workspace made by scripts/setup_workspace.sh (default /tmp/bench_ws)
#   OUT        OpenHA output root                           (default /tmp/openha_eval_out)
#   SMOKE=1    minimal run that checks the pipeline end to end (scores are not meaningful)
#   OPENHA_RES OpenHA model-input resolution, 480p (default) or 720p
#
# GPU layout overrides (defaults assume 8 GPUs; change only for smaller nodes):
#   VZD_REPLICAS (4)  OGA_REPLICAS (3)  OGA_PAR (5)  GW_PORTS (8088..8095)  OPENHA_DP (8)
set -uo pipefail

BENCH="${1:?usage: run_bench.sh <openha|vizdoom|oga|gameworld>}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the HF checkpoint dir}"
TAG="${TAG:?set TAG, e.g. my_ckpt}"
WS="${WS:-/tmp/bench_ws}"
SMOKE="${SMOKE:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$HERE/.." && pwd)"

[ -f "$MODEL_DIR/config.json" ] || { echo "[run] FATAL: no config.json in $MODEL_DIR"; exit 21; }
export WS_OVERRIDE="$WS" BENCH_CODE_HINT="$WS"
# Compilation caches; keep them on local disk.
export VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton_cache \
       TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache FLASHINFER_WORKSPACE_BASE=/tmp

case "$BENCH" in
# ---------------------------------------------------------------- ViZDoom
# 4 maps x 10 episodes, a5-480p, pdq, free think, projective mouse. 4 vLLM replicas.
vizdoom)
  [ -f "$WS/evaluation/vizdoom/docker/eval_vzd.sh" ] || { echo "run setup_workspace.sh first"; exit 20; }
  EPI=10; EXTRA=()
  [ "$SMOKE" = 1 ] && { EPI=1; EXTRA=(MAX_STEPS=12 RECORD_VIDEO=0); }
  env SERVED_NAME="${TAG}_doom_a5_480p" MODEL_DIR="$MODEL_DIR" \
      MAPS="battle1 battle2 defend_the_center defend_the_line" EPISODES="$EPI" \
      PAR=6 VLLM_REPLICAS="${VZD_REPLICAS:-4}" PROMPT_STYLE=oga TEMP=1 REASONING=1 REASONING_STYLE=free \
      HISTORY=5 HISTORY_REASONING=0 RESIZE=854x480 MOUSE_MODE=projective CLOCK=pdq \
      "${EXTRA[@]}" \
      bash "$WS/evaluation/vizdoom/docker/eval_vzd.sh"
  ;;
# ---------------------------------------------------------------- OmniGameArena
# 7 solo games x 10 episodes, a5-480p, pdq, free think. 3 vLLM + 5 UE5 (one per GPU).
oga)
  [ -x "$WS/evaluation/omnigamearena/omnigamearena_ue5/Linux/OmniGameArena.sh" ] || {
    echo "[run] FATAL: UE5 build missing; download it and rerun scripts/setup_workspace.sh (see README.md)"; exit 20; }
  EPI=10; EXTRA=(); [ "$SMOKE" = 1 ] && { EPI=1; EXTRA=(MAX_STEPS=12); }
  env CKPT="$TAG" SERVED_NAME="${TAG}_oga_a5_480p_e${EPI}" MODEL_DIR="$MODEL_DIR" \
      GAMES="obstacle_run_3d obstacle_run_2d last_stand monster_shoot scene_escape cue_chase solo_craft" \
      GAMES_CONFIG=solo_suite EPISODES="$EPI" REASONING=1 PAR="${OGA_PAR:-5}" VLLM_REPLICAS="${OGA_REPLICAS:-3}" \
      TEMP=1 REASONING_STYLE=free CLOCK=pdq HISTORY=5 HISTORY_REASONING=0 RESIZE=854x480 \
      UE5_RESTART_PER_EPISODE=solo_craft "${EXTRA[@]}" \
      bash "$WS/evaluation/omnigamearena/docker/eval_parallel.sh"
  ;;
# ---------------------------------------------------------------- GameWorld
# 34 games x 5 tasks = 170, a5-480p, paused clock, think on. 8 vLLM (served name /model).
gameworld)
  [ -f "$WS/evaluation/gameworld/run_suite.py" ] || { echo "run setup_workspace.sh first"; exit 20; }
  [ -f "$WS/evaluation/gameworld/games/benchmark/01_2048/index.html" ] || {
    echo "[run] FATAL: GameWorld games missing; git clone https://github.com/gameworld-project/GameWorld-Games.git into bench_eval/gameworld/games/gameworld-games, symlink games/benchmark -> gameworld-games/benchmark (see README.md) and rerun scripts/setup_workspace.sh"; exit 20; }
  SHARDS="g1 g2 g3 g4 g5 g6 g7 g8"; PORTS="${GW_PORTS:-8088 8089 8090 8091 8092 8093 8094 8095}"; EXP=170
  if [ "$SMOKE" = 1 ]; then SHARDS="gsmoke"; PORTS="8088"; EXP=""; fi
  env CKPT="$TAG" SHARDS="$SHARDS" MAXP=3 VLLM_PORTS="$PORTS" \
      SUITE_PREFIX=qwen3-vl-8b-sft8 MODEL_PATH="$MODEL_DIR" RESULTS_ROOT="results/${TAG}_gw_a5_480p" \
      MAXLEN=32768 IMG_LIMIT=8 SFT_KEY_TIMELINE=1 SFT_WASD_REMAP=1 SFT_MEMORY_ROUNDS=5 \
      SFT_RESIZE_HEIGHT=480 SFT_TEMPERATURE=1 SFT_THINK=1 SFT_SKIP_SPECIAL_TOKENS=0 \
      CLOCK_MODE=paused VLLM_WARMUP=1 ${EXP:+EXPECTED_TASKS=$EXP} \
      bash "$WS/evaluation/gameworld/docker/eval_ckpt.sh"
  ;;
# ---------------------------------------------------------------- OpenHA (Minecraft)
# 30-task table (task_list.json) x 5 rollouts, 480p (OPENHA_RES=720p for the 720p variant), optional think, history=3 action-only,
# 600 steps, parallel 30 -- the reference protocol. One vLLM, TP=1 DP=8.
openha)
  OH="$EVAL_ROOT/openha_minecraft"
  OUT="${OUT:-/tmp/openha_eval_out}"; mkdir -p "$OUT/logs"
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 PATH="/usr/lib/jvm/java-8-openjdk-amd64/bin:$PATH"
  export MINESTUDIO_DIR=/tmp/minestudio LP_NUM_THREADS=2 MINESTUDIO_MAX_WAIT=120
  export OPENHA_EVAL_OUTPUT="$OUT"
  PORT=8002; NAME="${TAG}-openha"; OH_RES="${OPENHA_RES:-480p}"
  if ! curl -s "http://localhost:$PORT/v1/models" | grep -q "\"$NAME\""; then
    pkill -9 -f "vllm serve|VLLM::EngineCore|VLLM::Worker|VLLM::APIServer|VLLM::DPCoordinator" 2>/dev/null || true
    sleep 3
    echo "[run] starting vLLM $NAME on :$PORT (log $OUT/logs/serve.log)"
    ( cd "$OH" && MODEL="$MODEL_DIR" SERVED_NAME="$NAME" PORT=$PORT TP=1 DP="${OPENHA_DP:-8}" \
        MAX_IMAGES=16 MAX_NUM_SEQS=32 GPU_MEM_UTIL=0.90 \
        nohup bash serve_qwen3vl.sh >"$OUT/logs/serve.log" 2>&1 & )
    for i in $(seq 1 120); do
      curl -s "http://localhost:$PORT/v1/models" | grep -q "\"$NAME\"" && break
      sleep 10
    done
    curl -s "http://localhost:$PORT/v1/models" | grep -q "\"$NAME\"" || { echo "[run] vLLM not ready, see serve.log"; exit 1; }
  fi
  ROLL=5; PAR=30; STEPS=600; TASKS=""; [ "$SMOKE" = 1 ] && { ROLL=1; PAR=2; STEPS=40; TASKS="kill_entity:sheep,mine_block:dirt"; }
  cd "$OH" && env RES="$OH_RES" GUI_MOUSE_MAPPING=calibrated \
      MODEL_ID="$NAME" MODEL_PATH="qwen3.5-$NAME" MODEL_IP=localhost MODEL_PORT=$PORT \
      REASONING_MODE=optional THINK_TAG_PREFILLED=0 PROMPT_LANG=en INSTR_LANG=en GUI_KEY_MAP=wasd \
      HISTORY_MODE=full MAXHIST=3 HISTORY_CONTENT=action_only STOP_MODE=string \
      CATEGORIES=all TASKS="$TASKS" ROLLOUTS=$ROLL MAX_STEPS=$STEPS PARALLEL=$PAR \
      NO_RENDER=1 ROLLOUT_TIMEOUT=2000 RECORD_PATH="$OUT/${TAG}_${OH_RES}_30tasks" \
      bash run_eval_qwen35_cpt.sh 2>&1 | tee "$OUT/logs/eval_${TAG}.log"
  echo "[run] stop the server afterwards: pkill -f 'vllm serve'"
  ;;
*) echo "unknown bench: $BENCH (openha|vizdoom|oga|gameworld)"; exit 2 ;;
esac
