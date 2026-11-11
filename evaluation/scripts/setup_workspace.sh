#!/usr/bin/env bash
# Build a runnable bench workspace from this repository.
#
# bench_eval's pod scripts refuse to run unless they find a workspace that
#   (a) contains .bench-workspace-id = bench-workspace-v1, and
#   (b) has the code at $WS/evaluation/{vizdoom,omnigamearena,gameworld}.
# The repository keeps the code under bench_eval/<bench>, so this script
# lays it out the way the scripts expect and links the OGA UE5 build in.
#
# Usage:
#   bash scripts/setup_workspace.sh [WS] [UE5_DIR]
#     WS       workspace to create      (default /tmp/bench_ws)
#     UE5_DIR  extracted UE5 build root, the dir that contains Linux/OmniGameArena.sh
#              (default /tmp/omnigamearena_ue5; skip OGA if missing)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$HERE/.." && pwd)"
SRC="$EVAL_ROOT/bench_eval"
WS="${1:-/tmp/bench_ws}"
UE5_DIR="${2:-/tmp/omnigamearena_ue5}"

[ -d "$SRC/vizdoom" ] || { echo "[setup] FATAL: $SRC/vizdoom not found"; exit 1; }

mkdir -p "$WS/evaluation"
for b in vizdoom omnigamearena gameworld; do
  rm -rf "${WS:?}/evaluation/$b"
  cp -a "$SRC/$b" "$WS/evaluation/$b"
  ln -sfn "evaluation/$b" "$WS/$b"
done
echo "bench-workspace-v1" > "$WS/.bench-workspace-id"

if [ -x "$UE5_DIR/Linux/OmniGameArena.sh" ]; then
  ln -sfn "$UE5_DIR" "$WS/evaluation/omnigamearena/omnigamearena_ue5"
  echo "[setup] UE5 build linked: $UE5_DIR"
else
  echo "[setup] WARN: no UE5 build at $UE5_DIR/Linux/OmniGameArena.sh -- OGA will not run"
fi

echo "[setup] workspace ready: $WS"
echo "export WS_OVERRIDE=$WS BENCH_CODE_HINT=$WS"
