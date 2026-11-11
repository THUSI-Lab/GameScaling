#!/usr/bin/env bash
# ============================================================================
# Launch and warm ONE UE5 instance on a given port + Vulkan adapter, then leave
# it running detached. Several copies of this script run concurrently inside one
# pod: the LD_PRELOAD shim (ue5_port_shim.so) moves the hardcoded RemoteInput
# port 12345 to $PORT, so no network namespace -- and therefore no CAP_NET_ADMIN,
# which a k8s pod does not have -- is needed to keep instances apart.
#
# The H200's thin graphics slice makes UE5's first heavy render land near the
# ~10s VulkanRHI fence, so roughly half of all launches die on the first frame.
# Launch+warm is therefore retried per instance; a dead instance only costs its
# own slot instead of the whole job.
#
# env (required): PORT ADAPTER UE5_DIR SHIM
# env (optional): LOGD TRIES WARM_MAPS WARMUP_SLEEP BOOT_WAIT PORT_SPAN
#
# PvP/Coop maps serve player 2 from a second listener on 12346, so those runs set
# PORT_SPAN=2 and the instance owns [PORT, PORT+1]. Callers must then stride their
# port allocation by the same span.
# exit: 0 = listening + rendered (pgid in $LOGD/ue5_$PORT.pgid), 17 = never warmed
# ============================================================================
set -uo pipefail

PORT="${PORT:?set PORT}"
ADAPTER="${ADAPTER:?set ADAPTER}"
UE5_DIR="${UE5_DIR:?set UE5_DIR}"
SHIM="${SHIM:?set SHIM (path to ue5_port_shim.so)}"
LOGD="${LOGD:-/tmp/oga_parallel}"
PORT_SPAN="${PORT_SPAN:-1}"
TRIES="${TRIES:-4}"
BOOT_WAIT="${BOOT_WAIT:-300}"
WARMUP_SLEEP="${WARMUP_SLEEP:-30}"
SHOT_PY="${SHOT_PY:-$(cd "$(dirname "$0")" && pwd)/ue5_shot.py}"

mkdir -p "$LOGD"
LOG="$LOGD/ue5_$PORT.log"
PGIDF="$LOGD/ue5_$PORT.pgid"
tag() { printf '[ue5:%s] %s\n' "$PORT" "$*"; }

[ -f "$SHIM" ] || { tag "FATAL shim missing: $SHIM"; exit 3; }
[ -x "$UE5_DIR/OmniGameArena.sh" ] || { tag "FATAL launcher missing: $UE5_DIR"; exit 4; }

# UE5 refuses to run as root, and in a k8s pod the pod is root: drop to the uid owning
# the build. That uid needs a writable HOME/XDG_RUNTIME_DIR, read access to the
# shim, and traverse (o+x, not read) on every ancestor of the build so ld.so can
# resolve the absolute LD_LIBRARY_PATH entries.
UE5_UID="$(stat -c %u "$UE5_DIR/OmniGameArena.sh")"; [ "$UE5_UID" = "0" ] && UE5_UID=1000
# Grant only the supplementary groups needed to traverse private mount ancestors.
# UE5 remains non-root; shared directory permissions are left unchanged.
UE5_GROUPS="$(python3 - "$UE5_DIR" <<'PYGROUP'
from pathlib import Path
import stat, sys
root = Path(sys.argv[1]).resolve()
groups = set()
for directory in (root, *root.parents):
    info = directory.stat()
    if not (info.st_mode & stat.S_IXOTH) and info.st_mode & stat.S_IXGRP:
        groups.add(info.st_gid)
print(",".join(str(group) for group in sorted(groups)))
PYGROUP
)"
UE5_GROUP_FLAGS="--clear-groups"
[ -n "$UE5_GROUPS" ] && UE5_GROUP_FLAGS="--groups=$UE5_GROUPS"
tag "non-root uid=$UE5_UID extra mount groups=${UE5_GROUPS:-none}"
UE5_HOME="/tmp/ue5home_${UE5_UID}_$PORT"
mkdir -p "$UE5_HOME"; chown -R "$UE5_UID:$UE5_UID" "$UE5_HOME" 2>/dev/null || true
UE5_LIBS="$UE5_DIR/Engine/Binaries/Linux:$UE5_DIR/Engine/Binaries/ThirdParty/Vulkan/Linux:$UE5_DIR/OmniGameArena/Binaries/Linux"

_start() {
  rm -f "$PGIDF"
  # setsid makes this instance its own process group, so the orchestrator can
  # kill exactly this UE5 tree without touching its siblings.
  setsid bash -c "cd '$UE5_DIR' && env HOME='$UE5_HOME' XDG_RUNTIME_DIR='$UE5_HOME' \
      LD_LIBRARY_PATH='$UE5_LIBS' LD_PRELOAD='$SHIM' UE5_PORT_OVERRIDE=$PORT \
      UE5_PORT_SPAN=$PORT_SPAN \
      setpriv --reuid=$UE5_UID --regid=$UE5_UID $UE5_GROUP_FLAGS \
      ./OmniGameArena.sh -RenderOffScreen -graphicsadapter=$ADAPTER -nosound \
      -unattended -stdout -FullStdOutLogOutput ${UE5_EXTRA_FLAGS:-}" \
      </dev/null >>"$LOG" 2>&1 &
  local pgid=$!
  echo "$pgid" > "$PGIDF"
  tag "launched pgid=$pgid adapter=$ADAPTER home=$UE5_HOME"
  for _ in $(seq 1 $((BOOT_WAIT/5))); do
    (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && { tag "listening"; return 0; }
    kill -0 "$pgid" 2>/dev/null || { tag "died during boot"; tail -8 "$LOG"; return 1; }
    sleep 5
  done
  tag "not listening after ${BOOT_WAIT}s"; return 1
}

_stop() {
  local pgid; pgid="$(cat "$PGIDF" 2>/dev/null || true)"
  [ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null || true
  rm -f "$PGIDF"
}

_warm() {
  local pgid; pgid="$(cat "$PGIDF" 2>/dev/null || true)"
  # Survive the boot render (where the fence timeout hits), then force a
  # throwaway screenshot to prove the pipeline is compiled and stable.
  for w in $(seq 1 "$WARMUP_SLEEP"); do
    kill -0 "$pgid" 2>/dev/null || { tag "died during boot render at ~${w}s"; return 1; }
    sleep 1
  done
  PORT="$PORT" MAPS="${WARM_MAPS:-}" WARMUP_SLEEP="$WARMUP_SLEEP" python3 "$SHOT_PY" || {
    tag "warm screenshot failed"; return 1; }
  kill -0 "$pgid" 2>/dev/null || { tag "died at warm screenshot"; return 1; }
  tag "warm OK"; return 0
}

for attempt in $(seq 1 "$TRIES"); do
  tag "attempt $attempt/$TRIES"
  if _start && _warm; then
    tag "READY port=$PORT adapter=$ADAPTER"
    exit 0
  fi
  _stop
  sleep 5
done
tag "FAILED to warm after $TRIES attempts (log $LOG)"
exit 17
