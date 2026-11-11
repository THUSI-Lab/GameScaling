#!/usr/bin/env bash
# Pin every eval timestamp to UTC+8, sourced by the eval entry points.
#
# Pods default to UTC while we read results in UTC+8, so a run directory
# named 20260803_073243 was actually 15:32 local -- log lines, run dir names,
# the per-node run log and the date column of the results table (derived from the
# run dir name) all disagreed with the clock people quote in reviews.
#
# Exporting TZ covers shell `date`, Python `datetime.now()` and vLLM's log
# formatter in one place. The image may ship without tzdata, so fall back to the
# POSIX spec -- note its sign is inverted there: "CST-8" means 8h EAST of UTC.
if [ -z "${TZ:-}" ]; then
  if [ -e /usr/share/zoneinfo/Asia/Shanghai ]; then
    export TZ="Asia/Shanghai"
  else
    export TZ="CST-8"
  fi
fi
