#!/usr/bin/env bash
# Shared default output directory for OpenHA evaluation runs.
OPENHA_EVAL_OUTPUT=${OPENHA_EVAL_OUTPUT:-./openha_eval_output}
mkdir -p "$OPENHA_EVAL_OUTPUT"
