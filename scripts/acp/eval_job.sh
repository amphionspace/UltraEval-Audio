#!/usr/bin/env bash
# Runs on the ACP node: one evaluation command of the repository, as the repository owner.
# Usage: eval_job.sh REPO LOG_DIR PYTHON_ARGS...   (e.g. -m scripts.run_seed_tts --run-dir ...)
set -euo pipefail
REPO="$1"; LOG_DIR="$2"; shift 2
: "${PYTHON_BIN:?PYTHON_BIN is required}"
if [[ $EUID -eq 0 ]]; then
  # Drop to the owner of the repository so outputs stay writable from the dev host.
  exec "$PYTHON_BIN" -c '
import os, sys
st = os.stat(sys.argv[1])
os.setgroups([]); os.setgid(st.st_gid); os.setuid(st.st_uid)
os.execv("/bin/bash", ["bash", *sys.argv[2:]])' "$REPO" "$0" "$REPO" "$LOG_DIR" "$@"
fi
# quarkfs renames are unreliable from nodes; keep JIT and tool caches node-local.
export HOME=/tmp/eval-home XDG_CACHE_HOME=/tmp/eval-cache TRITON_CACHE_DIR=/tmp/eval-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/eval-cache/inductor
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$LOG_DIR"
LOG="$LOG_DIR/acp-node-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "uid=$(id -u) host=$(hostname) gpus=$(nvidia-smi -L | wc -l) args=$*" | tee "$LOG"
cd "$REPO"
exec "$PYTHON_BIN" -u "$@" >> "$LOG" 2>&1
