#!/usr/bin/env bash
# Usage: tools/run.sh <args to arena.py>   (from the TFI folder)
K="/c/Users/SAI BALAJI/OneDrive/Desktop/catan-hackathon-agent"
PY="/c/Users/SAI BALAJI/.catan-venv/Scripts/python.exe"
export PYTHONHASHSEED=0 PYTHONPATH="$K"
exec "$PY" "$(dirname "$0")/arena.py" "$@"
