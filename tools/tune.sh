#!/usr/bin/env bash
K="/c/Users/SAI BALAJI/OneDrive/Desktop/catan-hackathon-agent"
PY="/c/Users/SAI BALAJI/.catan-venv/Scripts/python.exe"
export PYTHONHASHSEED=0 PYTHONPATH="$K"
exec "$PY" "$(dirname "$0")/tune.py" "$@"
