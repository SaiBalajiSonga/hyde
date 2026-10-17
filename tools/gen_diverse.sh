#!/usr/bin/env bash
# Diverse bot-vs-bot training data: our agents vs F and F variants, and F variants vs each other.
cd "$(dirname "$0")/.."
K="/c/Users/SAI BALAJI/OneDrive/Desktop/catan-hackathon-agent"
PY="/c/Users/SAI BALAJI/.catan-venv/Scripts/python.exe"
export PYTHONHASHSEED=0 PYTHONPATH="$K"
C="$PWD/tools/cfg"
run() { "$PY" tools/gen_data.py --games 1000 --workers 20 --eps 0.02 --seed0 "$1" --a "$2" --b "$3" --out "tools/data/div_$4.npz" 2>&1 | tail -1; }
run 450000 "$C/v3_mlp.json"   "$C/fv/F_build.json"     v3_Fbuild
run 451000 "$C/v3_mlp.json"   "$C/fv/F_dev.json"       v3_Fdev
run 452000 "$C/v3_mlp.json"   "$C/fv/F_road.json"      v3_Froad
run 453000 "$C/v3_mlp.json"   "$C/fv/F_city.json"      v3_Fcity
run 454000 "$C/v3_mlp.json"   "$C/fv/F_contender.json" v3_Fcont
run 455000 "$C/v6_devrush.json" F                      v6_F
run 456000 "$C/v8_vp.json"    F                        v8_F
run 457000 "$C/fv/F_build.json" "$C/fv/F_dev.json"     Fbuild_Fdev
run 458000 "$C/fv/F_city.json"  "$C/fv/F_road.json"    Fcity_Froad
run 459000 "$C/fv/F_contender.json" F                  Fcont_F
echo ALLDONE
