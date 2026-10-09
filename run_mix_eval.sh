#!/bin/zsh
cd "$(dirname "$0")"
PY=.venv/bin/python
R=gen_mix_yolo26n-seg_s0
until [ -f runs/$R/DONE ]; do sleep 60; done
$PY 06_evaluate.py --run $R --regime gen_mix > runs/eval_$R.log 2>&1
$PY 13_revision_eval.py --det yolo:$R 2>&1 | grep -v Warn
$PY 17_event_grid.py 2>&1 | tail -1
$PY 16_operating_points.py 2>&1 | grep -E "mix|detector"
$PY 20_split_calibration.py 2>&1 | grep mix
$PY 23_fresh_eval.py --det yolo:$R 2>&1 | grep -v Warn
$PY 15_summarise.py > /dev/null 2>&1
echo MIX EVAL DONE
