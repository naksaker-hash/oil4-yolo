#!/bin/zsh
# Revision queue (reviewer requests). Resumable: DONE runs are skipped.
cd "$(dirname "$0")"
PY=.venv/bin/python
synth () {  # dataset dir name, then 04_synth args
  local d=$1; shift
  if [ ! -f "data/yolo_$d/data_test_narli.yaml" ]; then
    echo "$(date) building $d"; $PY 04_synth.py "$@" > "runs/synth_$d.log" 2>&1 || { echo "synth $d failed"; exit 1; }
  fi
}
train () {  # regime model seed
  local run="$1_${2%.pt}_s$3"
  if [ ! -f "runs/$run/DONE" ]; then
    echo "$(date) training $run"
    $PY 05_train.py --regime "$1" --model "$2" --epochs 60 --seed "$3" --resume > "runs/train_$run.log" 2>&1
    [ -f "runs/$run/DONE" ] || { echo "$(date) $run did NOT finish, stopping"; exit 1; }
  fi
  if [ ! -f "runs/$run/eval_summary.json" ]; then
    echo "$(date) evaluating $run"; $PY 06_evaluate.py --run "$run" --regime "$1" > "runs/eval_$run.log" 2>&1
  fi
}
M=yolo26n-seg.pt
# 1 cross event: generator built from one event only, tested on the other
synth siverek_v2 --regime siverek --profile v2 --copies 3; train siverek_v2 $M 0
synth narli --regime narli --profile v1 --copies 3;        train narli $M 0
# 2 domain randomisation, no event information
synth gen_dr --profile dr --copies 3; for s in 0 1 2; do train gen_dr $M $s; done
# 3 native 20 m implanting
synth gen_v2_n20 --profile v2 --native20 --copies 3; train gen_v2_n20 $M 0
# 4 re-drawn synthetic data (dataset variance)
synth gen_v2_d1 --profile v2 --seed 1 --copies 3; train gen_v2_d1 $M 0
synth gen_d1 --profile v1 --seed 1 --copies 3;    train gen_d1 $M 0
# 5 more seeds per ablation
for s in 1 2; do for a in halo swir green sharp; do train gen_v2_no$a $M $s; done; done
echo "$(date) REVISION DONE"
