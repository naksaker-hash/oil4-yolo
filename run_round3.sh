#!/bin/zsh
# Round 3 review queue. Resumable: finished steps are skipped.
setopt nullglob
cd "$(dirname "$0")"
PY=.venv/bin/python
synth () { local d=$1; shift
  [ -f "data/yolo_$d/data_test_narli.yaml" ] || { echo "$(date) building $d"; $PY 04_synth.py "$@" > runs/synth_$d.log 2>&1 || exit 1; } }
evalall () { local spec=$1 name=$2
  [ -f runs/revision/$name.json ] || $PY 13_revision_eval.py --det $spec 2>&1 | grep -v Warn | tail -1
  [ -f runs/revision_fresh/$name.json ] || $PY 23_fresh_eval.py --det $spec 2>&1 | grep -v Warn | tail -1; }
unet () { local r=$1
  [ -f runs/${r}_unet_s0/DONE ] || { echo "$(date) unet $r"; $PY 24_unet.py --regime $r --seed 0 > runs/train_${r}_unet_s0.log 2>&1 || exit 1; }
  evalall unet:${r}_unet_s0 unet_${r}_unet_s0; }
yolo () { local regime=$1 seed=$2 extra=$3
  local run="${regime}_yolo26n-seg${extra:+_scratch}_s$seed"
  if [ ! -f runs/$run/DONE ]; then
    echo "$(date) training $run"
    $PY 05_train.py --regime $regime --model yolo26n-seg.pt --epochs 60 --seed $seed ${extra:+--scratch} --resume > runs/train_$run.log 2>&1
    [ -f runs/$run/DONE ] || { echo "$(date) $run did NOT finish"; exit 1; }
  fi
  [ -f runs/$run/eval_summary.json ] || $PY 06_evaluate.py --run $run --regime $regime > runs/eval_$run.log 2>&1
  evalall yolo:$run yolo_$run; }
unet gen_v2; unet gen
synth gen_drs --profile drs --copies 3; yolo gen_drs 0
yolo gen_v2 0 scratch
synth narli_v2 --regime narli --profile v2 --copies 3; yolo narli_v2 0
synth siverek --regime siverek --profile v1 --copies 3; yolo siverek 0
yolo gen_drs 1; yolo gen_drs 2
$PY 17_event_grid.py > /dev/null 2>&1; $PY 16_operating_points.py > runs/op_round3.log 2>&1
$PY 20_split_calibration.py > runs/split_round3.log 2>&1; $PY 15_summarise.py > /dev/null 2>&1
echo "$(date) ROUND3 DONE"
