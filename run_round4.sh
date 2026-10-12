#!/bin/zsh
# Round 4 review queue: U-Net at the 384 px input of the YOLO networks. Resumable.
setopt nullglob
cd "$(dirname "$0")"
PY=.venv/bin/python
synth () { local d=$1; shift
  [ -f "data/yolo_$d/data_test_narli.yaml" ] || { echo "$(date) building $d"; $PY 04_synth.py "$@" > runs/synth_$d.log 2>&1 || exit 1; } }
evalall () { local spec=$1 name=$2
  [ -f runs/revision/$name.json ] || $PY 13_revision_eval.py --det $spec 2>&1 | grep -v Warn | tail -1
  [ -f runs/revision_fresh/$name.json ] || $PY 23_fresh_eval.py --det $spec 2>&1 | grep -v Warn | tail -1; }
unet () { local r=$1
  [ -f runs/${r}_unet384_s0/DONE ] || { echo "$(date) unet384 $r"; $PY 24_unet.py --regime $r --seed 0 --size 384 --batch 8 > runs/train_${r}_unet384_s0.log 2>&1 || exit 1; }
  evalall unet:${r}_unet384_s0 unet_${r}_unet384_s0; }
unet gen_v2
synth gen --regime gen --profile v1 --copies 3; unet gen
$PY 17_event_grid.py > /dev/null 2>&1; $PY 16_operating_points.py > runs/op_round4.log 2>&1
$PY 20_split_calibration.py > runs/split_round4.log 2>&1; $PY 15_summarise.py > /dev/null 2>&1
echo "$(date) ROUND4 DONE"
