#!/bin/zsh
# Independent calibration set (external review, 13 Oct 2026): count false
# alarms of every network except the ablations, and of the baselines, on
# data/calib, then compare calibration sources (28_calibration_set.py).
setopt nullglob
cd "$(dirname "$0")"
PY=.venv/bin/python
while [ $(ls data/calib/*.npz 2>/dev/null | wc -l) -lt 300 ]; do sleep 60; done
for f in runs/revision/yolo_*.json; do
  n=$(basename $f .json); n=${n#yolo_}
  case $n in *nogreen*|*nohalo*|*nosharp*|*noswir*) continue;; esac
  [ -f runs/revision_calib/yolo_$n.json ] || $PY 23_fresh_eval.py --set calib --det yolo:$n 2>&1 | grep -v Warn
done
for n in gen_unet_s0 gen_v2_unet_s0 gen_v2_unet384_s0; do
  [ -f runs/revision_calib/unet_$n.json ] || $PY 23_fresh_eval.py --set calib --det unet:$n 2>&1 | grep -v Warn
done
for d in zscore rx irmad rf:gen rf:gen_v2; do
  n=${d/:/_}
  [ -f runs/revision_calib/$n.json ] || $PY 23_fresh_eval.py --set calib --det $d 2>&1 | grep -v Warn
done
$PY 28_calibration_set.py
echo CALIB DONE
