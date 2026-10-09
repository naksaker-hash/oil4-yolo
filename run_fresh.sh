#!/bin/zsh
setopt nullglob
cd "$(dirname "$0")"
PY=.venv/bin/python
for f in runs/revision/yolo_*.json; do
  n=$(basename $f .json); n=${n#yolo_}
  [ -f runs/revision_fresh/yolo_$n.json ] || $PY 23_fresh_eval.py --det yolo:$n 2>&1 | grep -v Warn
done
for d in zscore rx irmad rf:gen rf:gen_v2; do
  n=${d/:/_}
  [ -f runs/revision_fresh/$n.json ] || $PY 23_fresh_eval.py --det $d 2>&1 | grep -v Warn
done
echo FRESH DONE
