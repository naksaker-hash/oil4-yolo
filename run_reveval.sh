#!/bin/zsh
# Evaluates every detector on all real sets once the extra imagery is in;
# new YOLO runs are picked up as the revision queue finishes them.
cd "$(dirname "$0")"
setopt nullglob
PY=.venv/bin/python
while [ ! -f runs/fetch2.DONE ]; do sleep 120; done
for d in zscore rx irmad rf:gen rf:gen_v2; do
  n=${d/:/_}; [ -f "runs/revision/$n.json" ] || { echo "$(date) eval $d"; nice -n 10 $PY 13_revision_eval.py --det $d > "runs/reveval_$n.log" 2>&1; }
done
while true; do
  for r in runs/gen*_s[0-9] runs/siverek_v2*_s[0-9] runs/narli_*_s[0-9]; do
    [ -f "$r/DONE" ] && [ -f "$r/eval_summary.json" ] || continue
    run=${r#runs/}
    [ -f "runs/revision/yolo_$run.json" ] && continue
    echo "$(date) eval $run"; nice -n 10 $PY 13_revision_eval.py --det yolo:$run > "runs/reveval_$run.log" 2>&1
  done
  grep -q "REVISION DONE" runs/revision.log && break
  sleep 600
done
echo "$(date) REVEVAL DONE"
