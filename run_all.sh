#!/usr/bin/env bash
# Full pipeline of the paper, sequential. Each training run takes about
# 3.5 h on an 8 GB Apple M1 (MPS); a CUDA GPU is much faster.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

# 1. real event and null chips, plume masks, measured attenuation k
$PY 02_fetch_real.py
# 2. clean background pairs (random, so the set differs between runs;
#    the paper used 1,370 chips drawn with several seeds)
for s in 1 2 3 4 5 6; do $PY 03_fetch_train.py --n 250 --seed $s; done
# 3. synthetic datasets
$PY 04_synth.py --regime gen --copies 3                          # v1
$PY 04_synth.py --regime gen --copies 3 --profile v2             # v2
for a in sharp halo green swir; do
  $PY 04_synth.py --regime gen --copies 3 --profile v2 --ablate $a
done

train_eval () {  # regime model seed
  run="$1_${2%.pt}_s$3"
  $PY 05_train.py --regime "$1" --model "$2" --epochs 60 --seed "$3" --resume
  $PY 06_evaluate.py --run "$run" --regime "$1"
  $PY 07_analysis.py --run "$run" --regime "$1"
}
# 4. three seeds per generator, two other YOLO generations, four ablations
for s in 0 1 2; do train_eval gen yolo26n-seg.pt $s; train_eval gen_v2 yolo26n-seg.pt $s; done
train_eval gen_v2 yolov8n-seg.pt 0
train_eval gen_v2 yolo11n-seg.pt 0
for a in sharp halo green swir; do train_eval gen_v2_no$a yolo26n-seg.pt 0; done
# 5. figures
$PY make_figs.py
