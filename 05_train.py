"""Train YOLO (YOLO26 by default, YOLOv8 for reference) segmentation on the synthetic set.

The three channels are physical differences, so every colour augmentation
is switched off; only geometric ones remain. Chips are upsampled from 256
to 384 because a 0.5 ha plume is about 7 x 7 pixels at 10 m, near the
coarsest YOLO stride otherwise. 512 with the s model does not fit in the
8 GB of the M1 (swap thrash, about 4 h per epoch); 384 with the n model
runs at about 3.5 min per epoch.

    .venv/bin/python 05_train.py --regime gen --model yolo26n-seg.pt --epochs 60
"""
import argparse

import torch
from ultralytics import YOLO

from config import DATA, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="gen")
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from weights/last.pt")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else \
        (0 if torch.cuda.is_available() else "cpu")
    name = f"{args.regime}_{args.model.split('.')[0]}_s{args.seed}"
    out = ROOT / "runs" / name
    if args.resume and (out / "weights" / "last.pt").exists():
        YOLO(str(out / "weights" / "last.pt")).train(resume=True)
        (out / "DONE").write_text("finished\n")
        return
    YOLO(args.model).train(
        data=str(DATA / f"yolo_{args.regime}" / "data_test_narli.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        seed=args.seed, deterministic=True, project=str(ROOT / "runs"), name=name,
        exist_ok=True, patience=20, workers=args.workers,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,          # channels are physical
        fliplr=0.5, flipud=0.5, degrees=0.0, mosaic=0.5, mixup=0.0,
        scale=0.2, translate=0.1,
    )
    # written only when training returns normally, so a killed run is never
    # mistaken for a finished one
    (out / "DONE").write_text("finished\n")


if __name__ == "__main__":
    main()
