"""Uncertainty of the IoU of each detected spill (round 4 review).

The reference outline is uncertain by about a pixel (coregistration of 0.25
to 0.75 pixel, digitising). For every network that detected a spill at its
fixed threshold, the IoU of the detecting patch is recomputed against the
outline shifted by one pixel in each of eight directions and eroded or
dilated by one pixel, and the range is reported.

    .venv/bin/python 27_iou_uncertainty.py
"""
import json

import numpy as np
from scipy import ndimage as ndi

import revlib as R
from config import DATA, ROOT

OUT = ROOT / "runs" / "revision"
NETS = [f"gen_v2_yolo26n-seg_s{s}" for s in range(3)] + [
    "gen_v2_yolov8n-seg_s0", "gen_v2_yolo11n-seg_s0", "gen_v2_d1_yolo26n-seg_s0", "gen_v2_yolo26n-seg_scratch_s0",
    "narli_yolo26n-seg_s0", "narli_v2_yolo26n-seg_s0", "siverek_yolo26n-seg_s0", "siverek_v2_yolo26n-seg_s0"]


def variants(lab):
    out = [lab, ndi.binary_erosion(lab), ndi.binary_dilation(lab)]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out.append(np.roll(np.roll(lab, dy, 0), dx, 1))
    return out


def main():
    ev = {}
    for e in ("narli", "siverek"):
        pre, post, lab, _, _ = R.load_pair(DATA / "real" / f"{e}_event.npz", coreg=True)
        ev[e] = (R.pair_input(pre, post), lab)
    res = {}
    for run in NETS:
        th = json.load(open(OUT / f"yolo_{run}.json"))["thresholds"]["synthetic_f1_same_rule"]
        det = R.YoloDet(run)
        r = {}
        for e, (img, lab) in ev.items():
            s = det.score(img)
            pts = R.patches(s >= th, s)
            j = R.judge(pts, lab, "standard")
            if not j["hit"]:
                r[e] = None
                continue
            m = [m for sc, m in pts if round(sc, 4) == j["score"]][0]
            ious = [float((m & v).sum() / (m | v).sum()) for v in variants(lab)]
            r[e] = {"iou": j["iou"], "min": round(min(ious), 3), "max": round(max(ious), 3)}
        res[run] = r
        print(run, r, flush=True)
    R.save_json(res, OUT / "iou_uncertainty.json")


if __name__ == "__main__":
    main()
