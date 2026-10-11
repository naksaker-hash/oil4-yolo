"""Transplant test with a growing footprint (follows 25_diagnose.py).

The observed per pixel ratio after to before the event is applied to the
image after each null pair of the same chip inside the reference outline
dilated by r pixels, for r = 0, 5, 10, 20 and 40 and for the whole chip.
If detection by a network collapses at a small r, the spill's own weaker
surroundings (its oiled margin) defeat it; if only for the whole chip, the
rest of the event pair does.

    .venv/bin/python 26_transplant_rings.py
"""
import json

import numpy as np
from scipy import ndimage as ndi

import realio
import revlib as R
from config import DATA, ROOT

OUT = ROOT / "runs" / "revision"
NETS = [f"gen_yolo26n-seg_s{s}" for s in range(3)] + [f"gen_v2_yolo26n-seg_s{s}" for s in range(3)] + \
       [f"gen_dr_yolo26n-seg_s{s}" for s in range(3)]
RINGS = [0, 5, 10, 20, 40, "chip"]


def main():
    sets = {}
    for e in ("narli", "siverek"):
        ev = realio.load(DATA / "real" / f"{e}_event.npz", coreg=True)
        lab = ev["mask"].astype(bool)
        ratio = np.clip(ev["post"] / np.maximum(ev["pre"], 1e-4), 0.05, 3.0)
        nulls = [R.load_pair(f, coreg=True)[:2] for f in sorted((DATA / "real").glob(f"{e}_null*.npz"))]
        for r in RINGS:
            zone = np.ones_like(lab) if r == "chip" else (ndi.binary_dilation(lab, iterations=r) if r else lab)
            rr = np.where(zone[None], ratio, 1.0)
            sets[(e, r)] = (lab, [R.pair_input(pre, (post * rr).astype(np.float32)) for pre, post in nulls])
    res = {}
    for run in NETS:
        th = json.load(open(OUT / f"yolo_{run}.json"))["thresholds"]["synthetic_f1_same_rule"]
        det = R.YoloDet(run)
        out = {}
        for (e, r), (lab, imgs) in sets.items():
            n = 0
            for x in imgs:
                s = det.score(x)
                n += R.judge(R.patches(s >= th, s), lab, "standard")["hit"]
            out[f"{e}_{r}"] = f"{n}/{len(imgs)}"
        res[run] = out
        print(run, out, flush=True)
    R.save_json(res, OUT / "transplant_rings.json")


if __name__ == "__main__":
    main()
