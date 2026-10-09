"""False alarms on the fresh plains (19_fetch_fresh.py), which played no
part in training, model selection, the fixed thresholds or the null
calibration. Each detector is read on a threshold grid, and at its fixed,
matched and null calibrated thresholds taken from runs/revision.

    .venv/bin/python 23_fresh_eval.py --det yolo:gen_v2_yolo26n-seg_s0
"""
import argparse
import importlib.util
import json

import numpy as np

import revlib as R
from config import DATA, ROOT

_spec = importlib.util.spec_from_file_location("ev", ROOT / "13_revision_eval.py")
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)
OUT = ROOT / "runs" / "revision_fresh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    det, name = EV.make_detector(args.det)
    rev = json.load(open(ROOT / "runs" / "revision" / f"{name}.json"))
    op = json.load(open(ROOT / "runs" / "revision" / "operating_points.json")).get(name, {})
    th = rev["thresholds"]
    fixed = th.get("synthetic_f1_same_rule", th.get("conventional", th.get("synthetic_f1")))
    pts = {"fixed": fixed, "matched": op.get("both_thr"),
           "null_cal": (op.get("null_cal_0.01") or {}).get("thr")}
    grid = np.array(rev["grid"])
    files = sorted((DATA / "fresh").glob("*.npz"))
    counts = []
    at = {k: [] for k in pts}
    for f in files:
        pre, post, _, _, _ = R.load_pair(f, coreg=True)
        s = det.score(R.pair_input(pre, post), pre, post)
        counts.append([len(R.patches(s >= t, s)) for t in grid])
        for k, t in pts.items():
            if t is not None:
                at[k].append(len(R.patches(s >= t, s)))
    res = {"detector": name, "n_pairs": len(files), "thresholds": pts, "grid": grid.tolist(),
           "fa_grid_km2": (np.array(counts).sum(0) / (R.KM2 * len(files))).round(4).tolist(),
           "at": {k: R.fa_bootstrap(v) for k, v in at.items() if v}}
    R.save_json(res, OUT / f"{name}.json")
    print(name, {k: (pts[k], v["fa_per_km2"], v["ci95"]) for k, v in res["at"].items()}, flush=True)


if __name__ == "__main__":
    main()
