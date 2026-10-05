"""Revision evaluation: every detector on every real test set.

Detectors
  yolo runs   threshold from synthetic validation F1, now with the same hit
              rule (5x cap) as the test; calibration bins on synthetic val
  zscore      t = 4 (earlier work) and t tuned on synthetic v2 validation
  rx          RX on the three channels, chi square p = 0.001 and tuned
  irmad       IR-MAD on the ten raw bands, chi square p = 0.001
  rf_gen, rf_gen_v2   random forests trained on the v1 and v2 synthetic data
Real sets
  event       the two event chips, coregistered and raw, three hit rules
  null22      the 22 pre-event null pairs of the event footprints
  nullval     316 held out clean pairs of the two validation plains
  scene       7 x 7 chips around each spill on the event dates
  later       later post-event images of both spills
  burn, shadow  confounder pairs
For every set the patch counts are stored on a threshold grid, so matched
operating points and FROC curves can be drawn later.

    .venv/bin/python 13_revision_eval.py --det yolo:gen_v2_yolo26n-seg_s0
    .venv/bin/python 13_revision_eval.py --det zscore
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import chi2

import revlib as R
from config import DATA, EVENTS, ROOT

OUT = ROOT / "runs" / "revision"
VAL_ROOT = DATA / "yolo_gen_v2"
BAD_SCL = [3, 8, 9, 10]


def grid_for(name):
    if name.startswith("yolo") or name.startswith("rf"):
        return np.round(np.arange(0.02, 0.96, 0.02), 2)
    if name.startswith("zscore"):
        return np.round(np.arange(2.0, 15.01, 0.25), 2)
    if name.startswith("rx"):
        return np.round(np.geomspace(5, 5000, 50), 2)
    return np.round(np.geomspace(10, 20000, 50), 1)


def real_sets():
    s = {"event": sorted((DATA / "real").glob("*_event.npz")),
         "null22": sorted((DATA / "real").glob("*_null*.npz")),
         "nullval": [], "scene": sorted((DATA / "scene").glob("*.npz")),
         "later": sorted((DATA / "later").glob("*.npz")),
         "burn": sorted((DATA / "confound").glob("burn_*.npz")),
         "shadow": sorted((DATA / "confound").glob("shadow_*.npz"))}
    for f in sorted((DATA / "clean").glob("*.npz")):
        meta = json.loads(str(np.load(f)["meta"]))
        if R.synth.plain_of(meta) in R.synth.VAL_PLAINS:
            s["nullval"].append(f)
    return s


def make_detector(spec):
    kind, _, arg = spec.partition(":")
    if kind == "yolo":
        return R.YoloDet(arg), f"yolo_{arg}"
    if kind == "zscore":
        return R.ZScore(), "zscore"
    if kind == "rx":
        return R.RX(), "rx"
    if kind == "irmad":
        return R.IRMAD(), "irmad"
    if kind == "rf":
        return R.RandomForestDet(joblib.load(ROOT / "runs" / f"rf_{arg}" / "rf.joblib")), f"rf_{arg}"
    raise ValueError(spec)


def thresholds(det, name, spec, limit):
    """Operating points fixed on synthetic data or by convention."""
    th = {}
    if name.startswith("yolo"):
        regime = spec.split(":")[1].split("_yolo")[0]
        root = DATA / f"yolo_{regime}"
        g = np.round(np.arange(0.05, 0.96, 0.05), 2)
        old = json.load(open(ROOT / "runs" / spec.split(":")[1] / "eval_summary.json"))["yolo_val_threshold"]
        th["synthetic_f1"] = old
        t, info = R.val_f1_threshold(det.score, root, g, "standard", limit)
        th["synthetic_f1_same_rule"] = t
        th["_val_f1"] = info
        th["_calibration"] = calibration(det, root, limit)
    elif name == "zscore":
        th["conventional"] = 4.0
        th["synthetic_f1"], th["_val_f1"] = R.val_f1_threshold(det.score, VAL_ROOT, grid_for(name), "standard", limit)
    elif name == "rx":
        th["conventional"] = float(chi2.ppf(0.999, 3))
        th["synthetic_f1"], th["_val_f1"] = R.val_f1_threshold(det.score, VAL_ROOT, grid_for(name), "standard", limit)
    elif name == "irmad":
        th["conventional"] = float(chi2.ppf(0.999, 10))
    elif name.startswith("rf"):
        root = DATA / f"yolo_{name[3:]}"
        th["synthetic_f1"], th["_val_f1"] = R.val_f1_threshold(det.score, root, grid_for(name), "standard", limit)
    return th


def calibration(det, root, limit):
    """Fraction of predicted instances that hit a synthetic plume, by score."""
    bins = np.linspace(0, 1, 11)
    hit, tot = np.zeros(10), np.zeros(10)
    for mp in sorted((root / "masks" / "val").glob("*.png"))[:limit]:
        lab, n = ndi.label(np.asarray(Image.open(mp)) > 0)
        truth = [lab == i for i in range(1, n + 1)]
        img = np.asarray(Image.open(root / "images" / "val" / mp.name))
        for s, m in det.instances(img):
            if s < 0.05:
                continue
            ok = any(((m & g).sum() / g.sum() >= 0.5 and m.sum() <= 5 * g.sum())
                     or (m & g).sum() / (m | g).sum() >= 0.25 for g in truth)
            b = min(int(s * 10), 9)
            tot[b] += 1
            hit[b] += ok
    frac = np.where(tot > 0, hit / np.maximum(tot, 1), np.nan)
    centres = (bins[:-1] + bins[1:]) / 2
    ece = float(np.nansum(tot * np.abs(frac - centres)) / max(tot.sum(), 1))
    return {"bin_centre": centres.round(2).tolist(), "n": tot.astype(int).tolist(),
            "frac_hit": [None if np.isnan(v) else round(float(v), 4) for v in frac], "ece": round(ece, 4)}


def bad_mask(scl_pre, scl_post):
    if scl_pre is None:
        return None
    return np.isin(scl_pre, BAD_SCL) | np.isin(scl_post, BAD_SCL)


def count(pts, bad=None):
    if bad is None:
        return len(pts)
    return sum(1 for _, m in pts if (m & bad).sum() < 0.5 * m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--val-limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    det, name = make_detector(args.det)
    th = thresholds(det, name, args.det, args.val_limit)
    fixed = {k: v for k, v in th.items() if not k.startswith("_")}
    grid = grid_for(name)
    res = {"detector": name, "thresholds": th, "grid": grid.tolist(), "sets": {}}
    sets = real_sets()
    for sname, files in sets.items():
        rows = []
        variants = [("coreg", True), ("raw", False)] if sname == "event" else [("coreg", True)]
        for f in files:
            for vname, coreg in variants:
                pre, post, mask, sp, sq = R.load_pair(f, coreg=coreg)
                img = R.pair_input(pre, post)
                s = det.score(img, pre, post)
                bad = bad_mask(sp, sq)
                row = {"file": f.name, "variant": vname, "meta": json.loads(str(np.load(f)["meta"]))}
                row["counts"] = [len(R.patches(s >= t, s)) for t in grid]
                row["counts_cloudmasked"] = [count(R.patches(s >= t, s), bad) for t in grid] if bad is not None else None
                row["max_score"] = float(s.max())
                if mask.any():
                    row["spill_max_score"] = float(s[mask].max())
                    row["outside_max_score"] = float(s[~ndi.binary_dilation(mask, iterations=10)].max())
                row["fixed"] = {}
                for k, t in fixed.items():
                    pts = R.patches(s >= t, s)
                    r = {"n_patches": len(pts), "n_cloudmasked": count(pts, bad) if bad is not None else None}
                    if mask.any():
                        for rule in ("standard", "strict", "centroid"):
                            r[rule] = R.judge(pts, mask, rule)
                    if sname == "burn":
                        meta = row["meta"]
                        from rasterio.warp import transform as wtf
                        xs, ys = wtf("EPSG:4326", f"EPSG:{meta['epsg']}", [meta["fire_lon"]], [meta["fire_lat"]])
                        x, y = xs[0], ys[0]
                        b = meta["bounds"]
                        cx, cy = (x - b[0]) / 10, (b[3] - y) / 10
                        yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
                        near = np.hypot(xx - cx, yy - cy) <= 40   # 400 m
                        r["at_fire"] = any((m & near).any() for _, m in pts)
                    if sname == "shadow":
                        sh = np.isin(sq, [3])
                        r["on_shadow"] = sum(1 for _, m in pts if (m & sh).sum() >= 0.5 * m.sum())
                    row["fixed"][k] = r
                rows.append(row)
        res["sets"][sname] = rows
        print(name, sname, len(rows), flush=True)
    R.save_json(res, OUT / f"{name}.json")


if __name__ == "__main__":
    main()
