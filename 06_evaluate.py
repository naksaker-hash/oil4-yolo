"""Evaluate a trained model on the real chips only, against a bitemporal
outlier detector run on the same chips, with both scored the same way.

Both detectors produce
disjoint patches:
  z-score   z = (dNBR - mean) / sd over the chip, patches are
            the connected components of z <= -t with at least 5 px
            (the reference rule flags |z| >= 4 and dropped patches under 5 px; oil
            lowers NBR, so the oil tail is the negative one). Patch score
            is -min z.
  YOLO      the union of all instance masks scoring >= t, split into
            connected components of at least 5 px. Patch score is the
            highest instance score inside it.
At a threshold t each detector gives a set of patches per chip, and:
  hit      some patch no larger than 5x the label covers >= 50 % of it,
           OR any patch has IoU >= 0.25 (the
           coverage rule keeps a correct full-extent outline of Siverek
           from scoring as a miss against the 0.49 ha core label)
  IoU      of that same patch, never the best of all patches
  rank     of that patch by score among the patches of its own chip
  FA       patches per km2 on the pre-event null chips, with the exact
           Poisson upper limit, the upper end of a two sided 95 % interval
           (zero patches on 6 chips of 6.55 km2 each bounds the rate only
           below 0.094 per km2)

Operating points, fixed BEFORE looking at the real chips:
  z-score   t = 4, the reference rule
  YOLO      the threshold that maximises patch-level F1 on the synthetic
            validation plains (no real data involved)
The oracle point (highest threshold that still hits) is reported as well
and labelled as post hoc. Co-registered null pairs are the main result;
the raw, misregistered pairs are reported as a sensitivity check.

    .venv/bin/python 06_evaluate.py --run gen_v2_yolo26n-seg_s0 --regime gen_v2
"""
import argparse
import csv
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import chi2

import realio
from config import CHIP, DATA, EVENTS, PIX_M, ROOT

KM2 = (CHIP * PIX_M / 1000) ** 2
MIN_PX = 5
Z_FIXED = 4.0
MAX_COVER_RATIO = 5


def patches(mask, score_map):
    lab, n = ndi.label(mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= MIN_PX:
            out.append((float(score_map[m].max()), m))
    return out


class ZScore:
    name = "zscore"

    def __init__(self, coreg):
        self.coreg = coreg
        self.cache = {}

    def z(self, path):
        if path not in self.cache:
            d = realio.load(path, coreg=self.coreg)
            nbr = lambda x: (x[6] - x[9]) / np.maximum(x[6] + x[9], 1e-6)
            dn = nbr(d["post"]) - nbr(d["pre"])
            self.cache[path] = np.nan_to_num((dn - np.nanmean(dn)) / np.nanstd(dn))
        return self.cache[path]

    def at(self, path, t):
        z = self.z(path)
        return patches(z <= -t, -z)


class Yolo:
    name = "yolo"

    def __init__(self, model, img_dir, imgsz):
        self.model, self.img_dir, self.imgsz = model, img_dir, imgsz
        self.cache = {}

    def instances(self, png):
        if png not in self.cache:
            r = self.model.predict(str(png), imgsz=self.imgsz, conf=0.01, verbose=False,
                                   retina_masks=True, device="cpu")[0]
            self.cache[png] = [] if r.masks is None else \
                list(zip(r.boxes.conf.cpu().numpy().tolist(), r.masks.data.cpu().numpy() > 0.5))
        return self.cache[png]

    def at_png(self, png, t):
        inst = [(s, m) for s, m in self.instances(png) if s >= t]
        if not inst:
            return []
        smap = np.zeros((CHIP, CHIP), np.float32)
        for s, m in inst:
            smap = np.where(m, np.maximum(smap, s), smap)
        return patches(smap > 0, smap)

    def at(self, path, t):
        return self.at_png(self.img_dir / f"{path.stem}.png", t)


def judge(pts, label):
    best = None
    for s, m in pts:
        inter = (m & label).sum()
        cover, iou = inter / label.sum(), inter / (m | label).sum()
        # coverage counts only for a patch at most 5x the label, enough for
        # Siverek's 2.33 ha extent around its 0.49 ha core (4.75x), not for
        # a patch that floods the chip
        if (cover >= 0.5 and m.sum() <= MAX_COVER_RATIO * label.sum()) or iou >= 0.25:
            if best is None or s > best[0]:
                best = (s, float(iou), float(cover))
    if best is None:
        return {"hit": False, "iou": None, "cover": None, "rank": None, "n_patches": len(pts)}
    rank = 1 + sum(s > best[0] for s, _ in pts)
    return {"hit": True, "iou": round(best[1], 3), "cover": round(best[2], 3),
            "rank": rank, "n_patches": len(pts)}


def fa(det, nulls, t):
    n = sum(len(det.at(p, t)) for p in nulls)
    area = KM2 * len(nulls)
    upper = chi2.ppf(0.975, 2 * (n + 1)) / 2 / area
    return {"fa_n": n, "fa_per_km2": round(n / area, 4), "fa_upper95": round(upper, 4)}


def point(det, event_path, label, nulls, t):
    return {"thr": round(float(t), 4), **judge(det.at(event_path, t), label), **fa(det, nulls, t)}


def sweep(det, event_path, label, nulls, grid):
    rows = [point(det, event_path, label, nulls, t) for t in grid]
    hits = [r for r in rows if r["hit"]]
    return rows, (hits[-1] if hits else None)


def yolo_val_threshold(yolo_val, root):
    """Patch-level F1 on the synthetic validation plains."""
    masks = sorted((root / "masks" / "val").glob("*.png"))
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tp, fp, fn = np.zeros(len(grid)), np.zeros(len(grid)), np.zeros(len(grid))
    for mp in masks:
        truth = [m for m in (lambda l, n: [l == i for i in range(1, n + 1)])(
            *ndi.label(np.asarray(Image.open(mp)) > 0))]
        png = root / "images" / "val" / mp.name
        for j, t in enumerate(grid):
            pts = yolo_val.at_png(png, t)
            matched = set()
            for s, m in pts:
                hit = [i for i, g in enumerate(truth)
                       if (m & g).sum() / g.sum() >= 0.5 or (m & g).sum() / (m | g).sum() >= 0.25]
                if hit:
                    matched.update(hit)
                else:
                    fp[j] += 1
            tp[j] += len(matched)
            fn[j] += len(truth) - len(matched)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    j = int(f1.argmax())
    return float(grid[j]), {"grid": grid.tolist(), "f1": f1.round(4).tolist(),
                            "n_val_images": len(masks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--regime", required=True, help="dataset suffix, e.g. gen or gen_v2")
    ap.add_argument("--imgsz", type=int, default=384)
    args = ap.parse_args()
    out = ROOT / "runs" / args.run
    root = DATA / f"yolo_{args.regime}"
    from ultralytics import YOLO
    model = YOLO(str(out / "weights" / "best.pt"))

    t_yolo, val_info = yolo_val_threshold(Yolo(model, root / "images" / "val", args.imgsz), root)
    print(f"YOLO threshold from synthetic val F1: {t_yolo} (F1 {max(val_info['f1']):.3f})")

    summary = {"run": args.run, "regime": args.regime, "yolo_val_threshold": t_yolo,
               "yolo_val_f1": val_info, "z_fixed": Z_FIXED}
    for variant, coreg in [("coreg", True), ("raw", False)]:
        for event in EVENTS:
            real = DATA / "real"
            ev = real / f"{event}_event.npz"
            nulls = sorted(real.glob(f"{event}_null*.npz"))
            label = realio.load(ev, coreg=False)["mask"]
            dets = {"zscore": (ZScore(coreg), Z_FIXED, np.round(np.arange(2.0, 15.01, 0.25), 2)),
                    "yolo": (Yolo(model, root / "images" / f"test_{event}{'' if coreg else '_raw'}",
                                  args.imgsz), t_yolo, np.round(np.arange(0.01, 0.96, 0.01), 2))}
            for name, (det, t_fixed, grid) in dets.items():
                rows, oracle = sweep(det, ev, label, nulls, grid)
                with open(out / f"eval_{event}_{name}_{variant}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
                key = f"{event}_{name}_{variant}"
                summary[key] = {"fixed": point(det, ev, label, nulls, t_fixed),
                                "oracle_post_hoc": oracle, "n_null_chips": len(nulls),
                                "null_area_km2": round(KM2 * len(nulls), 2)}
                print(key, json.dumps(summary[key]))
    json.dump(summary, open(out / "eval_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
