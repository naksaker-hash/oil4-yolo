"""Semi-real detection test: the measured attenuation of each spill,
implanted into the 316 held out clean pairs, scored per plume size.

Every pair receives one plume whose shape is drawn as in training, with
the raw measured factors of one spill (no jitter, no scaling) and a full
coating inside the outline smoothed by 0.7 pixel. Three variants:
  core    no unlabelled margin, random placement
  margin  an unlabelled oiled margin 3 to 6 pixels wide at weight 0.3,
          as around the real Siverek core (0.49 ha core, 2.33 ha extent)
  green   core and margin, placed so that half the plume lies on green
          crop (NDVI before > 0.45), the substrate of Siverek Each network is read at its fixed threshold with the
standard hit rule. Two real spills cannot give a probability of detection;
this test gives one for spills that attenuate like them.

    .venv/bin/python 22_implant_test.py
"""
import json

import numpy as np
from scipy import ndimage as ndi

import revlib as R
from config import CHIP, DATA, ROOT

S = R.synth
OUT = ROOT / "runs" / "revision"
NETS = [f"gen_yolo26n-seg_s{s}" for s in range(3)] + \
       [f"gen_v2_yolo26n-seg_s{s}" for s in range(3)] + \
       [f"gen_dr_yolo26n-seg_s{s}" for s in range(3)]
BINS = [(0, 50), (50, 150), (150, 10 ** 6)]


def plumes(files, k, seed, variant):
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        z = np.load(f)
        pre, post = z["pre"], z["post"].copy()
        s = S.plume_shape(rng)
        h, w = s.shape
        y0, x0 = rng.integers(2, CHIP - h - 2), rng.integers(2, CHIP - w - 2)
        if variant == "green":
            nd = (pre[S.IB["B8"]] - pre[S.IB["B4"]]) / np.maximum(pre[S.IB["B8"]] + pre[S.IB["B4"]], 1e-6)
            green = np.nan_to_num(nd) > 0.45
            gy, gx = np.nonzero(green)
            for _ in range(60 if len(gy) else 0):
                j = rng.integers(len(gy))
                ty = int(np.clip(gy[j] - h // 2, 2, CHIP - h - 2))
                tx = int(np.clip(gx[j] - w // 2, 2, CHIP - w - 2))
                if green[ty:ty + h, tx:tx + w][s].mean() >= 0.5:
                    y0, x0 = ty, tx
                    break
            else:
                continue                      # no green field large enough
        m = np.zeros((CHIP, CHIP), bool)
        m[y0:y0 + h, x0:x0 + w] = s
        wgt = m.astype(float)
        if variant in ("margin", "green"):
            halo = ndi.binary_dilation(m, iterations=int(rng.integers(3, 7))) & ~m
            wgt = np.where(halo, 0.3, wgt)
        wgt = ndi.gaussian_filter(wgt, 0.7)
        post = post * (1 - wgt[None] * (1 - np.asarray(k)[:, None, None]))
        post += rng.normal(0, 0.002, post.shape).astype(np.float32)
        img, _ = S.features(pre, post.astype(np.float32))
        out.append((img, m))
    return out


def main():
    files = [f for f in sorted((DATA / "clean").glob("*.npz"))
             if S.plain_of(json.loads(str(np.load(f)["meta"]))) in S.VAL_PLAINS]
    k_meas = {e: list(json.load(open(DATA / f"k_{e}.json"))["k"].values()) for e in ("narli", "siverek")}
    sets = {f"{e}_{v}": plumes(files, k_meas[e], 10 * i + j, v)
            for i, e in enumerate(("narli", "siverek")) for j, v in enumerate(("core", "margin", "green"))}
    res = {}
    for run in NETS:
        th = json.load(open(OUT / f"yolo_{run}.json"))["thresholds"]["synthetic_f1_same_rule"]
        det = R.YoloDet(run)
        r = {"thr": th}
        for e, items in sets.items():
            hit = np.zeros(len(BINS))
            tot = np.zeros(len(BINS))
            for img, m in items:
                s = det.score(img)
                ok = R.judge(R.patches(s >= th, s), m, "standard")["hit"]
                b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= m.sum() < hi)
                hit[b] += ok
                tot[b] += 1
            r[e] = {"rate_by_size": (hit / np.maximum(tot, 1)).round(3).tolist(), "n_by_size": tot.astype(int).tolist(),
                    "rate": round(float(hit.sum() / tot.sum()), 3)}
        res[run] = r
        print(run, {e: r[e]["rate"] for e in sets}, flush=True)
    R.save_json(res, OUT / "implant_test.json")


if __name__ == "__main__":
    main()
