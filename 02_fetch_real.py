"""Fetch the real test chips: the event pair and every pre-event null pair
for Narli 2018 and Siverek 2021, with the plume rasterised onto each grid.

These chips are never used in training. They also give the measured
attenuation k = post / pre inside each plume, written to data/k_<event>.json.

    python 02_fetch_real.py
"""
import json

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds as tf_from_bounds
from rasterio.warp import transform_geom

from config import BANDS, CHIP, DATA, EVENTS, K_SIVEREK_CORE, LABELS
from s2io import bad_fraction, read_pair


def plume_mask(label_file, epsg, bounds):
    gj = json.load(open(LABELS / label_file))
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geoms = [transform_geom("EPSG:4326", f"EPSG:{epsg}", f["geometry"]) for f in feats]
    t = tf_from_bounds(*bounds, CHIP, CHIP)
    return rasterize(geoms, out_shape=(CHIP, CHIP), transform=t,
                     fill=0, default_value=1, all_touched=False).astype(np.uint8)


def main():
    out = DATA / "real"
    out.mkdir(parents=True, exist_ok=True)
    for name, ev in EVENTS.items():
        pairs = [("event", ev["pre"], ev["post"])] + \
                [(f"null{i:02d}", a, b) for i, (a, b) in enumerate(ev["null_pairs"])]
        for tag, a, b in pairs:
            f = out / f"{name}_{tag}.npz"
            if f.exists():
                print("have", f.name)
                continue
            pre, post, sp, sq, meta = read_pair(ev["lon"], ev["lat"], a, b)
            mask = plume_mask(ev["label"], meta["epsg"], meta["bounds"]) \
                if tag == "event" else np.zeros((CHIP, CHIP), np.uint8)
            np.savez_compressed(f, pre=pre, post=post, scl_pre=sp, scl_post=sq,
                                mask=mask, meta=json.dumps({**meta, "pre": a, "post": b}))
            print(f"{f.name}  tile {meta['tile']}  bad {bad_fraction(sp, pre):.3f}/"
                  f"{bad_fraction(sq, post):.3f}  plume px {int(mask.sum())}")

        z = np.load(out / f"{name}_event.npz")
        m = z["mask"].astype(bool)
        k = (np.nanmean(z["post"][:, m], 1) / np.nanmean(z["pre"][:, m], 1)).tolist()
        json.dump({"event": name, "plume_px_10m": int(m.sum()),
                   "k": dict(zip(BANDS, k))}, open(DATA / f"k_{name}.json", "w"), indent=1)
        print(name, "k:", " ".join(f"{b} {v:.3f}" for b, v in zip(BANDS, k)))
        if name == "siverek":
            print("published core k:", " ".join(f"{v:.3f}" for v in K_SIVEREK_CORE))


if __name__ == "__main__":
    main()
