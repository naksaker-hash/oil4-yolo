"""Random forest pixel classifier trained on the same synthetic data as the
network (reviewer request): six features per pixel, the three change
channels and their 5 x 5 means, from the 8 bit network input. Training
masks are rasterised from the YOLO polygons.

    .venv/bin/python 12_train_rf.py --regime gen_v2
"""
import argparse

import joblib
import numpy as np
from PIL import Image
from skimage.draw import polygon

from config import CHIP, DATA, ROOT
from revlib import RandomForestDet


def label_mask(txt):
    m = np.zeros((CHIP, CHIP), bool)
    for line in txt.read_text().splitlines():
        xy = np.array(line.split()[1:], float).reshape(-1, 2) * CHIP
        rr, cc = polygon(xy[:, 1], xy[:, 0], (CHIP, CHIP))
        m[rr, cc] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="gen_v2")
    ap.add_argument("--per_image", type=int, default=120)
    args = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier
    root = DATA / f"yolo_{args.regime}"
    rng = np.random.default_rng(0)
    X, y = [], []
    for p in sorted((root / "images" / "train").glob("*.png")):
        img = np.asarray(Image.open(p))
        m = label_mask(root / "labels" / "train" / f"{p.stem}.txt")
        f = RandomForestDet.pixel_features(img)
        pos, neg = np.flatnonzero(m.ravel()), np.flatnonzero(~m.ravel())
        take = [rng.choice(pos, min(len(pos), args.per_image), replace=False)] if len(pos) else []
        take.append(rng.choice(neg, args.per_image, replace=False))
        idx = np.concatenate(take)
        X.append(f[idx]); y.append(m.ravel()[idx])
    X, y = np.concatenate(X), np.concatenate(y)
    print("pixels", len(y), "positive", int(y.sum()))
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5,
                                n_jobs=2, random_state=0).fit(X, y)
    out = ROOT / "runs" / f"rf_{args.regime}"
    out.mkdir(exist_ok=True)
    joblib.dump(rf, out / "rf.joblib", compress=3)
    print("saved", out)


if __name__ == "__main__":
    main()
