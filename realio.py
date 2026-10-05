"""Load a real event or null chip, optionally co-registered.

Five of the six Narli null pairs include scenes from the original
sentinel-2-l2a archive (processing baseline 02.07/02.08), because
Collection 1 has no item on those dates. Those scenes lack the reprocessed
geolocation and their pairs are misregistered by 0.5 to 0.75 px, against
0.25 px or less for every Collection 1 pair. The shift
is estimated on high-passed B8 and split evenly between pre and post, so
both images receive the same interpolation smoothing.
"""
import json

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from config import BANDS

B8 = BANDS.index("B8")


def _hp(x):
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    return x - ndi.gaussian_filter(x, 3)


def shift_of(pre, post):
    s, _, _ = phase_cross_correlation(_hp(pre[B8]), _hp(post[B8]), upsample_factor=8)
    return s                                  # (dy, dx) that moves post onto pre


def _move(img, s):
    return np.stack([ndi.shift(b, s, order=1, mode="nearest") for b in img])


def load(path, coreg=True):
    z = np.load(path)
    pre, post = z["pre"], z["post"]
    meta = json.loads(str(z["meta"]))
    s = shift_of(pre, post)
    meta["shift_px"] = [float(s[0]), float(s[1])]
    meta["collections"] = ["c1" if "_T" in meta[k] else "l2a_old" for k in ("pre_id", "post_id")]
    if coreg:
        pre, post = _move(pre, -s / 2), _move(post, s / 2)
    return {"pre": pre, "post": post, "mask": z["mask"].astype(bool),
            "scl_pre": z["scl_pre"], "scl_post": z["scl_post"], "meta": meta}
