"""Physical checks of the attenuation model requested in review.

k_raw    median post/pre ratio of the pixels inside each reference outline
k_corr   the same divided by the median ratio of control pixels, i.e. unoiled
         pixels of the same chip (outside a 200 m buffer) whose pre-event
         reflectance in the band lies within the interquartile range of the
         oiled pixels, which removes the ordinary change between the dates
k_20m    k_corr after averaging both images and the outline to 20 m
model    per band, the control corrected post reflectance of oiled pixels is
         regressed on the pre reflectance. A multiplicative coating predicts
         a zero intercept (post = k pre); linear mixing with an oil endmember
         predicts a positive intercept (post = (1 - f) pre + f rho_oil).
         The two fits are compared by AIC.

    .venv/bin/python 14_physics.py
"""
import json

import numpy as np
from scipy import ndimage as ndi

import revlib as R
from config import BANDS, DATA, ROOT


def ratios(pre, post, mask, ctrl_mask):
    out = {}
    for b, name in enumerate(BANDS):
        p, q = pre[b][mask], post[b][mask]
        r = q / np.maximum(p, 1e-4)
        lo, hi = np.quantile(p, [0.25, 0.75])
        c = ctrl_mask & (pre[b] >= lo) & (pre[b] <= hi)
        rc = np.median(post[b][c] / np.maximum(pre[b][c], 1e-4)) if c.sum() > 50 else np.nan
        out[name] = {"k_raw": round(float(np.median(r)), 3),
                     "iqr": [round(float(np.quantile(r, 0.25)), 3), round(float(np.quantile(r, 0.75)), 3)],
                     "control_ratio": round(float(rc), 3), "n_control": int(c.sum()),
                     "k_corr": round(float(np.median(r) / rc), 3)}
    return out


def fit_models(pre, post, mask, ctrl_ratio):
    out = {}
    for b, name in enumerate(BANDS):
        x = pre[b][mask].astype(float)
        y = post[b][mask].astype(float) / ctrl_ratio[name]
        n = len(x)
        k0 = float((x @ y) / (x @ x))                      # through the origin
        rss0 = float(((y - k0 * x) ** 2).sum())
        A = np.vstack([x, np.ones(n)]).T
        (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
        rss1 = float(((y - a * x - c) ** 2).sum())
        aic0 = n * np.log(rss0 / n) + 2
        aic1 = n * np.log(rss1 / n) + 4
        f = 1 - a
        out[name] = {"multiplicative_k": round(k0, 3), "mixing_slope": round(float(a), 3),
                     "mixing_intercept": round(float(c), 4),
                     "implied_rho_oil": round(float(c / f), 4) if 0.05 < f < 1 else None,
                     "delta_aic_mixing_minus_mult": round(float(aic1 - aic0), 1)}
    return out


def to20(x):
    h, w = x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2
    x = x[..., :h, :w]
    return x.reshape(*x.shape[:-2], h // 2, 2, w // 2, 2).mean((-3, -1))


def main():
    res = {}
    for ev in ["narli", "siverek"]:
        pre, post, mask, _, _ = R.load_pair(DATA / "real" / f"{ev}_event.npz", coreg=True)
        ctrl = ~ndi.binary_dilation(mask, iterations=20)
        r10 = ratios(pre, post, mask, ctrl)
        m20 = to20(mask.astype(float)) >= 0.5
        c20 = to20(ctrl.astype(float)) >= 1.0
        r20 = ratios(to20(pre), to20(post), m20, c20)
        cr = {b: r10[b]["control_ratio"] for b in BANDS}
        res[ev] = {"n_px": int(mask.sum()), "k10": r10, "k20": r20,
                   "models": fit_models(pre, post, mask, cr)}
        print(ev)
        for b in BANDS:
            m = res[ev]["models"][b]
            print(f"  {b:4s} k_raw {r10[b]['k_raw']:.2f} k_corr {r10[b]['k_corr']:.2f} "
                  f"k20 {r20[b]['k_corr']:.2f} | mix slope {m['mixing_slope']:.2f} "
                  f"int {m['mixing_intercept']:+.3f} dAIC {m['delta_aic_mixing_minus_mult']:+.0f}")
    R.save_json(res, ROOT / "runs" / "revision" / "physics.json")


if __name__ == "__main__":
    main()
