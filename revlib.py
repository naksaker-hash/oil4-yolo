"""Shared pieces of the revision analyses.

Every channel based detector (YOLO, z score, RX, random forest) sees the same
8 bit network input, decoded back to physical change values where needed, so
the comparison is about the detector and not about the input. IR-MAD works
on the ten raw bands of a pair, as it was designed to.

Hit rules
  standard  IoU >= 0.25, or a patch at most 5x the label covering >= 50 %
  strict    IoU >= 0.5
  centroid  the patch contains the centroid of the label
"""
import importlib.util
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import chi2

from config import CHIP, DATA, PIX_M, ROOT

_spec = importlib.util.spec_from_file_location("synth", ROOT / "04_synth.py")
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)
import realio  # noqa: E402

KM2 = (CHIP * PIX_M / 1000) ** 2
MIN_PX = 5
CH = ["dNBR", "dlogBAI", "dlogA"]


# ---------------------------------------------------------------- inputs
def decode(img):
    """8 bit network input -> physical change values, (3, H, W)."""
    out = []
    for i, c in enumerate(CH):
        lo, hi = synth.RANGE[c]
        out.append(lo + img[..., i].astype(np.float32) / 255 * (hi - lo))
    return np.stack(out)


def pair_input(pre, post):
    img, _ = synth.features(pre, post)
    return img


def load_pair(path, coreg=True):
    z = np.load(path)
    if "mask" in z.files:
        d = realio.load(path, coreg=coreg)
        return d["pre"], d["post"], d["mask"].astype(bool), d.get("scl_pre"), d.get("scl_post")
    pre, post = z["pre"], z["post"]
    if coreg:
        s = realio.shift_of(pre, post)
        pre, post = realio._move(pre, -np.asarray(s) / 2), realio._move(post, np.asarray(s) / 2)
    return pre, post, np.zeros((CHIP, CHIP), bool), None, None


# ---------------------------------------------------------------- patches
def patches(mask, score):
    lab, n = ndi.label(mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= MIN_PX:
            out.append((float(score[m].max()), m))
    return out


class ScoreMapDetector:
    """A detector defined by a per pixel score map; patches are the
    components of score >= t, scored by their maximum."""
    name = "scoremap"

    def score(self, img, pre=None, post=None):
        raise NotImplementedError

    def at(self, img, t, pre=None, post=None):
        s = self.score(img, pre, post)
        return patches(s >= t, s)


class ZScore(ScoreMapDetector):
    name = "zscore"

    def score(self, img, pre=None, post=None):
        d = decode(img)[0]
        return -(d - d.mean()) / max(d.std(), 1e-6)


class RX(ScoreMapDetector):
    """Reed Xiaoli anomaly detector on the three change channels, mean and
    covariance estimated robustly over the chip (central 98 % of pixels)."""
    name = "rx"

    def score(self, img, pre=None, post=None):
        x = decode(img).reshape(3, -1).T
        med = np.median(x, 0)
        d0 = np.sum((x - med) ** 2 / np.maximum(x.var(0), 1e-9), 1)
        keep = d0 <= np.quantile(d0, 0.98)
        mu = x[keep].mean(0)
        cov = np.cov(x[keep].T) + np.eye(3) * 1e-6
        diff = x - mu
        d = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)
        return d.reshape(CHIP, CHIP)


class IRMAD(ScoreMapDetector):
    """Iteratively reweighted MAD (Nielsen 2007) on the ten raw bands; the
    score is the chi square no change statistic (10 degrees of freedom)."""
    name = "irmad"

    def score(self, img, pre=None, post=None, iters=20):
        X = np.nan_to_num(pre.reshape(10, -1)).astype(np.float64)
        Y = np.nan_to_num(post.reshape(10, -1)).astype(np.float64)
        n = X.shape[1]
        w = np.ones(n)
        for _ in range(iters):
            ws = w / w.sum()
            mx, my = X @ ws, Y @ ws
            Xc, Yc = X - mx[:, None], Y - my[:, None]
            Sxx = (Xc * ws) @ Xc.T + np.eye(10) * 1e-9
            Syy = (Yc * ws) @ Yc.T + np.eye(10) * 1e-9
            Sxy = (Xc * ws) @ Yc.T
            # canonical correlation via generalised eigenproblems
            A = np.linalg.solve(Sxx, Sxy) @ np.linalg.solve(Syy, Sxy.T)
            ev, a = np.linalg.eig(A)
            order = np.argsort(-ev.real)
            ev, a = ev.real[order], a.real[:, order]
            b = np.linalg.solve(Syy, Sxy.T) @ a
            a /= np.sqrt(np.einsum("ij,jk,ki->i", a.T, Sxx, a))[None]
            b /= np.sqrt(np.einsum("ij,jk,ki->i", b.T, Syy, b))[None]
            rho = np.clip(np.sqrt(np.clip(ev, 0, 1)), 0, 0.999999)
            mad = a.T @ Xc - b.T @ Yc
            var = 2 * (1 - rho)
            z = np.sum(mad ** 2 / var[:, None], 0)
            w = 1 - chi2.cdf(z, 10)
        return z.reshape(CHIP, CHIP)          # chi square statistic, 10 dof


class RandomForestDet(ScoreMapDetector):
    """Pixel classifier trained on the same synthetic data as the network."""
    name = "rf"

    def __init__(self, model):
        self.model = model

    @staticmethod
    def pixel_features(img):
        d = decode(img)
        local = np.stack([ndi.uniform_filter(c, 5) for c in d])
        return np.concatenate([d, local]).reshape(6, -1).T

    def score(self, img, pre=None, post=None):
        p = self.model.predict_proba(self.pixel_features(img))[:, 1]
        return p.reshape(CHIP, CHIP)


class YoloDet:
    name = "yolo"

    def __init__(self, run, imgsz=384):
        from ultralytics import YOLO
        self.model = YOLO(str(ROOT / "runs" / run / "weights" / "best.pt"))
        self.imgsz = imgsz

    def instances(self, img):
        # the training images were read by OpenCV, i.e. in BGR order
        r = self.model.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=self.imgsz, conf=0.01,
                               verbose=False, retina_masks=True, device="cpu")[0]
        if r.masks is None:
            return []
        return list(zip(r.boxes.conf.cpu().numpy().tolist(), r.masks.data.cpu().numpy() > 0.5))

    def score(self, img, pre=None, post=None):
        smap = np.zeros((CHIP, CHIP), np.float32)
        for s, m in self.instances(img):
            smap = np.where(m, np.maximum(smap, s), smap)
        return smap

    def at(self, img, t, pre=None, post=None):
        s = self.score(img)
        return patches(s >= t, s)


# ---------------------------------------------------------------- scoring
def judge(pts, label, rule="standard"):
    best = None
    cy, cx = [int(round(v)) for v in ndi.center_of_mass(label)]
    for s, m in pts:
        inter = (m & label).sum()
        cover, iou = inter / label.sum(), inter / (m | label).sum()
        if rule == "standard":
            ok = (cover >= 0.5 and m.sum() <= 5 * label.sum()) or iou >= 0.25
        elif rule == "strict":
            ok = iou >= 0.5
        else:
            ok = bool(m[cy, cx])
        if ok and (best is None or s > best[0]):
            best = (s, float(iou), float(cover))
    if best is None:
        return {"hit": False, "iou": None, "rank": None, "score": None, "n_patches": len(pts)}
    return {"hit": True, "iou": round(best[1], 3), "rank": 1 + sum(s > best[0] for s, _ in pts),
            "score": round(best[0], 4), "n_patches": len(pts)}


def fa_bootstrap(counts, n_boot=2000, seed=0):
    """Chip level (cluster) bootstrap of the false alarm rate per km2."""
    counts = np.asarray(counts, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), (n_boot, len(counts)))
    rates = counts[idx].sum(1) / (KM2 * len(counts))
    return {"n_chips": len(counts), "fa": int(counts.sum()),
            "fa_per_km2": round(counts.sum() / (KM2 * len(counts)), 4),
            "ci95": [round(float(np.quantile(rates, 0.025)), 4), round(float(np.quantile(rates, 0.975)), 4)],
            "upper95_poisson": round(chi2.ppf(0.95, 2 * (counts.sum() + 1)) / 2 / (KM2 * len(counts)), 4)}


def val_f1_threshold(score_fn, root, grid, rule="standard", limit=None):
    """Threshold maximising patch F1 on the synthetic validation plains, with
    the same hit rule as the test (5x cap included)."""
    masks = sorted((root / "masks" / "val").glob("*.png"))[:limit]
    tp, fp, fn = np.zeros(len(grid)), np.zeros(len(grid)), np.zeros(len(grid))
    for mp in masks:
        lab, n = ndi.label(np.asarray(Image.open(mp)) > 0)
        truth = [lab == i for i in range(1, n + 1)]
        img = np.asarray(Image.open(root / "images" / "val" / mp.name))
        s = score_fn(img)
        for j, t in enumerate(grid):
            matched = set()
            for _, m in patches(s >= t, s):
                hit = []
                for i, g in enumerate(truth):
                    inter = (m & g).sum()
                    if rule == "standard":
                        ok = (inter / g.sum() >= 0.5 and m.sum() <= 5 * g.sum()) or inter / (m | g).sum() >= 0.25
                    else:
                        ok = inter / (m | g).sum() >= 0.5
                    if ok:
                        hit.append(i)
                if hit:
                    matched.update(hit)
                else:
                    fp[j] += 1
            tp[j] += len(matched)
            fn[j] += len(truth) - len(matched)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    j = int(f1.argmax())
    return float(grid[j]), {"grid": [float(g) for g in grid], "f1": f1.round(4).tolist()}


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=float)
