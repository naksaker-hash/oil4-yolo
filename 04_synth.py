"""Implant synthetic oil plumes into clean pairs and write a YOLO
segmentation dataset. Real event and null chips are written alongside as
the test sets, untouched.

The plume model follows the Siverek measurements: oil multiplies each band's
reflectance by a factor k (R' = k R), with k closest to 1 at the plume
margin and smallest over the thickest coating. Only the post image is
altered.

Attenuation regimes (--regime):
  gen      generic prior, not fitted to either event: VNIR k drawn from
           0.2 to 0.7, SWIR attenuated less (k pulled toward 1)
  siverek  Siverek core k measured from its event chip, jittered
  narli    Narli k measured from its event chip, jittered
Training on siverek and testing on Narli, and the reverse, is the cross
event check. gen is the main experiment.

    python 04_synth.py --regime gen --copies 3
    python 04_synth.py --regime gen --copies 3 --profile v2   # -> data/yolo_gen_v2

v2 was designed AFTER v1 missed the Siverek plume (the real core was darker
and sharper edged than almost any small v1 plume, and the v1 SWIR prior
could not reach either event's measured SWIR k). It adds the core and
margin structure measured at Siverek and widens the SWIR prior. Because it
was motivated by a test result, v1 and v2 are both reported in the paper.
--ablate returns one v2 property to its v1 setting (sharp, halo, green, swir).
--profile dr is a domain randomised generator set without reference to
either event (random darkening, independent random factor per band, random
edge, margin and placement). --native20 implants the 20 m bands at their
native resolution. A non zero --seed draws an independent dataset.

Details that matter for every profile:
  labels are pixel-corner exact polygons
  the stretch encodes zero change as 114 in every channel, the value
  ultralytics pads with, so mosaic and translate padding reads as no change
  validation is spatially held out (two whole plains)
  real chips are co-registered (realio.py); raw versions go to test_*_raw
"""
import argparse
import json
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

import realio
from config import BANDS, CHIP, DATA, PLAINS

# feature channels and fixed 8 bit stretch, shared by every split
CH = ["dNBR", "dlogBAI", "dlogA"]
# lower bounds as before; upper bounds set so that 0 maps to 114/255
PAD = 114 / 255
RANGE = {c: (lo, lo - lo / PAD) for c, lo in
         {"dNBR": -0.6, "dlogBAI": -1.0, "dlogA": -1.2}.items()}
VAL_PLAINS = ("ceylanpinar", "amik_hatay")
IB = {b: i for i, b in enumerate(BANDS)}
VNIR = [IB[b] for b in ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]]
SWIR = [IB["B11"], IB["B12"]]


def features(pre, post):
    def nbr(x):
        return (x[IB["B8"]] - x[IB["B12"]]) / np.maximum(x[IB["B8"]] + x[IB["B12"]], 1e-6)

    def logbai(x):
        return -np.log10(np.maximum((0.1 - x[IB["B4"]]) ** 2 + (0.06 - x[IB["B8"]]) ** 2, 1e-6))

    def loga(x):
        return np.log(np.maximum(x.mean(0), 1e-4))

    f = {"dNBR": nbr(post) - nbr(pre),
         "dlogBAI": logbai(post) - logbai(pre),
         "dlogA": loga(post) - loga(pre)}
    img = np.zeros((CHIP, CHIP, 3), np.uint8)
    for i, c in enumerate(CH):
        lo, hi = RANGE[c]
        v = np.nan_to_num(f[c], nan=0.0)
        img[..., i] = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return img, f


def plume_shape(rng):
    """A connected, irregular, elongated blob of 20 to 1200 pixels."""
    target = int(np.exp(rng.uniform(np.log(20), np.log(1200))))
    n = 96
    yy, xx = np.mgrid[:n, :n] - n / 2
    th = rng.uniform(0, np.pi)
    el = rng.uniform(1.0, 3.0)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    blob = np.exp(-(u ** 2 / el + v ** 2 * el) / (2 * (n / 6) ** 2))
    rough = ndi.gaussian_filter(rng.standard_normal((n, n)), rng.uniform(2, 6))
    field = blob + rng.uniform(0.15, 0.45) * rough / rough.std() * blob.max()
    thr = np.sort(field.ravel())[::-1][min(target, n * n - 1)]
    m = field > thr
    lab, _ = ndi.label(m)
    m = lab == lab[n // 2, n // 2] if lab[n // 2, n // 2] else lab == np.bincount(lab.ravel())[1:].argmax() + 1
    m = ndi.binary_fill_holes(m)
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def core_k(rng, regime, k_meas, profile="v1", off=()):
    if regime == "gen":
        kv = rng.uniform(0.2 if profile == "v1" else 0.15, 0.7)
        k = kv + rng.normal(0, 0.04, len(BANDS))
        # SWIR is pulled toward 1. v1: pull 0.2 to 0.6, which reaches neither
        # event (Siverek 0.59 to 0.64, Narli 0.80 to 1.09). v2: 0.4 to 1.1.
        pull = rng.uniform(0.2, 0.6) if profile == "v1" or "swir" in off else rng.uniform(0.4, 1.1)
        k[SWIR] = kv + pull * (1 - kv) + rng.normal(0, 0.03, 2)
    else:
        k = np.asarray(k_meas[regime]) * rng.uniform(0.85, 1.15) + rng.normal(0, 0.03, len(BANDS))
    if regime == "gen" and profile == "dr":
        # domain randomisation, set without reference to either event: a
        # random overall darkening and an independent random factor per band
        k = rng.uniform(0.1, 0.95) * np.exp(rng.normal(0, 0.25, len(BANDS)))
    return np.clip(k, 0.1, 1.1)


BANDS20 = [IB[b] for b in ("B5", "B6", "B7", "B8A", "B11", "B12")]


def implant(rng, pre, post, regime, k_meas, profile="v1", off=(), native20=False):
    post = post.copy()
    mask = np.zeros((CHIP, CHIP), bool)
    n = rng.choice([0, 1, 1, 1, 2, 3], p=None)
    polys = []
    nd = (pre[IB["B8"]] - pre[IB["B4"]]) / np.maximum(pre[IB["B8"]] + pre[IB["B4"]], 1e-6)
    green = np.nan_to_num(nd) > 0.45
    for _ in range(n):
        s = plume_shape(rng)
        h, w = s.shape
        if h >= CHIP - 4 or w >= CHIP - 4:
            continue
        y0, x0 = rng.integers(2, CHIP - h - 2), rng.integers(2, CHIP - w - 2)
        if profile == "v2" and "green" not in off and rng.random() < 0.5 and green.any():
            # half the v2 plumes are meant for green crop (pre NDVI > 0.45),
            # the substrate of Siverek; Narli was bare harvested soil. A
            # placement counts only if at least half the plume is on green.
            gy, gx = np.nonzero(green)
            for _try in range(30):
                j = rng.integers(len(gy))
                ty = int(np.clip(gy[j] - h // 2, 2, CHIP - h - 2))
                tx = int(np.clip(gx[j] - w // 2, 2, CHIP - w - 2))
                if green[ty:ty + h, tx:tx + w][s].mean() >= 0.5:
                    y0, x0 = ty, tx
                    break
        m = np.zeros((CHIP, CHIP), bool)
        m[y0:y0 + h, x0:x0 + w] = s
        if (m & ndi.binary_dilation(mask, iterations=3)).any():
            continue
        # coating thickness: lowest at the edge, 1 in the interior
        d = ndi.distance_transform_edt(m)
        wgt = np.clip(d / max(d.max() * rng.uniform(0.3, 0.8), 1), 0, 1)
        # v1 always fades to 0.3 at the edge. v2 also allows a sharp edged
        # core, the Siverek core being strong up to its boundary.
        if profile == "dr":
            floor = rng.uniform(0.2, 1.0)
        else:
            floor = 0.3 if profile == "v1" or "sharp" in off else rng.uniform(0.3, 1.0)
        wgt = np.where(m, floor + (1 - floor) * wgt, 0)
        if profile == "dr" and rng.random() < 0.5:
            # random unlabelled margin, width and weight drawn wide
            halo = ndi.binary_dilation(m, iterations=int(rng.integers(1, 7)))
            halo &= ndi.gaussian_filter(rng.standard_normal(m.shape), 2) > rng.uniform(-0.6, 0.3)
            halo &= ~m
            wgt = np.where(halo, rng.uniform(0.05, 0.7), wgt)
        if profile == "v2" and "halo" not in off and rng.random() < 0.5:
            # weaker oiled margin around the core, not labelled, as measured
            # at Siverek (core 0.49 ha inside a 2.33 ha extent)
            halo = ndi.binary_dilation(m, iterations=int(rng.integers(1, 6)))
            halo &= ndi.gaussian_filter(rng.standard_normal(m.shape), 2) > -0.3
            halo &= ~m
            wgt = np.where(halo, rng.uniform(0.15, 0.5), wgt)
        wgt = ndi.gaussian_filter(wgt, 0.7)          # mixed edge pixels
        k = core_k(rng, regime, k_meas, profile, off)
        att = np.repeat(wgt[None], len(BANDS), 0)
        if native20:
            # the 20 m bands see the coating averaged over their native
            # 2 x 2 footprint, then resampled to 10 m like the data
            w20 = wgt.reshape(CHIP // 2, 2, CHIP // 2, 2).mean((1, 3))
            att[BANDS20] = ndi.zoom(w20, 2, order=1)[None]
        post *= 1 - att * (1 - k[:, None, None])
        mask |= m
        polys.append(m)
    post += rng.normal(0, 0.002, post.shape).astype(np.float32)
    return post, mask, polys


def yolo_lines(masks):
    lines = []
    for m in masks:
        # find_contours works in pixel-centre index space; YOLO polygons are
        # in pixel-corner space, so undo the pad and add half a pixel
        c = max(measure.find_contours(np.pad(m, 1).astype(float), 0.5), key=len) - 0.5
        c = measure.approximate_polygon(c, 0.7)
        if len(c) < 3:
            continue
        xy = np.clip(c[:, ::-1] / CHIP, 0, 1)
        lines.append("0 " + " ".join(f"{v:.5f}" for v in xy.ravel()))
    return lines


def components(mask):
    lab, n = ndi.label(mask)
    return [lab == i for i in range(1, n + 1)]


def write(root, split, name, img, lines, mask=None):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(root / "images" / split / f"{name}.png")
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    if mask is not None:                     # pixel truth for 06_evaluate
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(root / "masks" / split / f"{name}.png")


def plain_of(meta):
    for name, (a, b, c, d) in PLAINS.items():
        if a <= meta["lon"] <= c and b <= meta["lat"] <= d:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="gen", choices=["gen", "siverek", "narli"])
    ap.add_argument("--copies", type=int, default=3, help="synthetic versions per clean pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native20", action="store_true",
                    help="implant the 20 m bands at their native resolution")
    ap.add_argument("--profile", default="v1", choices=["v1", "v2", "dr"],
                    help="v2 adds sharp edged cores and an unlabelled oiled margin")
    ap.add_argument("--ablate", default="", choices=["", "sharp", "halo", "green", "swir"],
                    help="v2 with one ingredient switched back to v1 behaviour")
    args = ap.parse_args()
    off = (args.ablate,) if args.ablate else ()
    rng = np.random.default_rng(args.seed)
    k_meas = {e: list(json.load(open(DATA / f"k_{e}.json"))["k"].values())
              for e in ["narli", "siverek"]}

    root = DATA / (f"yolo_{args.regime}" + ("" if args.profile == "v1" else f"_{args.profile}")
                  + (f"_no{args.ablate}" if args.ablate else "")
                  + ("_n20" if args.native20 else "") + (f"_d{args.seed}" if args.seed else ""))
    shutil.rmtree(root, ignore_errors=True)
    clean = sorted((DATA / "clean").glob("*.npz"))
    for f in clean:
        z = np.load(f)
        # whole plains are held out for validation, so val never shares a
        # footprint or a scene with train
        split = "val" if plain_of(json.loads(str(z["meta"]))) in VAL_PLAINS else "train"
        for c in range(args.copies):
            post, mask, polys = implant(rng, z["pre"], z["post"], args.regime, k_meas,
                                        args.profile, off, args.native20)
            img, _ = features(z["pre"], post)
            write(root, split, f"{f.stem}_{c}", img, yolo_lines(polys),
                  mask if split == "val" else None)

    # real chips, never altered apart from co-registration: event pairs
    # carry the plume, null pairs none
    for f in sorted((DATA / "real").glob("*.npz")):
        event = f.stem.split("_")[0]
        for coreg, suffix in [(True, ""), (False, "_raw")]:
            d = realio.load(f, coreg=coreg)
            img, _ = features(d["pre"], d["post"])
            write(root, f"test_{event}{suffix}", f.stem, img,
                  yolo_lines(components(d["mask"])), d["mask"])

    names = {0: "oil"}
    for split in ["narli", "siverek"]:
        y = {"path": str(root), "train": "images/train", "val": "images/val",
             "test": f"images/test_{split}", "names": names}
        (root / f"data_test_{split}.yaml").write_text(json.dumps(y, indent=1))
    print(root, "train", len(list((root / "images/train").glob("*"))),
          "val", len(list((root / "images/val").glob("*"))))


if __name__ == "__main__":
    main()
