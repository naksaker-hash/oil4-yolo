"""Why do the v1 and dr networks miss the real spills although they detect
implanted plumes with the measured attenuation (22_implant_test.py)?

Three diagnostics on the event footprints, for v1, v2 and dr networks:
  lowscore    every instance down to a score of 0.001 on the event chips:
              the highest score of any instance overlapping the outline,
              and the highest score elsewhere
  inplace     the raw measured factors implanted with the real reference
              outline (and, at Siverek, a 3 to 6 pixel margin at weight 0.3)
              into the null pairs of the same chip, i.e. the same field on
              earlier dates
  transplant  the observed per pixel ratio after to before the event inside
              the outline dilated by 5 pixels, applied to the image after
              each null pair, which carries the real spatial and spectral
              structure of the spill but not the rest of the event pair
Detection is read at each network's fixed threshold, standard rule.

    .venv/bin/python 25_diagnose.py
"""
import json

import numpy as np
from scipy import ndimage as ndi

import realio
import revlib as R
from config import DATA, ROOT

S = R.synth
OUT = ROOT / "runs" / "revision"
NETS = [f"gen_yolo26n-seg_s{s}" for s in range(3)] + [f"gen_v2_yolo26n-seg_s{s}" for s in range(3)] + \
       [f"gen_dr_yolo26n-seg_s{s}" for s in range(3)]


def chips(event, k):
    ev = realio.load(DATA / "real" / f"{event}_event.npz", coreg=True)
    lab = ev["mask"].astype(bool)
    ring = ndi.binary_dilation(lab, iterations=5)
    ratio = np.where(ring[None], ev["post"] / np.maximum(ev["pre"], 1e-4), 1.0)
    ratio = np.clip(ratio, 0.05, 3.0)
    rng = np.random.default_rng(0)
    inplace, transplant = [], []
    for f in sorted((DATA / "real").glob(f"{event}_null*.npz")):
        pre, post, _, _, _ = R.load_pair(f, coreg=True)
        w = lab.astype(float)
        if event == "siverek":
            halo = ndi.binary_dilation(lab, iterations=int(rng.integers(3, 7))) & ~lab
            w = np.where(halo, 0.3, w)
        w = ndi.gaussian_filter(w, 0.7)
        p1 = post * (1 - w[None] * (1 - np.asarray(k)[:, None, None]))
        inplace.append(R.pair_input(pre, p1.astype(np.float32)))
        transplant.append(R.pair_input(pre, (post * ratio).astype(np.float32)))
    return R.pair_input(ev["pre"], ev["post"]), lab, inplace, transplant


def main():
    k_meas = {e: list(json.load(open(DATA / f"k_{e}.json"))["k"].values()) for e in ("narli", "siverek")}
    data = {e: chips(e, k_meas[e]) for e in ("narli", "siverek")}
    res = {}
    for run in NETS:
        th = json.load(open(OUT / f"yolo_{run}.json"))["thresholds"]["synthetic_f1_same_rule"]
        det = R.YoloDet(run)
        r = {"thr": th}
        for e, (img, lab, inpl, trans) in data.items():
            ins = det.model.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=384, conf=0.001, verbose=False,
                                    retina_masks=True, device="cpu")[0]
            on, off = 0.0, 0.0
            if ins.masks is not None:
                for s, m in zip(ins.boxes.conf.numpy(), ins.masks.data.numpy() > 0.5):
                    if (m & lab).sum() >= 0.1 * min(m.sum(), lab.sum()):
                        on = max(on, float(s))
                    elif not (m & ndi.binary_dilation(lab, iterations=10)).any():
                        off = max(off, float(s))
            hits = lambda imgs: int(sum(R.judge(R.patches(det.score(x) >= th, det.score(x)), lab, "standard")["hit"]
                                        for x in imgs))
            r[e] = {"max_on_spill": round(on, 4), "max_elsewhere": round(off, 4),
                    "inplace": f"{hits(inpl)}/{len(inpl)}", "transplant": f"{hits(trans)}/{len(trans)}"}
        res[run] = r
        print(run, r, flush=True)
    # channel signature: real spill against the in place implant, medians inside the outline
    sig = {}
    for e, (img, lab, inpl, trans) in data.items():
        dec = lambda x: [round(float(v), 3) for v in np.median(R.decode(x)[:, lab], 1)]
        sig[e] = {"real": dec(img), "inplace_median": [round(float(v), 3) for v in
                                                       np.median([dec(x) for x in inpl], 0)],
                  "transplant_median": [round(float(v), 3) for v in np.median([dec(x) for x in trans], 0)]}
    res["_channels_dNBR_dlogBAI_dlnA"] = sig
    print(sig)
    R.save_json(res, OUT / "diagnose.json")


if __name__ == "__main__":
    main()
