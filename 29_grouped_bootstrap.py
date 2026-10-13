"""Checks requested by external review 3 (13 Oct 2026).

grouped    false alarms per km2 at the fixed threshold on the held out,
           fresh and calibration pairs, with 95% intervals from a bootstrap
           over chips, over plains and over acquisitions (the Sentinel-2
           product after the event, which groups chips of one scene and
           date), because chips of one plain or scene can share errors
clipping   fraction of values at the eight bit limits 0 and 255 in each
           input channel, on real pairs and on the v2 synthetic training
           images
overlap    pairs of chips within one set whose footprints overlap

    .venv/bin/python 29_grouped_bootstrap.py
"""
import json
from collections import defaultdict

import numpy as np
from PIL import Image

import revlib as R
from config import DATA, PLAINS, ROOT

REV = ROOT / "runs" / "revision"
NETS = [f"yolo:gen_v2_yolo26n-seg_s{s}" for s in range(3)] + [
    "yolo:gen_v2_yolov8n-seg_s0", "yolo:gen_v2_yolo11n-seg_s0", "yolo:gen_v2_d1_yolo26n-seg_s0",
    "yolo:gen_v2_yolo26n-seg_scratch_s0", "yolo:narli_yolo26n-seg_s0", "yolo:narli_v2_yolo26n-seg_s0",
    "yolo:siverek_yolo26n-seg_s0", "yolo:siverek_v2_yolo26n-seg_s0", "unet:gen_v2_unet384_s0", "zscore"]
FRESH = {"bismil": (40.62, 37.80, 41.10, 38.00), "cizre_silopi": (42.00, 37.15, 42.45, 37.35),
         "barak": (37.75, 36.85, 38.05, 37.05)}
CALIB = {"malatya": (38.05, 38.28, 38.45, 38.45), "elbistan_afsin": (36.95, 38.15, 37.35, 38.30),
         "nusaybin": (41.10, 37.03, 41.45, 37.15), "mus": (41.30, 38.70, 41.80, 38.85)}


def plain_of(m, boxes):
    return next((n for n, (a, b, c, d) in boxes.items() if a <= m["lon"] <= c and b <= m["lat"] <= d), "other")


def grouped(counts, groups, n_boot=2000, seed=0):
    counts, groups = np.asarray(counts, float), np.asarray(groups)
    keys = np.unique(groups)
    by = [counts[groups == k] for k in keys]
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        c = np.concatenate([by[i] for i in pick])
        rates.append(c.sum() / (R.KM2 * len(c)))
    return {"n_groups": len(keys), "ci95": [round(float(np.quantile(rates, 0.025)), 4),
                                            round(float(np.quantile(rates, 0.975)), 4)]}


def main():
    _ev = __import__("importlib.util").util
    spec = _ev.spec_from_file_location("ev", ROOT / "13_revision_eval.py")
    EV = _ev.module_from_spec(spec)
    spec.loader.exec_module(EV)
    sets = {}
    for name, boxes in (("fresh", FRESH), ("calib", CALIB)):
        files = sorted((DATA / name).glob("*.npz"))
        metas = [json.loads(str(np.load(f)["meta"])) for f in files]
        sets[name] = (files, [plain_of(m, boxes) for m in metas], [m["post_id"] for m in metas], metas)
    res = {"sets": {}, "grouped": {}, "clipping": {}, "overlap": {}}
    for name, (files, plains, scenes, _) in sets.items():
        res["sets"][name] = {"n_chips": len(files), "n_plains": len(set(plains)), "n_scenes": len(set(scenes)),
                             "n_dates": len({s.split("_")[2][:8] for s in scenes})}
    for spec_ in NETS:
        det, name = EV.make_detector(spec_)
        rv = json.load(open(REV / f"{name}.json"))
        th = rv["thresholds"]
        t = th.get("synthetic_f1_same_rule", th.get("conventional", th.get("synthetic_f1")))
        key = "synthetic_f1_same_rule" if "synthetic_f1_same_rule" in th else (
            "conventional" if "conventional" in th else "synthetic_f1")
        nv = rv["sets"]["nullval"]
        out = {"thr": t}
        c = [r["fixed"][key]["n_patches"] for r in nv]
        pl = [plain_of(r["meta"], PLAINS) for r in nv]
        sc = [r["meta"]["post_id"] for r in nv]
        if "held_out" not in res["sets"]:
            res["sets"]["held_out"] = {"n_chips": len(c), "n_plains": len(set(pl)), "n_scenes": len(set(sc)),
                                       "n_dates": len({s.split("_")[2][:8] for s in sc})}
        out["held_out"] = {"fa_km2": round(sum(c) / (R.KM2 * len(c)), 4), "fa": int(sum(c)),
                           "chips": R.fa_bootstrap(c)["ci95"], "plains": grouped(c, pl), "scenes": grouped(c, sc)}
        for sname, (files, plains, scenes, _) in sets.items():
            c = []
            for f in files:
                pre, post, _, _, _ = R.load_pair(f, coreg=True)
                s = det.score(R.pair_input(pre, post), pre, post)
                c.append(len(R.patches(s >= t, s)))
            out[sname] = {"fa_km2": round(sum(c) / (R.KM2 * len(c)), 4), "fa": int(sum(c)),
                          "chips": R.fa_bootstrap(c)["ci95"], "plains": grouped(c, plains),
                          "scenes": grouped(c, scenes)}
        res["grouped"][name] = out
        print(name, {k: (v["fa_km2"], v["chips"], v["plains"]["ci95"], v["scenes"]["ci95"])
                     for k, v in out.items() if k != "thr"}, flush=True)

    # clipping of the eight bit channels
    def frac(imgs):
        a = np.stack(imgs).reshape(-1, 3)
        return {"low": (a == 0).mean(0).round(6).tolist(), "high": (a == 255).mean(0).round(6).tolist()}
    real = []
    for f in sorted((DATA / "clean").glob("*.npz"))[:400] + sorted((DATA / "fresh").glob("*.npz")):
        pre, post, _, _, _ = R.load_pair(f, coreg=True)
        real.append(R.pair_input(pre, post))
    res["clipping"]["real_clean"] = frac(real)
    ev = []
    for e in ("narli", "siverek"):
        pre, post, lab, _, _ = R.load_pair(DATA / "real" / f"{e}_event.npz", coreg=True)
        img = R.pair_input(pre, post)
        ev.append(img)
        res["clipping"][f"{e}_inside_outline"] = frac([img[lab > 0][None]])
    res["clipping"]["event_chips"] = frac(ev)
    syn = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((DATA / "yolo_gen_v2" / "images" / "train").glob("*"))[:600]]
    res["clipping"]["synthetic_v2_train"] = frac(syn)

    # overlapping footprints within each set
    def overlaps(metas):
        n = 0
        for i in range(len(metas)):
            for j in range(i + 1, len(metas)):
                a, b = metas[i], metas[j]
                if a["epsg"] != b["epsg"]:
                    continue
                x0, y0, x1, y1 = a["bounds"]
                u0, v0, u1, v1 = b["bounds"]
                if x0 < u1 and u0 < x1 and y0 < v1 and v0 < y1:
                    n += 1
        return n
    clean = [json.loads(str(np.load(f)["meta"])) for f in sorted((DATA / "clean").glob("*.npz"))]
    for name, metas in (("clean", clean), ("fresh", sets["fresh"][3]), ("calib", sets["calib"][3])):
        res["overlap"][name] = {"n_chips": len(metas), "overlapping_pairs": overlaps(metas)}
    res["overlap"]["across_sets"] = overlaps(clean + sets["fresh"][3] + sets["calib"][3]) - sum(
        res["overlap"][k]["overlapping_pairs"] for k in ("clean", "fresh", "calib"))
    R.save_json(res, REV / "grouped_bootstrap.json")
    print(json.dumps({k: res[k] for k in ("sets", "clipping", "overlap")}, indent=1))


if __name__ == "__main__":
    main()
