"""Null calibration on an independent set (external review, 13 Oct 2026).

The null calibrated threshold of the paper was set on the 316 held out
pairs, whose synthetic versions also chose the best epoch and the fixed
threshold. Here it is set instead on 300 pairs from four further plains
(19_fetch_fresh.py --set calib), which took no part in training or model
selection, and its false alarms are counted on the 150 fresh pairs, and the
other way round. For each network the threshold is the lowest grid value
whose false alarm rate on the calibration set is at most the target, and
the standard hit rule on the two spills is read at that threshold from
runs/revision/event_grid.json.

    .venv/bin/python 28_calibration_set.py
"""
import json

import numpy as np

import revlib as R
from config import ROOT

REV = ROOT / "runs" / "revision"
TARGET = 0.01


def fa_grid(d):
    return np.array(d["fa_grid_km2"])


def cal(grid, fa, target=TARGET):
    j = np.where(fa <= target)[0]
    return int(j[0]) if len(j) else None


def main():
    eg = json.load(open(REV / "event_grid.json"))
    res = {}
    for f in sorted((ROOT / "runs" / "revision_calib").glob("*.json")):
        name = f.stem
        c = json.load(open(f))
        fr = json.load(open(ROOT / "runs" / "revision_fresh" / f"{name}.json"))
        rv = json.load(open(REV / f"{name}.json"))
        g = np.array(rv["grid"])
        assert np.allclose(g, c["grid"]) and np.allclose(g, fr["grid"])
        nv = np.array([r["counts"] for r in rv["sets"]["nullval"]], float)
        fa = {"held_out": nv.sum(0) / (R.KM2 * len(nv)), "calib": fa_grid(c), "fresh": fa_grid(fr)}
        hits = eg.get(rv["detector"])
        out = {"n_calib": c["n_pairs"], "n_fresh": fr["n_pairs"]}
        for src, dst in (("held_out", "fresh"), ("calib", "fresh"), ("fresh", "calib"), ("calib", "held_out")):
            j = cal(g, fa[src])
            if j is None:
                out[f"{src}->{dst}"] = None
                continue
            o = {"thr": float(g[j]), "fa_cal": round(float(fa[src][j]), 4),
                 "fa_count": round(float(fa[dst][j]), 4), "binding": j > 0}
            if hits:
                o["narli"] = bool(hits["narli"]["standard"][j])
                o["siverek"] = bool(hits["siverek"]["standard"][j])
            out[f"{src}->{dst}"] = o
        res[name] = out
    R.save_json(res, REV / "calibration_set.json")
    print(f"{'detector':38s} {'held_out->fresh':>24s} {'calib->fresh':>24s} {'fresh->calib':>24s}")
    for n, o in res.items():
        cells = []
        for k in ("held_out->fresh", "calib->fresh", "fresh->calib"):
            x = o[k]
            cells.append("none" if x is None else
                         f"{x['thr']:.2f} {x['fa_count']:.4f} {'N' if x.get('narli') else '-'}{'S' if x.get('siverek') else '-'}{'' if x['binding'] else '*'}")
        print(f"{n[:38]:38s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
