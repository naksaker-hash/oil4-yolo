"""Matched and null calibrated operating points from runs/revision/*.json.

margin      for each event, the spill score minus the highest score of any
            patch on that event's 22 pre-event null pairs
both_fa     false alarms per km2 on the 316 held out clean pairs at the
            highest grid threshold at which both spills pass the standard
            hit rule (from 17_event_grid.py), i.e. the false alarm price of
            detecting both
null_cal    the threshold is set on the held out clean pairs as the lowest
            grid value whose false alarm rate is at most the target, with
            no labelled spill involved, and the standard hit rule is then
            read on the events at that threshold

    .venv/bin/python 16_operating_points.py
"""
import json

import numpy as np

import revlib as R
from config import ROOT

OUT = ROOT / "runs" / "revision"
TARGETS = (0.01, 0.05)


def main():
    eg = json.load(open(OUT / "event_grid.json"))
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.stem in ("physics", "summary", "operating_points", "event_grid"):
            continue
        d = json.load(open(f))
        g = np.array(d["grid"])
        ev = {r["file"].split("_")[0]: r for r in d["sets"]["event"] if r["variant"] == "coreg"}
        nv = np.array([r["counts"] for r in d["sets"]["nullval"]], float)
        fa_nv = nv.sum(0) / (R.KM2 * len(nv))
        out = {}
        for e in ("narli", "siverek"):
            nulls = [r["max_score"] for r in d["sets"]["null22"] if r["file"].startswith(e)]
            out[f"{e}_spill"] = round(ev[e]["spill_max_score"], 4)
            out[f"{e}_null_max"] = round(max(nulls), 4)
            out[f"{e}_margin"] = round(ev[e]["spill_max_score"] - max(nulls), 4)
        hits = eg[d["detector"]]
        both = [j for j in range(len(g)) if hits["narli"]["standard"][j] and hits["siverek"]["standard"][j]]
        out["both_thr"] = float(g[both[-1]]) if both else None
        out["both_fa_km2"] = round(float(fa_nv[both[-1]]), 4) if both else None
        out["both_fa_n"] = int(nv.sum(0)[both[-1]]) if both else None
        for tgt in TARGETS:
            j = np.where(fa_nv <= tgt)[0]
            if len(j):
                t = float(g[j[0]])
                out[f"null_cal_{tgt}"] = {"thr": t, "fa_km2": round(float(fa_nv[j[0]]), 4),
                                          "narli": hits["narli"]["standard"][j[0]],
                                          "siverek": hits["siverek"]["standard"][j[0]]}
            else:
                out[f"null_cal_{tgt}"] = None
        res[d["detector"]] = out
    R.save_json(res, OUT / "operating_points.json")
    print(f"{'detector':36s} {'mN':>7s} {'mS':>7s} {'FA@both':>8s} cal0.01 cal0.05")
    for n, o in res.items():
        c = [o[f"null_cal_{t}"] for t in TARGETS]
        cs = ["".join("N" if x and x["narli"] else "-" for _ in [0]) + ("S" if x and x["siverek"] else "-") for x in c]
        print(f"{n[:36]:36s} {o['narli_margin']:7.3f} {o['siverek_margin']:7.3f} {str(o['both_fa_km2']):>8s} {cs[0]:>7s} {cs[1]:>7s}")


if __name__ == "__main__":
    main()
