"""Null calibration on one held out plain, false alarms read on the other.

The 316 held out pairs of Ceylanpinar and Amik also formed (with synthetic
plumes) the validation set used to pick the best epoch and the fixed
threshold of every network. Calibrating the threshold on one plain and
counting false alarms on the other removes the calibration step from the
reported rate; the fresh plains (19_fetch_fresh.py) remove model selection
as well.

    .venv/bin/python 20_split_calibration.py
"""
import json

import numpy as np

import revlib as R
from config import ROOT

OUT = ROOT / "runs" / "revision"
PL = ("ceylanpinar", "amik_hatay")


def main():
    eg = json.load(open(OUT / "event_grid.json"))
    res = {}
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f))
        if "detector" not in d:
            continue
        g = np.array(d["grid"])
        rows = d["sets"]["nullval"]
        by = {p: np.array([r["counts"] for r in rows if R.synth.plain_of(r["meta"]) == p], float) for p in PL}
        fa = {p: by[p].sum(0) / (R.KM2 * len(by[p])) for p in PL}
        hits = eg[d["detector"]]
        both = [j for j in range(len(g)) if hits["narli"]["standard"][j] and hits["siverek"]["standard"][j]]
        out = {"n": {p: len(by[p]) for p in PL}}
        if both:
            out["both_fa_by_plain"] = {p: round(float(fa[p][both[-1]]), 4) for p in PL}
        for cal, rep in (PL, PL[::-1]):
            j = np.where(fa[cal] <= 0.01)[0]
            if len(j):
                j = j[0]
                out[f"cal_{cal}"] = {"thr": float(g[j]), "fa_cal": round(float(fa[cal][j]), 4),
                                     "fa_other": round(float(fa[rep][j]), 4),
                                     "narli": hits["narli"]["standard"][j],
                                     "siverek": hits["siverek"]["standard"][j]}
        res[d["detector"]] = out
    R.save_json(res, OUT / "split_calibration.json")
    for n, o in res.items():
        a, b = o.get("cal_ceylanpinar"), o.get("cal_amik_hatay")
        fmt = lambda x: f"t{x['thr']:.2f} FAo {x['fa_other']:.4f} {'N' if x['narli'] else '-'}{'S' if x['siverek'] else '-'}" if x else "none"
        print(f"{n[:34]:34s} n={o['n']} both={o.get('both_fa_by_plain')} | C->A {fmt(a)} | A->C {fmt(b)}")


if __name__ == "__main__":
    main()
