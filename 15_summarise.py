"""Compact summary of runs/revision/*.json for the manuscript.

    .venv/bin/python 15_summarise.py
"""
import json
from pathlib import Path

import numpy as np

import revlib as R
from config import ROOT

OUT = ROOT / "runs" / "revision"


def scene_rank(rows):
    out = {}
    for ev in ("narli", "siverek"):
        rs = [r for r in rows if r["file"].startswith(ev)]
        if not rs:
            continue
        sp = max(r.get("spill_max_score", -np.inf) for r in rs)
        oth = [r["outside_max_score"] if "spill_max_score" in r else r["max_score"] for r in rs]
        out[ev] = int(sum(o > sp for o in oth))
    return out


def summarise(d, key):
    ev = {(r["file"].split("_")[0], r["variant"]): r["fixed"][key] for r in d["sets"]["event"]}
    s = {"thr": round(d["thresholds"][key], 3)}
    for e in ("narli", "siverek"):
        f = ev[(e, "coreg")]
        s[e] = {"hit": f["standard"]["hit"], "iou": f["standard"]["iou"], "rank": f["standard"]["rank"],
                "strict": f["strict"]["hit"], "centroid": f["centroid"]["hit"],
                "raw_hit": ev[(e, "raw")]["standard"]["hit"]}
    for st in ("null22", "nullval", "burn", "shadow"):
        rows = d["sets"].get(st, [])
        if not rows:
            continue
        b = R.fa_bootstrap([r["fixed"][key]["n_patches"] for r in rows])
        s[st] = {"fa_km2": b["fa_per_km2"], "ci": b["ci95"], "fa": b["fa"]}
        if st == "nullval" and rows[0]["fixed"][key]["n_cloudmasked"] is not None:
            s[st]["cloudmasked_fa_km2"] = round(sum(r["fixed"][key]["n_cloudmasked"] for r in rows) / (R.KM2 * len(rows)), 4)
        if st == "burn":
            s[st]["at_fire"] = f"{sum(r['fixed'][key]['at_fire'] for r in rows)}/{len(rows)}"
        if st == "shadow":
            s[st]["on_shadow"] = sum(r["fixed"][key]["on_shadow"] for r in rows)
    later = {}
    for r in d["sets"].get("later", []):
        later[r["file"][:-4]] = r["fixed"][key]["standard"]["hit"]
    s["later_hits"] = [k for k, v in later.items() if v]
    return s


def main():
    allres = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name in ("physics.json", "summary.json", "operating_points.json", "event_grid.json"):
            continue
        d = json.load(open(f))
        name = d["detector"]
        res = {"scene_rank_spill_exceeded_by": scene_rank(d["sets"].get("scene", []))}
        for key in [k for k in d["thresholds"] if not k.startswith("_")]:
            res[key] = summarise(d, key)
        if "_calibration" in d["thresholds"]:
            res["ece"] = d["thresholds"]["_calibration"]["ece"]
        allres[name] = res
    R.save_json(allres, OUT / "summary.json")
    for n, r in allres.items():
        for k, s in r.items():
            if not isinstance(s, dict) or "thr" not in s:
                continue
            na, sv = s["narli"], s["siverek"]
            nv = s.get("nullval", {})
            bu = s.get("burn", {})
            print(f"{n[:34]:34s} {k[:22]:22s} t={s['thr']:<7} "
                  f"N {'H' if na['hit'] else '-'}{na['iou'] or '':<6} S {'H' if sv['hit'] else '-'}{sv['iou'] or '':<6} "
                  f"nullval {nv.get('fa_km2', ''):<7} burn {bu.get('at_fire', '')} later {len(s['later_hits'])} "
                  f"scene {r['scene_rank_spill_exceeded_by']}")


if __name__ == "__main__":
    main()
