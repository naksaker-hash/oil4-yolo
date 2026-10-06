"""Hit rule on the two event chips at every threshold of the grid, for every
detector in runs/revision, so that operating points can be matched on the
hit rule itself rather than on the highest spill score.

    .venv/bin/python 17_event_grid.py            # all detectors not yet done
"""
import importlib.util
import json

import numpy as np

import revlib as R
from config import DATA, ROOT

_spec = importlib.util.spec_from_file_location("ev", ROOT / "13_revision_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

OUT = ROOT / "runs" / "revision"


def spec_of(name):
    if name.startswith("yolo_"):
        return "yolo:" + name[5:]
    if name.startswith("rf_"):
        return "rf:" + name[3:]
    return name


def main():
    store = OUT / "event_grid.json"
    res = json.load(open(store)) if store.exists() else {}
    for f in sorted(OUT.glob("*.json")):
        d = json.load(open(f)) if f.stem not in ("physics", "summary", "operating_points", "event_grid") else None
        if d is None or "detector" not in d or d["detector"] in res:
            continue
        det, name = ev.make_detector(spec_of(d["detector"]))
        g = np.array(d["grid"])
        out = {"grid": g.tolist()}
        for e in ("narli", "siverek"):
            pre, post, mask, _, _ = R.load_pair(DATA / "real" / f"{e}_event.npz", coreg=True)
            s = det.score(R.pair_input(pre, post), pre, post)
            rows = {rule: [] for rule in ("standard", "strict", "centroid")}
            for t in g:
                pts = R.patches(s >= t, s)
                for rule in rows:
                    rows[rule].append(R.judge(pts, mask, rule)["hit"])
            out[e] = rows
        res[name] = out
        R.save_json(res, store)
        top = {e: max([t for t, h in zip(g, out[e]["standard"]) if h], default=None) for e in ("narli", "siverek")}
        print(name, top, flush=True)


if __name__ == "__main__":
    main()
