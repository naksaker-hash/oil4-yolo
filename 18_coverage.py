"""Coverage of the measured spill attenuation by each generator, a measure
of the simulation gap that needs one real target and no training.

Both spills share one pattern after background correction (Table S1):
mean visible and near infrared factor near 0.4 to 0.5, and a shortwave
infrared factor about 0.4 higher. The coverage of a generator is the
fraction of its attenuation draws with a mean visible and near infrared
factor of at most 0.6 and a shortwave infrared factor at least 0.35 above it.

    .venv/bin/python 18_coverage.py
"""
import json

import numpy as np

import revlib as R
from config import DATA, ROOT

S = R.synth
N = 20000


def main():
    k_meas = {e: list(json.load(open(DATA / f"k_{e}.json"))["k"].values()) for e in ("narli", "siverek")}
    sw = list(S.SWIR)
    vn = [i for i in range(len(S.BANDS)) if i not in sw]
    res = {}
    draws = {}
    for name, regime, prof, off in [("v1", "gen", "v1", ()), ("v2", "gen", "v2", ()), ("dr", "gen", "dr", ()),
                                    ("v2_noswir", "gen", "v2", ("swir",)),
                                    ("narli", "narli", "v1", ()), ("siverek", "siverek", "v2", ())]:
        rng = np.random.default_rng(0)
        ks = np.array([S.core_k(rng, regime, k_meas, prof, off) for _ in range(N)])
        v, s = ks[:, vn].mean(1), ks[:, sw].mean(1)
        res[name] = round(float(((v <= 0.6) & (s - v >= 0.35)).mean()), 4)
        draws[name] = (v, s)
    print(res)
    R.save_json(res, ROOT / "runs" / "revision" / "coverage.json")
    # sensitivity to the two cut-offs, which were read off the spills
    order = ["dr", "v1", "v2_noswir", "v2", "narli", "siverek"]
    grid = {}
    kept = 0
    cells = 0
    for vmax in (0.5, 0.55, 0.6, 0.65, 0.7):
        for gap in (0.25, 0.3, 0.35, 0.4, 0.45):
            c = {n: round(float(((draws[n][0] <= vmax) & (draws[n][1] - draws[n][0] >= gap)).mean()), 4) for n in order}
            grid[f"{vmax}_{gap}"] = c
            vals = [c[n] for n in order]
            ok = all(vals[i] <= vals[i + 1] for i in range(3))   # dr <= v1 <= noswir <= v2
            ok2 = c["v2"] > max(c["v1"], c["dr"])
            kept += ok
            cells += 1
            grid[f"{vmax}_{gap}"]["order_dr_v1_noswir_v2"] = ok
            grid[f"{vmax}_{gap}"]["v2_above_v1_dr"] = ok2
    grid["_n_cells"] = cells
    grid["_n_full_order"] = kept
    grid["_n_v2_above"] = sum(grid[k]["v2_above_v1_dr"] for k in grid if not k.startswith("_"))
    R.save_json(grid, ROOT / "runs" / "revision" / "coverage_grid.json")
    print("full order kept in", kept, "of", cells, "; v2 above v1 and dr in", grid["_n_v2_above"])


if __name__ == "__main__":
    main()
