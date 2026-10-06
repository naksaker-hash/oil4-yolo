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
    for name, regime, prof in [("v1", "gen", "v1"), ("v2", "gen", "v2"), ("dr", "gen", "dr"),
                               ("narli", "narli", "v1"), ("siverek", "siverek", "v2")]:
        rng = np.random.default_rng(0)
        ks = np.array([S.core_k(rng, regime, k_meas, prof) for _ in range(N)])
        v, s = ks[:, vn].mean(1), ks[:, sw].mean(1)
        res[name] = round(float(((v <= 0.6) & (s - v >= 0.35)).mean()), 4)
    print(res)
    R.save_json(res, ROOT / "runs" / "revision" / "coverage.json")


if __name__ == "__main__":
    main()
