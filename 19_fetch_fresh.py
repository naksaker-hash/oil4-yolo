"""Fetch a fresh set of clean pairs from three plains never used before,
for false alarm counting only. Neither the networks, their best epochs,
their thresholds nor the null calibrated thresholds saw these chips, unlike
the 316 held out pairs of Ceylanpinar and Amik, whose synthetic versions
formed the validation set used for model selection.

    for s in 1 2 3; do .venv/bin/python 19_fetch_fresh.py --n 50 --seed $s & done
"""
import argparse
import importlib.util
import json
import random

import numpy as np

import config
from config import DATA, EVENTS
from s2io import bad_fraction, chip_bounds, read_chip, tile_of

FRESH_PLAINS = {
    "bismil": (40.62, 37.80, 41.10, 38.00),
    "cizre_silopi": (42.00, 37.15, 42.45, 37.35),
    "barak": (37.75, 36.85, 38.05, 37.05),
}

_spec = importlib.util.spec_from_file_location("fetch", config.ROOT / "03_fetch_train.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


def random_point(rng):
    while True:
        b = FRESH_PLAINS[rng.choice(sorted(FRESH_PLAINS))]
        lon, lat = rng.uniform(b[0], b[2]), rng.uniform(b[1], b[3])
        if all(F.km(lat, lon, e["lat"], e["lon"]) > config.EXCLUDE_KM for e in EVENTS.values()):
            return lon, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = DATA / "fresh"
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob(f"f{args.seed}_*.npz")))
    rng = random.Random(args.seed * 7_000_003 + have)
    tries = 0
    while have < args.n and tries < args.n * 8:
        tries += 1
        lon, lat = random_point(rng)
        try:
            p = F.random_pair(rng, lon, lat)
            if p is None:
                continue
            a, b, gap = p
            epsg, bounds = chip_bounds(a, lon, lat)
            pre, sp = read_chip(a, bounds)
            if bad_fraction(sp, pre) > 0.02:
                continue
            post, sq = read_chip(b, bounds)
            if bad_fraction(sq, post) > 0.02:
                continue
        except Exception as e:
            print("skip", type(e).__name__, e, flush=True)
            continue
        f = out / f"f{args.seed}_{have:05d}.npz"
        meta = {"lon": lon, "lat": lat, "epsg": epsg, "bounds": bounds,
                "tile": tile_of(a), "pre_id": a["id"], "post_id": b["id"], "gap": gap}
        np.savez_compressed(f, pre=pre, post=post, meta=json.dumps(meta))
        have += 1
        print(f"{f.name} {meta['tile']} gap {gap} d", flush=True)


if __name__ == "__main__":
    main()
