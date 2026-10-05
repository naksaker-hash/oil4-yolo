"""Fetch clean Sentinel-2 image pairs over southeast Turkiye farmland, the
background onto which synthetic plumes are implanted.

No chip lies within EXCLUDE_KM of either real spill, and no chip is from
2018 or 2021, the two event years. A chip is kept only if both dates are
at least 98 % clear by SCL, with class 2 (dark area) counted as clear.

    for s in 1 2 3 4 5 6; do python 03_fetch_train.py --n 250 --seed $s & done
"""
import argparse
import json
import math
import random
from datetime import date

import numpy as np

from config import (CHIP, DATA, EVENTS, EXCLUDE_KM, MONTHS, PAIR_GAP_DAYS,
                    PLAINS, YEARS)
from s2io import bad_fraction, chip_bounds, read_chip, search, tile_of


def km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def random_point(rng):
    while True:
        b = PLAINS[rng.choice(sorted(PLAINS))]
        lon, lat = rng.uniform(b[0], b[2]), rng.uniform(b[1], b[3])
        if all(km(lat, lon, e["lat"], e["lon"]) > EXCLUDE_KM for e in EVENTS.values()):
            return lon, lat


def random_pair(rng, lon, lat):
    y = rng.choice(YEARS)
    m = rng.randint(*MONTHS)
    start = date(y, m, 1)
    end = date(y, m + 1, 10) if m < 12 else date(y, 12, 31)
    items = search(lon, lat, start.isoformat(), end.isoformat(), max_cloud=20)
    items = [f for f in items if not f["id"].endswith("_1_L2A")]
    by_tile = {}
    for f in items:
        by_tile.setdefault(tile_of(f), []).append(f)
    cands = []
    for fs in by_tile.values():
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                gap = (date.fromisoformat(b["properties"]["datetime"][:10]) -
                       date.fromisoformat(a["properties"]["datetime"][:10])).days
                if PAIR_GAP_DAYS[0] <= gap <= PAIR_GAP_DAYS[1]:
                    cands.append((a, b, gap))
    return rng.choice(cands) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = DATA / "clean"
    out.mkdir(parents=True, exist_ok=True)
    # several workers may share the folder, each with its own seed and prefix
    have = len(list(out.glob(f"s{args.seed}_*.npz")))
    # offset by the chips already on disk so a restart never repeats a draw
    rng = random.Random(args.seed * 1_000_003 + have)
    tries = 0
    while have < args.n and tries < args.n * 6:
        tries += 1
        lon, lat = random_point(rng)
        try:
            p = random_pair(rng, lon, lat)
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
        except Exception as e:          # network hiccups are expected
            print("skip", type(e).__name__, e)
            continue
        f = out / f"s{args.seed}_{have:05d}.npz"
        meta = {"lon": lon, "lat": lat, "epsg": epsg, "bounds": bounds,
                "tile": tile_of(a), "pre_id": a["id"], "post_id": b["id"], "gap": gap}
        np.savez_compressed(f, pre=pre, post=post, meta=json.dumps(meta))
        have += 1
        print(f"{f.name} {meta['tile']} {a['id'][-24:]} gap {gap} d")


if __name__ == "__main__":
    main()
