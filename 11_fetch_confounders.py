"""Confounder test set requested in review: real changes that mimic oil.

burn    VIIRS 375 m active fire detections (FIRMS, Turkey country files) in
        June to August on farmland, in the two held out validation plains or
        within 10 km of either pipeline, at least 15 km from both spills.
        The pair is the last clear image before the fire and the first clear
        image after it, centred on the fire.
shadow  pairs whose post image contains Sen2Cor cloud shadow (SCL 3) over
        0.5 to 20 % of the chip, in the validation plains.

    .venv/bin/python 11_fetch_confounders.py burn
    .venv/bin/python 11_fetch_confounders.py shadow
"""
import csv
import json
import sys
from datetime import date, timedelta
from math import cos, hypot, radians

import numpy as np

from config import DATA, EVENTS, PLAINS, ROOT
from s2io import bad_fraction, read_pair, search

VAL = ("ceylanpinar", "amik_hatay")
OUT = DATA / "confound"


def km(lon1, lat1, lon2, lat2):
    return hypot((lon1 - lon2) * 111.32 * cos(radians((lat1 + lat2) / 2)), (lat1 - lat2) * 111.32)


def pipeline_points():
    pts = []
    for f in ["pipeline_kirkuk_ceyhan", "pipeline_batman_dortyol"]:
        gj = json.load(open(ROOT / "assets" / f"{f}.geojson"))
        for ft in gj["features"]:
            g = ft["geometry"]
            lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
            for ln in lines:
                pts += ln[::5]
    return np.array(pts)


def in_box(lon, lat, names):
    return any(PLAINS[n][0] <= lon <= PLAINS[n][2] and PLAINS[n][1] <= lat <= PLAINS[n][3] for n in names)


def fires():
    pipe = pipeline_points()
    train = [n for n in PLAINS if n not in VAL]
    sel = []
    for f in sorted((DATA / "firms").glob("viirs_*.csv")):
        for r in csv.DictReader(open(f)):
            d = date.fromisoformat(r["acq_date"])
            if d.month not in (6, 7, 8) or r["type"] != "0" or r["confidence"] == "l":
                continue
            lon, lat = float(r["longitude"]), float(r["latitude"])
            if min(km(lon, lat, e["lon"], e["lat"]) for e in EVENTS.values()) < 15:
                continue
            near_pipe = np.min(np.hypot((pipe[:, 0] - lon) * 111.32 * cos(radians(lat)),
                                        (pipe[:, 1] - lat) * 111.32)) < 10
            if not (in_box(lon, lat, VAL) or (near_pipe and not in_box(lon, lat, train))):
                continue
            sel.append((d, lon, lat))
    # one fire per 5 km and per year, spread over years
    keep = []
    for d, lon, lat in sorted(sel):
        if all(not (k[0].year == d.year and km(lon, lat, k[1], k[2]) < 5) for k in keep):
            keep.append((d, lon, lat))
    return keep


def burn(limit=80):
    OUT.mkdir(exist_ok=True)
    cand = fires()
    rng = np.random.default_rng(0)
    rng.shuffle(cand)
    got = len(list(OUT.glob("burn_*.npz")))
    for d, lon, lat in cand:
        if got >= limit:
            break
        f = OUT / f"burn_{d}_{lon:.3f}_{lat:.3f}.npz"
        if f.exists():
            continue
        try:
            items = search(lon, lat, str(d - timedelta(days=12)), str(d + timedelta(days=12)), max_cloud=10)
            days = sorted({it["properties"]["datetime"][:10] for it in items})
            pre = [x for x in days if x < str(d)]
            post = [x for x in days if x > str(d)]
            if not pre or not post:
                continue
            a, b = pre[-1], post[0]
            P, Q, sp, sq, meta = read_pair(lon, lat, a, b)
        except Exception as e:  # noqa: BLE001
            print("fail", f.name, e)
            continue
        b1, b2 = bad_fraction(sp, P), bad_fraction(sq, Q)
        if max(b1, b2) > 0.05:
            continue
        np.savez_compressed(f, pre=P, post=Q, scl_pre=sp, scl_post=sq,
                            mask=np.zeros(P.shape[1:], np.uint8),
                            meta=json.dumps({**meta, "pre": a, "post": b, "fire_date": str(d),
                                             "fire_lon": lon, "fire_lat": lat}))
        got += 1
        print(got, f.name, a, b, flush=True)


def shadow(limit=40):
    OUT.mkdir(exist_ok=True)
    got = len(list(OUT.glob("shadow_*.npz")))
    rng = np.random.default_rng(1)
    for year in [2019, 2020, 2023, 2024]:
        for name in VAL:
            a0, b0, a1, b1 = PLAINS[name]
            for _ in range(12):
                if got >= limit:
                    return
                lon, lat = rng.uniform(a0, a1), rng.uniform(b0, b1)
                try:
                    items = search(lon, lat, f"{year}-06-01", f"{year}-08-31", max_cloud=40)
                except Exception:  # noqa: BLE001
                    continue
                days = sorted({it["properties"]["datetime"][:10] for it in items})
                for x, y in zip(days, days[1:]):
                    cc = {it["properties"]["datetime"][:10]: it["properties"]["eo:cloud_cover"] for it in items}
                    if not (cc.get(x, 99) < 2 and 3 <= cc.get(y, 0) <= 40):
                        continue
                    try:
                        P, Q, sp, sq, meta = read_pair(lon, lat, x, y)
                    except Exception:  # noqa: BLE001
                        continue
                    shf = float((sq == 3).mean())
                    cloud = float(np.isin(sq, [8, 9, 10]).mean())
                    if 0.005 <= shf <= 0.2 and cloud <= 0.3 and bad_fraction(sp, P) < 0.02:
                        f = OUT / f"shadow_{x}_{lon:.3f}_{lat:.3f}.npz"
                        np.savez_compressed(f, pre=P, post=Q, scl_pre=sp, scl_post=sq,
                                            mask=np.zeros(P.shape[1:], np.uint8),
                                            meta=json.dumps({**meta, "pre": x, "post": y,
                                                             "shadow_frac": shf, "cloud_frac": cloud}))
                        got += 1
                        print(got, f.name, f"shadow {shf:.3f} cloud {cloud:.3f}", flush=True)
                        break


if __name__ == "__main__":
    {"burn": burn, "shadow": shadow}[sys.argv[1]]()
