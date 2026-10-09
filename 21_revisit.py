"""How often a clear Sentinel-2 pair brackets a spill within a week.

For the centre of each of the ten plains and each summer (June to
September of 2019, 2020 and 2023 to 2025), every acquisition with scene
cloud cover below 10 % is listed. For every day d of the season, a spill
on day d is observable by the networks if a clear image falls on days
d + 1 to d + 7 and another clear image falls within 11 days before d,
the pair gap of the training data. Scene cloud cover is a proxy for the
chip level clear fraction, so the result is approximate.

    .venv/bin/python 21_revisit.py
"""
import json
from datetime import date, timedelta

import numpy as np

import revlib as R
from config import PLAINS, ROOT
from s2io import search

YEARS = (2019, 2020, 2023, 2024, 2025)


def main():
    res = {}
    allv = []
    for name, (a, b, c, d) in PLAINS.items():
        lon, lat = (a + c) / 2, (b + d) / 2
        vals = []
        for y in YEARS:
            items = search(lon, lat, f"{y}-05-15", f"{y}-10-10", max_cloud=10, limit=300)
            days = sorted({date.fromisoformat(f["properties"]["datetime"][:10]) for f in items})
            ds = set(days)
            day = date(y, 6, 1)
            while day <= date(y, 9, 30):
                post = any(day + timedelta(k) in ds for k in range(1, 8))
                pre = any(day - timedelta(k) in ds for k in range(0, 12))
                vals.append(post and pre)
                day += timedelta(1)
        res[name] = round(float(np.mean(vals)), 4)
        allv += vals
        print(name, res[name], flush=True)
    res["_all"] = round(float(np.mean(allv)), 4)
    print("all", res["_all"])
    R.save_json(res, ROOT / "runs" / "revision" / "revisit.json")


if __name__ == "__main__":
    main()
