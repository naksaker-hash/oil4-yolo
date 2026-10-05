"""Extra real imagery requested in review.

scene   a 7 x 7 grid of chips (about 18 x 18 km) centred on each spill, on
        the event dates, to rank the spill among every patch of a scene
later   the original pre-event image paired with every later clear image
        up to 40 days after the spill, to test oil age and weathering

    .venv/bin/python 10_fetch_extra.py scene
    .venv/bin/python 10_fetch_extra.py later
"""
import importlib.util
import json
import sys
from datetime import date, timedelta

import numpy as np

from config import CHIP, DATA, EVENTS, PIX_M, ROOT
from s2io import bad_fraction, read_pair, search

_spec = importlib.util.spec_from_file_location("fr", ROOT / "02_fetch_real.py")
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

N = 7


def save(f, pre, post, sp, sq, meta, label, extra):
    mask = fr.plume_mask(label, meta["epsg"], meta["bounds"])
    np.savez_compressed(f, pre=pre, post=post, scl_pre=sp, scl_post=sq, mask=mask,
                        meta=json.dumps({**meta, **extra}))
    return int(mask.sum()), bad_fraction(sp, pre), bad_fraction(sq, post)


def scene():
    out = DATA / "scene"
    out.mkdir(exist_ok=True)
    step_km = CHIP * PIX_M / 1000
    for name, ev in EVENTS.items():
        dlat = step_km / 111.32
        dlon = step_km / (111.32 * np.cos(np.radians(ev["lat"])))
        for r in range(N):
            for c in range(N):
                f = out / f"{name}_r{r}c{c}.npz"
                if f.exists():
                    continue
                lat = ev["lat"] + (N // 2 - r) * dlat
                lon = ev["lon"] + (c - N // 2) * dlon
                try:
                    pre, post, sp, sq, meta = read_pair(lon, lat, ev["pre"], ev["post"])
                except Exception as e:  # noqa: BLE001
                    print("fail", f.name, e)
                    continue
                px, b1, b2 = save(f, pre, post, sp, sq, meta, ev["label"],
                                  {"pre": ev["pre"], "post": ev["post"], "row": r, "col": c})
                print(f.name, "plume px", px, f"bad {b1:.3f}/{b2:.3f}", flush=True)


def later():
    out = DATA / "later"
    out.mkdir(exist_ok=True)
    for name, ev in EVENTS.items():
        p0 = date.fromisoformat(ev["post"])
        items = search(ev["lon"], ev["lat"], str(p0 + timedelta(days=1)), str(p0 + timedelta(days=40)))
        days = sorted({it["properties"]["datetime"][:10] for it in items})
        for d in days:
            f = out / f"{name}_{d}.npz"
            if f.exists():
                continue
            try:
                pre, post, sp, sq, meta = read_pair(ev["lon"], ev["lat"], ev["pre"], d)
            except Exception as e:  # noqa: BLE001
                print("fail", d, e)
                continue
            px, b1, b2 = save(f, pre, post, sp, sq, meta, ev["label"],
                              {"pre": ev["pre"], "post": d,
                               "days_after_spill": (date.fromisoformat(d) - date.fromisoformat(ev["date"])).days})
            print(f.name, f"bad {b1:.3f}/{b2:.3f}", flush=True)


if __name__ == "__main__":
    {"scene": scene, "later": later}[sys.argv[1]]()
