"""Read co-registered Sentinel-2 L2A chips from Earth Search COGs.

Windowed reads over HTTP from cloud optimised GeoTIFFs, no Earth Engine
and no Drive round trip.
"""
import json
import os
import urllib.request
from datetime import date, timedelta

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

from config import ASSET, BANDS, CHIP, COLLECTION, FALLBACK, PIX_M, SCL_BAD, STAC

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def search(lon, lat, start, end, max_cloud=40, limit=200, collection=COLLECTION):
    """Items over a point between two ISO dates."""
    body = {"collections": [collection],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": limit,
            "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "oil4-yolo/1.0"})
    feats = json.load(urllib.request.urlopen(req, timeout=120))["features"]
    return sorted(feats, key=lambda f: f["properties"]["datetime"])


def item_on(lon, lat, day, tile=None):
    """The item acquired on one calendar day, optionally on a given tile."""
    d = date.fromisoformat(day)
    # Collection 1 has gaps in 2018; fall back to the original L2A archive,
    # whose scale and offset are read per asset, so values stay comparable.
    for col in (COLLECTION, FALLBACK):
        items = search(lon, lat, day, (d + timedelta(days=1)).isoformat(),
                       max_cloud=101, collection=col)
        items = [f for f in items if f["properties"]["datetime"][:10] == day
                 and not f["id"].endswith("_1_L2A")]
        if tile:
            items = [f for f in items if tile_of(f) == tile]
        if items:
            break
    if not items:
        raise LookupError(f"no {COLLECTION} item on {day} at {lat},{lon} tile {tile}")
    return items[0]


def tile_of(item):
    p = item["properties"]
    return p.get("grid:code", "").replace("MGRS-", "") or p.get("s2:mgrs_tile", "")


def _scale_offset(asset):
    rb = (asset.get("raster:bands") or [{}])[0]
    return rb.get("scale", 1e-4), rb.get("offset", 0.0)


def chip_bounds(item, lon, lat, size=CHIP):
    epsg = item["properties"]["proj:epsg"] if "proj:epsg" in item["properties"] \
        else int(item["properties"]["proj:code"].split(":")[1])
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    # snap the centre to the 10 m grid so pre and post share pixel edges
    cx = round(xs[0] / PIX_M) * PIX_M
    cy = round(ys[0] / PIX_M) * PIX_M
    h = size * PIX_M / 2
    return epsg, (cx - h, cy - h, cx + h, cy + h)


def read_chip(item, bounds, size=CHIP):
    """Ten bands as float32 reflectance (10, size, size) and the SCL layer."""
    out = np.empty((len(BANDS), size, size), np.float32)
    for i, b in enumerate(BANDS):
        a = item["assets"][ASSET[b]]
        scale, offset = _scale_offset(a)
        with rasterio.open(a["href"]) as src:
            w = from_bounds(*bounds, transform=src.transform)
            v = src.read(1, window=w, out_shape=(size, size),
                         resampling=Resampling.bilinear, boundless=True, fill_value=0)
        out[i] = np.where(v == 0, np.nan, v * scale + offset)
    with rasterio.open(item["assets"]["scl"]["href"]) as src:
        w = from_bounds(*bounds, transform=src.transform)
        scl = src.read(1, window=w, out_shape=(size, size),
                       resampling=Resampling.nearest, boundless=True, fill_value=0)
    return out, scl


def bad_fraction(scl, refl):
    bad = np.isin(scl, list(SCL_BAD)) | ~np.isfinite(refl).all(0)
    return float(bad.mean())


def read_pair(lon, lat, pre_day, post_day, size=CHIP):
    """Pre and post chips on the same MGRS tile and the same 10 m grid."""
    pre_item = item_on(lon, lat, pre_day)
    post_item = item_on(lon, lat, post_day, tile=tile_of(pre_item))
    epsg, bounds = chip_bounds(pre_item, lon, lat, size)
    pre, scl_pre = read_chip(pre_item, bounds, size)
    post, scl_post = read_chip(post_item, bounds, size)
    meta = {"epsg": epsg, "bounds": bounds, "tile": tile_of(pre_item),
            "pre_id": pre_item["id"], "post_id": post_item["id"]}
    return pre, post, scl_pre, scl_post, meta
