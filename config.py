"""Shared constants: YOLO detection of terrestrial oil spills trained on
synthetic plumes and tested on the Narli 2018 and Siverek 2021 events.

Event coordinates, dates, null pairs and the Siverek attenuation ratios
come from earlier analyses of the two spills.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LABELS = ROOT / "labels"

STAC = "https://earth-search.aws.element84.com/v1/search"
# Collection 1 is a uniform reprocessing (baseline 05.00), but it has gaps:
# in May and June 2018 at Narli it lacks 05-16, 05-21, 05-26 and 06-05, so
# those dates fall back to the original archive (baseline 02.07/02.08,
# offset 0, older geolocation). Five of the six Narli null pairs include
# such a scene and are misregistered by 0.5 to 0.75 px; realio.py
# co-registers them. The event pairs, all Siverek pairs and all training
# pairs are Collection 1. SCL class 2 is never treated as cloud, because
# processing baseline changes move Sen2Cor class 2 (dark area) onto and off the oil.
COLLECTION = "sentinel-2-c1-l2a"
FALLBACK = "sentinel-2-l2a"

# Ten bands at 10 and 20 m. Earth Search asset names.
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
ASSET = {"B2": "blue", "B3": "green", "B4": "red", "B5": "rededge1",
         "B6": "rededge2", "B7": "rededge3", "B8": "nir", "B8A": "nir08",
         "B11": "swir16", "B12": "swir22"}
WAVELENGTH_NM = [492, 560, 665, 704, 740, 783, 833, 865, 1610, 2190]

CHIP = 256          # pixels at 10 m, 2.56 km on a side
PIX_M = 10.0

# SCL classes counted as unusable. 2 (dark area) is deliberately absent.
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}

EVENTS = {
    "narli": {
        # plume centroid
        "lat": 37.350183, "lon": 37.130123,
        "date": "2018-06-19",
        "pre": "2018-06-15", "post": "2018-06-20",
        # six pre-event pairs
        "null_pairs": [("2018-05-16", "2018-05-21"), ("2018-05-21", "2018-05-26"),
                       ("2018-05-26", "2018-05-31"), ("2018-05-31", "2018-06-05"),
                       ("2018-06-05", "2018-06-10"), ("2018-06-10", "2018-06-15")],
        "label": "narli_plume.geojson",   # oil class rebuilt by spectral matching, 218 px at 20 m, 8.72 ha
        "substrate": "harvested farmland",
    },
    "siverek": {
        # plume centroid
        "lat": 37.667333, "lon": 39.241549,
        "date": "2021-08-13",
        "pre": "2021-08-05", "post": "2021-08-15",
        # sixteen pre-event pairs
        "null_pairs": [("2021-05-07", "2021-05-17"), ("2021-05-17", "2021-05-27"),
                       ("2021-05-27", "2021-06-01"), ("2021-06-01", "2021-06-06"),
                       ("2021-06-06", "2021-06-11"), ("2021-06-11", "2021-06-16"),
                       ("2021-06-16", "2021-06-21"), ("2021-06-21", "2021-06-26"),
                       ("2021-06-26", "2021-07-01"), ("2021-07-01", "2021-07-06"),
                       ("2021-07-06", "2021-07-11"), ("2021-07-11", "2021-07-16"),
                       ("2021-07-16", "2021-07-21"), ("2021-07-21", "2021-07-26"),
                       ("2021-07-26", "2021-07-31"), ("2021-07-31", "2021-08-05")],
        "label": "siverek_plume.geojson",  # 0.49 ha core
        "substrate": "irrigated maize",
    },
}

# Siverek core attenuation k = R(15 Aug) / R(05 Aug), mean core spectra.
K_SIVEREK_CORE = [0.0142 / 0.0511, 0.0208 / 0.0738, 0.0247 / 0.0768,
                  0.0402 / 0.1032, 0.0740 / 0.2327, 0.0947 / 0.3069,
                  0.0954 / 0.3119, 0.1092 / 0.3112, 0.1195 / 0.1685,
                  0.0837 / 0.1111]
# Narli k is measured from the event chip by 02_fetch_real.py and written to
# data/k_narli.json, so it never has to be typed in by hand.

# No real spill may fall inside a training chip.
EXCLUDE_KM = 15.0

# Farmland plains of southeast Turkiye used for synthetic training pairs,
# as lon_min, lat_min, lon_max, lat_max. Both test sites lie inside the
# Pazarcik and Siverek boxes and are removed by EXCLUDE_KM.
PLAINS = {
    "harran":     (38.70, 36.75, 39.30, 37.10),
    "suruc":      (38.30, 36.85, 38.60, 37.05),
    "ceylanpinar": (39.70, 36.70, 40.30, 36.95),
    "mardin_kiziltepe": (40.40, 36.95, 41.00, 37.20),
    "diyarbakir": (40.00, 37.75, 40.60, 38.05),
    "siverek_viransehir": (39.20, 37.35, 39.90, 37.75),
    "adiyaman_kahta": (38.30, 37.65, 38.75, 37.85),
    "pazarcik_maras": (36.85, 37.30, 37.20, 37.60),
    "amik_hatay": (36.25, 36.20, 36.55, 36.45),
    "cukurova_adana": (35.30, 36.80, 35.80, 37.05),
}
# 2018 and 2021 are held out. Collection 1 returned nothing here for 2017
# and 2022, so the training chips actually come from 2019, 2020, 2023-2025.
YEARS = [2017, 2019, 2020, 2022, 2023, 2024, 2025]
MONTHS = (5, 9)                 # May to September, the season of both events
PAIR_GAP_DAYS = (4, 11)         # both event pairs are 5 and 10 days apart
