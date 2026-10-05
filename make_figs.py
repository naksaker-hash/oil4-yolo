"""Manuscript figures. Uses whatever seeds have finished: seed 0 drawn bold,
further seeds thin.

    .venv/bin/python make_figs.py
"""
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import DATA, ROOT

OUT = ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True)
# fixed categorical order, entity bound (dataviz default palette, light mode)
COL = {"zscore": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a"}
LS = {"zscore": "-", "v1": "--", "v2": "-"}
NAME = {"zscore": "z score detector", "v1": "YOLO, generator v1", "v2": "YOLO, generator v2"}
EVENTS = [("narli", "Narlı 2018"), ("siverek", "Siverek 2021")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#888", "axes.linewidth": 0.6,
                     "xtick.color": "#555", "ytick.color": "#555", "axes.labelcolor": "#222",
                     "font.family": "DejaVu Sans"})
spec = importlib.util.spec_from_file_location("ev", ROOT / "06_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
import realio  # noqa: E402

ring = [pe.withStroke(linewidth=3, foreground="black")]


def runs(profile):
    reg = "gen_v2" if profile == "v2" else "gen"
    return sorted(p for p in (ROOT / "runs").glob(f"{reg}_yolo26n-seg_s*")
                  if (p / "eval_summary.json").exists())


def rows(path):
    with open(path) as f:
        return [{k: (float(v) if k in ("thr", "fa_per_km2") else v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def zoom(a, mask, r):
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    a.set_xlim(cx - r, cx + r)
    a.set_ylim(cy + r, cy - r)


def letter(a, s):
    a.text(0.02, 0.98, s, transform=a.transAxes, va="top", ha="left", fontsize=10,
           fontweight="bold", color="white", path_effects=ring)


def fig1():
    """Real event chips and synthetic v2 training chips with their labels."""
    root = DATA / "yolo_gen_v2"
    real = [root / "images/test_narli/narli_event.png", root / "images/test_siverek/siverek_event.png"]
    syn = []
    for p in sorted((root / "images/train").glob("*.png")):
        lab = (root / "labels/train" / (p.stem + ".txt")).read_text().strip()
        if lab and len(syn) < 6 and p.stem.endswith("_0"):
            syn.append(p)
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for i, (a, p) in enumerate(zip(ax.ravel(), real + syn)):
        a.imshow(Image.open(p), interpolation="nearest")
        lab = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        for line in lab.read_text().splitlines():
            xy = np.array(line.split()[1:], float).reshape(-1, 2) * 256 - 0.5
            a.plot(*np.vstack([xy, xy[:1]]).T, color="white", lw=0.9, path_effects=ring)
        a.set_xticks([]); a.set_yticks([])
        letter(a, "abcdefgh"[i])
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig1_synthetic.pdf"); fig.savefig(OUT / "fig1_synthetic.png", dpi=200)


def fig2():
    """Detections at the fixed operating points, seed 0."""
    from ultralytics import YOLO
    run = ROOT / "runs" / "gen_v2_yolo26n-seg_s0"
    s = json.load(open(run / "eval_summary.json"))
    model = YOLO(str(run / "weights" / "best.pt"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for i, (a, (event, title)) in enumerate(zip(ax, EVENTS)):
        p = DATA / "real" / f"{event}_event.npz"
        label = realio.load(p, coreg=False)["mask"]
        img_dir = DATA / "yolo_gen_v2" / "images" / f"test_{event}"
        a.imshow(Image.open(img_dir / f"{event}_event.png"), interpolation="nearest")
        a.contour(label.astype(float), [0.5], colors="white", linewidths=2.2)
        for _, m in ev.ZScore(True).at(p, 4.0):
            c = a.contour(m.astype(float), [0.5], colors=COL["zscore"], linewidths=1.3)
            c.set(path_effects=ring)
        for _, m in ev.Yolo(model, img_dir, 384).at(p, s["yolo_val_threshold"]):
            c = a.contour(m.astype(float), [0.5], colors=COL["v2"], linewidths=1.3)
            c.set(path_effects=ring)
        zoom(a, label, 45 if event == "siverek" else 70)
        a.set_xticks([]); a.set_yticks([])
        letter(a, "ab"[i])
        a.set_title(title, fontsize=9)
        # 200 m scale bar (20 px)
        x0, x1 = a.get_xlim(); y1 = a.get_ylim()[0]
        a.plot([x0 + 6, x0 + 26], [y1 - 6, y1 - 6], color="white", lw=2.5, path_effects=ring)
        a.text(x0 + 16, y1 - 9, "200 m", color="white", ha="center", fontsize=7, path_effects=ring)
    h = [plt.Line2D([], [], color="white", lw=2.2, path_effects=ring, label="Reference outline"),
         plt.Line2D([], [], color=COL["zscore"], lw=1.3, path_effects=ring, label=NAME["zscore"] + ", z = 4"),
         plt.Line2D([], [], color=COL["v2"], lw=1.3, path_effects=ring,
                    label=NAME["v2"] + f", score {s['yolo_val_threshold']}")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.3)
    fig.savefig(OUT / "fig2_detections.pdf"); fig.savefig(OUT / "fig2_detections.png", dpi=200)


def hit_band(a, rs, y, color, lw=3):
    xs = [r["thr"] for r in rs if r["hit"] == "True"]
    if xs:
        a.hlines(y, min(xs), max(xs), color=color, lw=lw, capstyle="butt")


def fig3():
    """False alarm rate against threshold, with the threshold range that hits."""
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey="row")
    for r_i, (event, title) in enumerate(EVENTS):
        # z score
        a = ax[r_i, 0]
        base = ROOT / "runs" / "gen_v2_yolo26n-seg_s0"
        rz = rows(base / f"eval_{event}_zscore_coreg.csv")
        a.plot([r["thr"] for r in rz], [r["fa_per_km2"] for r in rz], color=COL["zscore"], lw=2,
               ls=LS["zscore"], label=NAME["zscore"])
        a.axvline(4, color="#555", lw=0.8, ls=":")
        hit_band(a, rz, -0.05, COL["zscore"])
        a.set_xlabel("z threshold (patches with z at or below minus the threshold)")
        # YOLO
        b = ax[r_i, 1]
        for prof in ["v1", "v2"]:
            for k, run in enumerate(runs(prof)):
                ry = rows(run / f"eval_{event}_yolo_coreg.csv")
                b.plot([r["thr"] for r in ry], [r["fa_per_km2"] for r in ry], color=COL[prof],
                       lw=2 if k == 0 else 0.9, ls=LS[prof], alpha=1 if k == 0 else 0.6,
                       label=NAME[prof] if k == 0 else None)
                # one bar per run, v1 seeds then v2 seeds, top to bottom
                hit_band(b, ry, -0.025 - 0.013 * (3 * ["v1", "v2"].index(prof) + k), COL[prof],
                         lw=2.5 if k == 0 else 1.5)
                t = json.load(open(run / "eval_summary.json"))["yolo_val_threshold"]
                b.axvline(t, color=COL[prof], lw=0.8, ls=":")
        b.set_xlabel("YOLO score threshold")
        for x in (a, b):
            x.set_yscale("symlog", linthresh=0.1)
            x.set_ylim(-0.1, 30)
            x.set_yticks([0, 0.1, 1, 10])
            x.set_yticklabels(["0", "0.1", "1", "10"])
            x.grid(axis="y", color="#e5e5e5", lw=0.5)
            x.spines[["top", "right"]].set_visible(False)
        a.set_ylabel(f"{title}\nfalse alarms per km²")
        letter(a, "ac"[r_i]); letter(b, "bd"[r_i])
    h, l = ax[0, 0].get_legend_handles_labels()
    h2, l2 = ax[0, 1].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig3_thresholds.pdf"); fig.savefig(OUT / "fig3_thresholds.png", dpi=200)


def _lines(gj):
    d = json.load(open(gj))
    feats = d.get("features", [d])
    out = []
    for f in feats:
        g = f.get("geometry", f)
        geoms = g["geometries"] if g["type"] == "GeometryCollection" else [g]
        for gg in geoms:
            t, c = gg["type"], gg["coordinates"]
            if t == "LineString":
                out.append(np.array(c))
            elif t == "MultiLineString":
                out += [np.array(x) for x in c]
            elif t == "Polygon":
                out += [np.array(x) for x in c]
            elif t == "MultiPolygon":
                out += [np.array(r) for p in c for r in p]
    return out


def fig_map():
    """Study area: plains (training and held out validation), chip
    centres, the two spills and their pipelines."""
    from config import EVENTS, PLAINS
    val_plains = ("ceylanpinar", "amik_hatay")
    pts = {"train": [], "val": []}
    for f in sorted((DATA / "clean").glob("*.npz")):
        m = json.loads(str(np.load(f)["meta"]))
        pl = next((n for n, (a, b, c, d) in PLAINS.items() if a <= m["lon"] <= c and b <= m["lat"] <= d), None)
        pts["val" if pl in val_plains else "train"].append((m["lon"], m["lat"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    # national outline, not redistributed here; the map is drawn without it if absent
    if (ROOT / "assets" / "turkey.geojson").exists():
        for ln in _lines(ROOT / "assets" / "turkey.geojson"):
            ax.plot(ln[:, 0], ln[:, 1], color="#9a9a9a", lw=0.6)
    for name, f in [("Kirkuk–Ceyhan pipeline", "pipeline_kirkuk_ceyhan"), ("Batman–Dörtyol pipeline", "pipeline_batman_dortyol")]:
        for k, ln in enumerate(_lines(ROOT / "assets" / f"{f}.geojson")):
            ax.plot(ln[:, 0], ln[:, 1], color="#555", lw=1.0, ls="--" if "Batman" in name else "-",
                    label=name if k == 0 else None)
    for n, (a, b, c, d) in PLAINS.items():
        col = COL["v1"] if n in val_plains else "#777"
        ax.add_patch(plt.Rectangle((a, b), c - a, d - b, fill=False, ec=col, lw=1.0))
    for k, c in [("train", "#777"), ("val", COL["v1"])]:
        p = np.array(pts[k])
        ax.scatter(p[:, 0], p[:, 1], s=2, color=c, lw=0,
                   label=f"{'Training' if k == 'train' else 'Validation'} chips ({len(p):,})")
    for e, lab in [("narli", "Narlı 2018"), ("siverek", "Siverek 2021")]:
        ev_ = EVENTS[e]
        ax.scatter(ev_["lon"], ev_["lat"], marker="*", s=140, color=COL["zscore"], ec="black", lw=0.6, zorder=5)
        ax.text(ev_["lon"] + 0.08, ev_["lat"] + 0.08, lab, fontsize=8, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_xlim(35.0, 41.5); ax.set_ylim(35.9, 38.4)
    ax.set_aspect(1 / np.cos(np.radians(37.2)))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9, markerscale=4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_map.pdf"); fig.savefig(OUT / "fig_map.png", dpi=200)


def fig_workflow():
    """Processing chain from imagery to evaluation."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", fc="#f4f4f2", ec="#888", lw=0.7)
    hi = dict(boxstyle="round,pad=0.4", fc="#e3f3ec", ec=COL["v2"], lw=1.0)
    real = dict(boxstyle="round,pad=0.4", fc="#e6eefa", ec=COL["zscore"], lw=1.0)
    nodes = {
        "clean": (11, 27, "1,370 clean Sentinel-2\npairs, 10 plains\n(2019 to 2025)", box),
        "gen": (33, 27, "Plume generator\nmultiplicative attenuation\nv1 or v2", hi),
        "feat": (55, 27, "Change channels\nΔNBR, Δlog BAI,\nΔlog mean reflectance", box),
        "train": (77, 27, "YOLO26n-seg\n3,162 train, 948 val\nthreshold from val F1", hi),
        "real": (11, 8, "2 spill pairs and\n22 null pairs\n(coregistered)", real),
        "zs": (44, 8, "Outlier detector\nz ≤ −4 on ΔNBR", real),
        "eval": (77, 8, "Scene level evaluation\nhit, IoU, rank,\nfalse alarms per km²", box),
    }
    for k, (x, y, t, st) in nodes.items():
        ax.text(x, y, t, ha="center", va="center", fontsize=7, bbox=st)
    hw = {"clean": 8.2, "gen": 9.2, "feat": 8.2, "train": 7.8, "real": 6.0, "zs": 6.0, "eval": 8.2}
    hh = 5.5
    arr = dict(arrowstyle="-|>", color="#555", lw=0.8, shrinkA=0, shrinkB=0)
    for a_, b_ in [("clean", "gen"), ("gen", "feat"), ("feat", "train"), ("real", "zs"), ("zs", "eval")]:
        (xa, ya), (xb, yb) = nodes[a_][:2], nodes[b_][:2]
        ax.annotate("", xy=(xb - hw[b_] - 0.4, yb), xytext=(xa + hw[a_] + 0.4, ya), arrowprops=arr)
    ax.annotate("", xy=(77, 8 + hh + 1), xytext=(77, 27 - hh - 1), arrowprops=arr)
    ax.annotate("", xy=(55, 27 - hh - 1), xytext=(11, 8 + hh + 1),
                arrowprops=dict(arrowstyle="-|>", color=COL["zscore"], lw=0.8, ls="--",
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(26, 17.5, "same channels", fontsize=6.5, color=COL["zscore"], rotation=14)
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig_workflow.pdf"); fig.savefig(OUT / "fig_workflow.png", dpi=200)


def fig_training():
    """Synthetic validation mask mAP50-95 and training segmentation loss."""
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for prof in ["v1", "v2"]:
        for k, run in enumerate(sorted(p for p in (ROOT / "runs").glob(("gen_v2" if prof == "v2" else "gen") + "_yolo26n-seg_s*")
                                       if (p / "results.csv").exists())):
            r = list(csv.DictReader(open(run / "results.csv")))
            ep = [int(x["epoch"]) for x in r]
            kw = dict(color=COL[prof], ls=LS[prof], lw=1.6 if k == 0 else 0.8, alpha=1 if k == 0 else 0.6,
                      label=f"YOLO26n, generator {prof}" if k == 0 else None)
            ax[0].plot(ep, [float(x["train/seg_loss"]) for x in r], **kw)
            ax[1].plot(ep, [float(x["metrics/mAP50-95(M)"]) for x in r], **kw)
    ax[0].set_ylabel("Training segmentation loss"); ax[1].set_ylabel("Validation mask mAP50–95")
    for i, a in enumerate(ax):
        a.set_xlabel("Epoch"); a.grid(color="#e5e5e5", lw=0.5); a.spines[["top", "right"]].set_visible(False)
        a.text(-0.14, 1.02, "ab"[i], transform=a.transAxes, va="bottom", fontweight="bold", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_training.pdf"); fig.savefig(OUT / "fig_training.png", dpi=200)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig_map", "fig_workflow", "fig_training"]
    for f in todo:
        globals()[f]()
    print("figures in", OUT)
