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

OUT = ROOT / "manuscript" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
# fixed categorical order, entity bound (dataviz default palette, light mode)
COL = {"zscore": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a"}
LS = {"zscore": "-", "v1": "--", "v2": "-"}
NAME = {"zscore": "z score detector", "v1": "YOLO, generator v1", "v2": "YOLO, generator v2"}
EVENTS = [("narli", "Narlı 2018"), ("siverek", "Siverek 2021")]
EVENT_DATES = {"narli": ("15 Jun 2018", "20 Jun 2018"), "siverek": ("5 Aug 2021", "15 Aug 2021")}
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


def truecolour(pre, post):
    """B4, B3, B2 of both dates, one 1 to 99 % stretch shared by all bands and both
    dates (keeps the colour balance), gamma 0.8."""
    rgb = [np.moveaxis(x[[2, 1, 0]], 0, -1) for x in (pre, post)]
    lo, hi = np.nanpercentile(np.concatenate([r.ravel() for r in rgb]), [1, 99])
    return [np.clip((r - lo) / (hi - lo), 0, 1) ** 0.8 for r in rgb]


def outline(a, mask, **kw):
    c = a.contour(mask.astype(float), [0.5], colors="white", linewidths=kw.get("lw", 1.2),
                  linestyles=kw.get("ls", "solid"))
    c.set(path_effects=ring)


def scalebar(a, px=20, label="200 m"):
    x0 = a.get_xlim()[0]; y1 = a.get_ylim()[0]
    a.plot([x0 + 5, x0 + 5 + px], [y1 - 5, y1 - 5], color="white", lw=2.2, path_effects=ring)
    a.text(x0 + 5 + px / 2, y1 - 8, label, color="white", ha="center", fontsize=6.5, path_effects=ring)


def fig1():
    """True colour before and after each event, the network input, and synthetic v2 training pairs."""
    root = DATA / "yolo_gen_v2"
    syn = []
    for p in sorted((root / "images/train").glob("*.png")):
        lab = (root / "labels/train" / (p.stem + ".txt")).read_text().strip()
        if lab and len(syn) < 3 and p.stem.endswith("_0"):
            syn.append(p)
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 7.6))
    letters = iter("abcdefghi")
    for r, (event, title) in enumerate(EVENTS):
        d = realio.load(DATA / "real" / f"{event}_event.npz", coreg=True)
        before, after = truecolour(d["pre"], d["post"])
        net = Image.open(root / "images" / f"test_{event}" / f"{event}_event.png")
        pre_d, post_d = EVENT_DATES[event]
        for c, (img, head) in enumerate([(before, f"Before, {pre_d}"), (after, f"After, {post_d}"),
                                         (net, "Network input")]):
            a = ax[r, c]
            a.imshow(img, interpolation="nearest")
            outline(a, d["mask"], ls="dashed" if c == 0 else "solid")
            zoom(a, d["mask"], 60 if event == "siverek" else 90)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(head, fontsize=8)
            letter(a, next(letters))
            scalebar(a)
        ax[r, 0].set_ylabel(title, fontsize=9)
    for c, p in enumerate(syn):
        a = ax[2, c]
        a.imshow(Image.open(p), interpolation="nearest")
        lab = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        for line in lab.read_text().splitlines():
            xy = np.array(line.split()[1:], float).reshape(-1, 2) * 256 - 0.5
            a.plot(*np.vstack([xy, xy[:1]]).T, color="white", lw=0.9, path_effects=ring)
        a.set_xticks([]); a.set_yticks([])
        a.set_title("Synthetic training pair" if c == 1 else "", fontsize=8)
        letter(a, next(letters))
        scalebar(a, 50, "500 m")
    ax[2, 0].set_ylabel("Generator v2", fontsize=9)
    fig.tight_layout(pad=0.3, h_pad=0.8)
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
    """Location and study area. (a) Türkiye on the globe, (b) the whole country
    with the study area, (c) the study area: plains (training and held out
    validation), chip centres, the two spills and their pipelines."""
    from matplotlib.patches import ConnectionPatch
    from config import EVENTS, PLAINS
    val_plains = ("ceylanpinar", "amik_hatay")
    box_c = (35.0, 41.5, 35.9, 38.4)            # extent of panel c
    pts = {"train": [], "val": []}
    for f in sorted((DATA / "clean").glob("*.npz")):
        m = json.loads(str(np.load(f)["meta"]))
        pl = next((n for n, (a, b, c, d) in PLAINS.items() if a <= m["lon"] <= c and b <= m["lat"] <= d), None)
        pts["val" if pl in val_plains else "train"].append((m["lon"], m["lat"]))
    have = lambda n: (ROOT / "assets" / n).exists()
    pipes = [("Kirkuk–Ceyhan pipeline", "pipeline_kirkuk_ceyhan", "-"),
             ("Batman–Dörtyol pipeline", "pipeline_batman_dortyol", "--")]
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.35], height_ratios=[1, 1.45], wspace=0.04, hspace=0.12)
    # (a) globe, adapted from Wikimedia Commons (CC BY-SA 3.0)
    ga = fig.add_subplot(gs[0, 0])
    if have("globe_turkey.png"):
        ga.imshow(Image.open(ROOT / "assets" / "globe_turkey.png"))
    ga.axis("off")
    ga.text(0.0, 1.0, "a", transform=ga.transAxes, va="top", fontsize=10, fontweight="bold")

    # (b) the whole of Türkiye
    gb = fig.add_subplot(gs[0, 1])
    if have("world_outline.geojson"):
        for ln in _lines(ROOT / "assets" / "world_outline.geojson"):
            gb.plot(ln[:, 0], ln[:, 1], color="#c4c4c4", lw=0.5)
    if have("turkey.geojson"):
        for ln in _lines(ROOT / "assets" / "turkey.geojson"):
            gb.plot(ln[:, 0], ln[:, 1], color="#555", lw=0.8)
    for name, f, ls in pipes:
        for ln in _lines(ROOT / "assets" / f"{f}.geojson"):
            gb.plot(ln[:, 0], ln[:, 1], color="#555", lw=0.8, ls=ls)
    x0, x1, y0, y1 = box_c
    gb.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#c0392b", lw=1.3, zorder=4))
    for e in ("narli", "siverek"):
        gb.scatter(EVENTS[e]["lon"], EVENTS[e]["lat"], marker="*", s=55, color=COL["zscore"],
                   ec="black", lw=0.4, zorder=5)
    for x, y, t, kw in [(33.0, 39.3, "TÜRKİYE", dict(fontsize=9, color="#333", fontweight="bold")),
                        (34.5, 42.6, "Black Sea", dict(fontsize=7, color="#4a7bb7", style="italic")),
                        (29.4, 34.7, "Mediterranean Sea", dict(fontsize=7, color="#4a7bb7", style="italic")),
                        (38.6, 35.0, "SYRIA", dict(fontsize=6.5, color="#888")),
                        (44.7, 35.7, "IRAQ", dict(fontsize=6.5, color="#888")),
                        (45.3, 38.0, "IRAN", dict(fontsize=6.5, color="#888"))]:
        gb.text(x, y, t, ha="center", va="center", path_effects=halo, **kw)
    gb.set_xlim(25.5, 46.5); gb.set_ylim(34.3, 43.1)
    gb.set_aspect(1 / np.cos(np.radians(39)))
    gb.tick_params(labelsize=6.5, length=2)
    gb.set_xticks(range(26, 47, 4)); gb.set_yticks(range(35, 44, 2))
    gb.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    gb.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    gb.yaxis.tick_right()
    gb.text(0.01, 0.98, "b", transform=gb.transAxes, va="top", fontsize=10, fontweight="bold")

    # (c) study area
    ax = fig.add_subplot(gs[1, :])
    if have("turkey.geojson"):
        for ln in _lines(ROOT / "assets" / "turkey.geojson"):
            ax.plot(ln[:, 0], ln[:, 1], color="#9a9a9a", lw=0.7)
    for name, f, ls in pipes:
        for k, ln in enumerate(_lines(ROOT / "assets" / f"{f}.geojson")):
            ax.plot(ln[:, 0], ln[:, 1], color="#555", lw=1.0, ls=ls, label=name if k == 0 else None)
    for n, (a, b, c, d) in PLAINS.items():
        col = COL["v1"] if n in val_plains else "#777"
        ax.add_patch(plt.Rectangle((a, b), c - a, d - b, fill=False, ec=col, lw=1.0))
    for k, c in [("train", "#777"), ("val", COL["v1"])]:
        p = np.array(pts[k])
        ax.scatter(p[:, 0], p[:, 1], s=2, color=c, lw=0,
                   label=f"{'Training' if k == 'train' else 'Validation'} chips ({len(p):,})")
    for e, lab, dx, dy in [("narli", "Narlı 2018", 0.1, -0.17), ("siverek", "Siverek 2021", -0.09, 0.3)]:
        ev_ = EVENTS[e]
        ax.scatter(ev_["lon"], ev_["lat"], marker="*", s=140, color=COL["zscore"], ec="black", lw=0.6, zorder=5,
                   label="Spill" if e == "narli" else None)
        ax.text(ev_["lon"] + dx, ev_["lat"] + dy, lab, fontsize=8, ha="left" if dx > 0 else "right", zorder=6, path_effects=halo)
    ax.text(38.6, 36.2, "SYRIA", fontsize=7, color="#888", ha="center", path_effects=halo)
    ax.text(35.3, 36.1, "Mediterranean\nSea", fontsize=7, color="#4a7bb7", style="italic", ha="center",
            path_effects=halo)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians(37.2)))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    for s_ in ax.spines.values():
        s_.set_edgecolor("#c0392b"); s_.set_linewidth(1.3)
    ax.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.92, markerscale=1)
    for h in ax.get_legend().legend_handles:
        if hasattr(h, "set_sizes") and h.get_label().endswith(")"):
            h.set_sizes([12])
    ax.text(0.005, 0.985, "c", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
    # guide lines from the box in b to panel c
    for xb, yb, xc, yc in [(x0, y0, 0, 1), (x1, y0, 1, 1)]:
        fig.add_artist(ConnectionPatch(xyA=(xb, yb), coordsA=gb.transData, xyB=(xc, yc), coordsB=ax.transAxes,
                                       color="#c0392b", lw=0.7, ls=":"))
    fig.savefig(OUT / "fig_map.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "fig_map.png", dpi=200, bbox_inches="tight", pad_inches=0.04)


def fig_workflow():
    """Processing chain from imagery to evaluation, one column, top to bottom."""
    fig, ax = plt.subplots(figsize=(3.5, 4.7))
    ax.set_xlim(0, 50); ax.set_ylim(0, 100); ax.axis("off")
    box = dict(boxstyle="round,pad=0.45", fc="#f4f4f2", ec="#888", lw=0.7)
    hi = dict(boxstyle="round,pad=0.45", fc="#e3f3ec", ec=COL["v2"], lw=1.0)
    real = dict(boxstyle="round,pad=0.45", fc="#e6eefa", ec=COL["zscore"], lw=1.0)
    spec = {
        "clean": (13, 92, "1,370 clean Sentinel-2\npairs, 10 plains\n(2019 to 2025)", box),
        "gen": (13, 72, "Plume generator\nmultiplicative\nattenuation, v1 or v2", hi),
        "feat": (13, 52, "Change channels\nΔNBR, Δlog BAI,\nΔlog mean reflectance", box),
        "train": (13, 32, "YOLO26n-seg\n3,162 train, 948 val\nthreshold from val F1", hi),
        "real": (38, 72, "2 spill pairs and\n22 null pairs\n(coregistered)", real),
        "zs": (38, 32, "Outlier detector\nz ≤ −4 on ΔNBR", real),
        "eval": (25.5, 8, "Scene level evaluation\nhit, IoU, rank,\nfalse alarms per km²", box),
    }
    t = {k: ax.text(x, y, s, ha="center", va="center", fontsize=6.5, bbox=st, zorder=3)
         for k, (x, y, s, st) in spec.items()}
    fig.canvas.draw()

    def link(a_, b_, pa, pb, color="#555", ls="-", rad=0.0):
        ax.annotate("", xy=pb, xycoords=t[b_], xytext=pa, textcoords=t[a_], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, ls=ls, shrinkA=2, shrinkB=2,
                                    connectionstyle=f"arc3,rad={rad}"))
    down = ((0.5, 0), (0.5, 1))
    for a_, b_ in [("clean", "gen"), ("gen", "feat"), ("feat", "train"), ("real", "zs")]:
        link(a_, b_, *down)
    link("train", "eval", (0.5, 0), (0.25, 1))
    link("zs", "eval", (0.5, 0), (0.75, 1))
    # the real pairs pass through the same change channels as the synthetic ones
    link("real", "feat", (0.5, 0), (1, 0.5), color=COL["zscore"], ls="--", rad=-0.25)
    ax.text(27.5, 62.5, "same\nchannels", fontsize=6, color=COL["zscore"], ha="center", va="center")
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


OP_ROWS = [  # (label, detector key in operating_points.json, group)
    ("z score", "zscore", "base"), ("Reed Xiaoli", "rx", "base"), ("IR-MAD", "irmad", "base"),
    ("Random forest v1", "rf_gen", "base"), ("Random forest v2", "rf_gen_v2", "base"),
    ("YOLO26n v1, seed 0", "yolo_gen_yolo26n-seg_s0", "v1"), ("YOLO26n v1, seed 1", "yolo_gen_yolo26n-seg_s1", "v1"),
    ("YOLO26n v1, seed 2", "yolo_gen_yolo26n-seg_s2", "v1"),
    ("YOLO26n v2, seed 0", "yolo_gen_v2_yolo26n-seg_s0", "v2"), ("YOLO26n v2, seed 1", "yolo_gen_v2_yolo26n-seg_s1", "v2"),
    ("YOLO26n v2, seed 2", "yolo_gen_v2_yolo26n-seg_s2", "v2"), ("YOLOv8n v2", "yolo_gen_v2_yolov8n-seg_s0", "v2"),
    ("YOLO11n v2", "yolo_gen_v2_yolo11n-seg_s0", "v2"),
    ("YOLO26n v2, new dataset", "yolo_gen_v2_d1_yolo26n-seg_s0", "v2"),
    ("YOLO26n v2, native 20 m", "yolo_gen_v2_n20_yolo26n-seg_s0", "v2"),
    ("YOLO26n, Narlı factors", "yolo_narli_yolo26n-seg_s0", "cross"),
    ("YOLO26n, Siverek factors", "yolo_siverek_v2_yolo26n-seg_s0", "cross"),
    ("YOLO26n dr, seed 0", "yolo_gen_dr_yolo26n-seg_s0", "dr"), ("YOLO26n dr, seed 1", "yolo_gen_dr_yolo26n-seg_s1", "dr"),
    ("YOLO26n dr, seed 2", "yolo_gen_dr_yolo26n-seg_s2", "dr"),
]
GCOL = {"base": COL["zscore"], "v1": COL["v1"], "v2": COL["v2"], "cross": "#7a5cc4", "dr": "#b0892a"}


def fig_operating():
    """False alarms on the held out pairs at the highest threshold that
    detects both spills (matched operating point)."""
    op = json.load(open(ROOT / "runs" / "revision" / "operating_points.json"))
    summ = json.load(open(ROOT / "runs" / "revision" / "summary.json"))
    rows = [r for r in OP_ROWS if r[1] in op]
    floor = 2e-4
    fig, a = plt.subplots(figsize=(6.3, 0.22 * len(rows) + 0.9))
    for i, (lab, key, g) in enumerate(rows):
        y = len(rows) - 1 - i
        v = op[key]["both_fa_km2"]
        if v is None:
            a.text(floor * 1.1, y, "never detects both", va="center", fontsize=7, color="#777")
        else:
            a.plot([max(v, floor)], [y], "o", ms=6, color=GCOL[g], mec="white", mew=0.8, zorder=3)
            a.text(60, y, "0" if v == 0 else f"{v:.2g}", va="center", ha="right", fontsize=7, color="#333")
        s = summ[key]
        k = "synthetic_f1_same_rule" if "synthetic_f1_same_rule" in s else "conventional" if "conventional" in s else "synthetic_f1"
        f = s[k]
        if f["narli"]["hit"] and f["siverek"]["hit"]:
            a.plot([max(f["nullval"]["fa_km2"], floor)], [y], "o", ms=8, mfc="none", mec=GCOL[g], mew=0.9, zorder=2)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1])
    a.set_xscale("log"); a.set_xlim(floor * 0.6, 70)
    a.plot([], [], "o", color="#555", ms=5, label="matched point")
    a.plot([], [], "o", mfc="none", mec="#555", ms=7, label="fixed point, if both detected")
    a.legend(loc="lower right", bbox_to_anchor=(0.86, 0.0), frameon=False, fontsize=7)
    a.set_xticks([floor, 1e-3, 1e-2, 1e-1, 1, 10]); a.set_xticklabels(["0", "0.001", "0.01", "0.1", "1", "10"])
    a.set_xlabel("False alarms per km$^2$ on 316 held out pairs at the matched operating point")
    a.grid(axis="x", color="#e5e5e5", lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_operating.pdf"); fig.savefig(OUT / "fig_operating.png", dpi=200)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig_map", "fig_workflow", "fig_training", "fig_operating"]
    for f in todo:
        globals()[f]()
    print("figures in", OUT)
