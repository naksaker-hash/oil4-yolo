"""Supplementary analyses for one trained run, the ones recent YOLO papers
report as standard, written to runs/<run>/analysis.json:

  size     recall on the synthetic validation plains by plume size
           (< 50, 50 to 150, >= 150 px) at the run's fixed threshold, and
           precision, recall and F1 at that threshold
  imgsz    detection of the two real spills when inference is run at 256,
           320, 384, 512 and 640 px (training used 384)
  occlude  score of the real plume patch when each input channel is set
           to 114 (no change), a quantitative stand in for a saliency map
  val      Ultralytics box and mask P, R, mAP50 and mAP50-95 on the
           synthetic validation plains
  cost     parameters, GFLOPs at 384 px, and mean CPU and MPS latency per
           chip over 50 runs after warm up

    .venv/bin/python 07_analysis.py --run gen_v2_yolo26n-seg_s0 --regime gen_v2
"""
import argparse
import importlib.util
import json
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from config import DATA, ROOT

spec = importlib.util.spec_from_file_location("ev", ROOT / "06_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
import realio  # noqa: E402

CH = ["dNBR", "dlogBAI", "dlogA"]


def size_bins(det, root, t):
    bins = {"lt50": [0, 0], "50to150": [0, 0], "ge150": [0, 0]}
    tp = fp = fn = 0
    for mp in sorted((root / "masks" / "val").glob("*.png")):
        lab, n = ndi.label(np.asarray(Image.open(mp)) > 0)
        truth = [lab == i for i in range(1, n + 1)]
        pts = det.at_png(root / "images" / "val" / mp.name, t)
        matched = set()
        for _, p in pts:
            hit = [i for i, g in enumerate(truth)
                   if (p & g).sum() / g.sum() >= 0.5 or (p & g).sum() / (p | g).sum() >= 0.25]
            matched.update(hit)
            fp += not hit
        for i, g in enumerate(truth):
            a = g.sum()
            k = "lt50" if a < 50 else "50to150" if a < 150 else "ge150"
            bins[k][0] += i in matched
            bins[k][1] += 1
        tp += len(matched)
        fn += len(truth) - len(matched)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"recall_by_size": {k: {"hit": h, "n": n, "recall": round(h / n, 4)} for k, (h, n) in bins.items()},
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def real_hit(model, png, label, t, imgsz):
    pts = ev.Yolo(model, png.parent, imgsz).at_png(png, t)
    return ev.judge(pts, label)


def occlusion(model, png, label, imgsz, tmp):
    img = np.asarray(Image.open(png)).copy()
    out = {}
    for i, c in enumerate(["none"] + CH):
        x = img.copy()
        if i:
            x[..., i - 1] = 114
        f = tmp / f"occ_{c}.png"
        Image.fromarray(x).save(f)
        det = ev.Yolo(model, tmp, imgsz)
        inst = det.instances(f)
        best = max([s for s, m in inst if (m & label).sum() / label.sum() >= 0.5
                    or (m & label).sum() / (m | label).sum() >= 0.25] or [0.0])
        out[c] = round(float(best), 4)
    return out


def cost(model, png, imgsz):
    import torch
    # fused inference graph, as Ultralytics and most papers report it (the
    # YOLO26 one to many training head is dropped)
    from copy import deepcopy
    fused = deepcopy(model.model).fuse()
    info = {"params_training_graph": int(sum(p.numel() for p in model.model.parameters()))}
    try:
        from ultralytics.utils.torch_utils import get_flops, model_info
        n_l, n_p, n_g, fl = model_info(fused, verbose=True, imgsz=imgsz)
        info["params"] = int(n_p)
        info["gflops_at_imgsz"] = round(float(fl), 2)
    except Exception as e:  # noqa: BLE001
        info["gflops_at_imgsz"] = f"n/a ({type(e).__name__})"
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for _ in range(5):
            model.predict(str(png), imgsz=imgsz, device=dev, verbose=False)
        t0 = time.perf_counter()
        for _ in range(50):
            model.predict(str(png), imgsz=imgsz, device=dev, verbose=False)
        info[f"ms_per_chip_{dev}"] = round((time.perf_counter() - t0) / 50 * 1000, 1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--skip-cost", action="store_true", help="latency is unreliable while training runs")
    args = ap.parse_args()
    out = ROOT / "runs" / args.run
    root = DATA / f"yolo_{args.regime}"
    from ultralytics import YOLO
    model = YOLO(str(out / "weights" / "best.pt"))
    t = json.load(open(out / "eval_summary.json"))["yolo_val_threshold"]
    res = {"run": args.run, "threshold": t}
    m = model.val(data=str(root / "data_test_narli.yaml"), split="val", imgsz=args.imgsz,
                  batch=8, device="cpu", plots=False, verbose=False)
    res["val"] = {k: round(float(v), 4) for k, v in m.results_dict.items()}
    print("val", res["val"])
    res["size"] = size_bins(ev.Yolo(model, root / "images" / "val", args.imgsz), root, t)
    print("size", res["size"])
    res["imgsz"], res["occlude"] = {}, {}
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    for event in ["narli", "siverek"]:
        png = root / "images" / f"test_{event}" / f"{event}_event.png"
        label = realio.load(DATA / "real" / f"{event}_event.npz", coreg=False)["mask"]
        res["imgsz"][event] = {s: real_hit(model, png, label, t, s) for s in [256, 320, 384, 512, 640]}
        res["occlude"][event] = occlusion(model, png, label, args.imgsz, tmp)
        print(event, "imgsz", {s: (v["hit"], v["iou"]) for s, v in res["imgsz"][event].items()})
        print(event, "occlusion", res["occlude"][event])
    if args.skip_cost:
        res["cost"] = {"params_training_graph": int(sum(p.numel() for p in model.model.parameters()))}
    else:
        res["cost"] = cost(model, root / "images" / "test_narli" / "narli_event.png", args.imgsz)
    print("cost", res["cost"])
    json.dump(res, open(out / "analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
