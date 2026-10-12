"""U-Net baseline on the same synthetic data and the same three change
channels as the YOLO networks (round 3 review: a supervised segmentation
network outside the YOLO family). Training masks are rasterised from the
YOLO polygons; validation uses the stored pixel masks of the held out
plains. About 1.9 million parameters, close to YOLO26n-seg.

    .venv/bin/python 24_unet.py --regime gen_v2 --seed 0 [--size 384]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from config import DATA, ROOT

SIZE = 256


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([block(3, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([block(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(4)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


def to_tensor(img, size=SIZE):
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    if size != x.shape[-1]:
        x = F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    return x


def poly_mask(txt):
    m = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(m)
    for line in Path(txt).read_text().splitlines():
        v = [float(x) for x in line.split()[1:]]
        if len(v) >= 6:
            d.polygon([(v[i] * SIZE, v[i + 1] * SIZE) for i in range(0, len(v), 2)], fill=1)
    return np.asarray(m, bool)


class Data(torch.utils.data.Dataset):
    def __init__(self, root, split, aug, size=SIZE):
        self.imgs = sorted((root / "images" / split).glob("*.png"))
        self.root, self.split, self.aug, self.size = root, split, aug, size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = np.asarray(Image.open(p))
        if self.split == "train":
            m = poly_mask(self.root / "labels" / "train" / (p.stem + ".txt"))
        else:
            m = np.asarray(Image.open(self.root / "masks" / "val" / p.name)) > 0
        if self.aug:
            if np.random.rand() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if np.random.rand() < 0.5:
                img, m = img[::-1], m[::-1]
            k = np.random.randint(4)
            img, m = np.rot90(img, k), np.rot90(m, k)
        y = torch.from_numpy(np.ascontiguousarray(m)).float()[None]
        if self.size != SIZE:
            y = F.interpolate(y[None], size=(self.size, self.size), mode="nearest")[0]
        return to_tensor(img, self.size), y


def loss_fn(logit, y):
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(5.0, device=y.device)) + dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="gen_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=SIZE, help="input size; 384 matches the YOLO networks")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    root = DATA / f"yolo_{args.regime}"
    out = ROOT / "runs" / f"{args.regime}_unet{'' if args.size == SIZE else args.size}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    tr = torch.utils.data.DataLoader(Data(root, "train", True, args.size), batch_size=args.batch, shuffle=True, num_workers=2)
    va = torch.utils.data.DataLoader(Data(root, "val", False, args.size), batch_size=2 * args.batch, num_workers=2)
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, log = -1, []
    for ep in range(args.epochs):
        net.train()
        tl = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(x)
        sched.step()
        net.eval()
        inter = union = 0
        with torch.no_grad():
            for x, y in va:
                p = torch.sigmoid(net(x.to(dev))) > 0.5
                y = y.to(dev) > 0.5
                inter += (p & y).sum().item()
                union += (p | y).sum().item()
        iou = inter / max(union, 1)
        log.append({"epoch": ep + 1, "train_loss": tl / len(tr.dataset), "val_iou": iou})
        print(log[-1], flush=True)
        if iou > best:
            best = iou
            torch.save(net.state_dict(), out / "best.pt")
    json.dump({"log": log, "best_val_iou": best, "size": args.size, "batch": args.batch,
               "params": sum(p.numel() for p in net.parameters())}, open(out / "train_log.json", "w"), indent=1)
    (out / "DONE").write_text("finished\n")


if __name__ == "__main__":
    main()
