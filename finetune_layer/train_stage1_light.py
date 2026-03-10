from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from finetune_layer.io_utils import dump_json, load_csv, load_yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RoiMaskDataset(Dataset):
    def __init__(self, manifest_csv: str, patch_size: int = 256):
        df = load_csv(manifest_csv)
        df = df[df["has_mask"] == True].copy()  # noqa
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read()
        if arr.shape[0] >= 3:
            arr = arr[:3]
        elif arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        else:
            pad = 3 - arr.shape[0]
            arr = np.concatenate([arr, np.repeat(arr[-1:], pad, axis=0)], axis=0)

        arr = arr.astype(np.float32)
        p2 = np.percentile(arr, 98) if np.isfinite(arr).any() else 1.0
        if p2 <= 0:
            p2 = 1.0
        arr = np.clip(arr / p2, 0, 1)
        x = torch.from_numpy(arr).float().unsqueeze(0)
        x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def _read_mask(self, path: str) -> torch.Tensor:
        with rasterio.open(path) as src:
            arr = src.read(1)
        arr = (arr > 0).astype(np.float32)
        y = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        y = F.interpolate(y, size=(self.patch_size, self.patch_size), mode="nearest")
        return y.squeeze(0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._read_image(row["image_path"])
        y = self._read_mask(row["mask_sem_path"])
        return x, y


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        z = torch.cat([u, e1], dim=1)
        d = self.dec(z)
        return self.head(d)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"]) / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.get("seed", 42)))

    manifest_csv = Path(cfg["output_dir"]) / "pseudo_dataset" / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest.csv 不存在: {manifest_csv}")

    raw_manifest = pd.read_csv(manifest_csv)
    usable = raw_manifest[raw_manifest["has_mask"] == True].copy()  # noqa

    if len(usable) == 0:
        summary = {
            "status": "skipped_no_masks",
            "reason": "pseudo_dataset 中没有可用 mask_sem.tif，训练已跳过。",
            "manifest_csv": str(manifest_csv),
            "num_total_samples": int(len(raw_manifest)),
            "num_usable_samples": 0,
        }
        dump_json(summary, out_dir / "train_summary.json")
        print("[SKIP] no usable masks, training skipped.")
        return

    ds = RoiMaskDataset(str(manifest_csv), patch_size=int(cfg.get("train_patch_size", 256)))

    val_ratio = float(cfg.get("val_ratio", 0.2))
    val_len = max(1, int(len(ds) * val_ratio))
    train_len = max(1, len(ds) - val_len)
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet().to(device)

    pos_weight = torch.tensor([float(cfg.get("pos_weight", 2.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best_val = float("inf")
    best_ckpt = out_dir / "best_stage1_light.pt"
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": cfg}, best_ckpt)

    summary = {
        "status": "trained",
        "num_total_samples": int(len(raw_manifest)),
        "num_usable_samples": int(len(ds)),
        "train_len": train_len,
        "val_len": val_len,
        "best_val_loss": best_val,
        "best_ckpt": str(best_ckpt),
        "history": history,
        "note": "这是第五层的轻量独立训练器；只有在 stage1 推理脚本支持加载外部 ckpt 后，才能真正回灌到全流程。",
    }
    dump_json(summary, out_dir / "train_summary.json")
    print(f"[OK] training done: {best_ckpt}")


if __name__ == "__main__":
    main()