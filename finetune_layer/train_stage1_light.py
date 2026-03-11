from __future__ import annotations

import argparse
import random
import shlex
import subprocess
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
            nn.Conv2d(96, 32, 3, padding=1),
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


def _run_external_backend(cfg: dict, manifest_csv: Path, out_dir: Path) -> dict:
    import subprocess
    import sys

    # 第一步：准备 external dataset
    finetune_cfg_path = str(cfg.get("_finetune_config_path", ""))
    if not finetune_cfg_path:
        raise ValueError("missing _finetune_config_path for external backend")

    prep_cmd = [
        sys.executable,
        "-m",
        "finetune_layer.prepare_stage1_external_dataset",
        "--config",
        finetune_cfg_path,
    ]
    print("[RUN external dataset prepare]")
    print(" ".join(prep_cmd))
    prep_res = subprocess.run(prep_cmd, text=True)
    if prep_res.returncode != 0:
        raise RuntimeError("prepare_stage1_external_dataset failed")

    external_dataset_root = Path(cfg["output_dir"]) / "external_stage1_dataset"
    external_summary = external_dataset_root / "external_dataset_summary.json"
    if not external_summary.exists():
        raise FileNotFoundError(f"external dataset summary not found: {external_summary}")

    cmd_tmpl = cfg.get("train_command_template")
    expected_ckpt = cfg.get("expected_ckpt")

    if not cmd_tmpl:
        raise ValueError("trainer_backend=external 时，必须提供 train_command_template")

    # 这里提供几个统一占位符给外部训练脚本
    cmd_str = str(cmd_tmpl).format(
        config=str(cfg.get("base_config", "")),
        finetune_config=finetune_cfg_path,
        manifest_csv=str(manifest_csv),
        dataset_dir=str(external_dataset_root),
        output_dir=str(out_dir),
        epochs=str(cfg.get("epochs", 5)),
        batch_size=str(cfg.get("batch_size", 2)),
        lr=str(cfg.get("lr", 1e-4)),
    )

    print("[RUN external trainer]")
    print(cmd_str)

    res = subprocess.run(["bash", "-lc", cmd_str], text=True)
    if res.returncode != 0:
        raise RuntimeError("external finetune command failed")

    ckpt = None
    if expected_ckpt and Path(expected_ckpt).exists():
        ckpt = str(Path(expected_ckpt))
    else:
        candidates = (
            sorted(out_dir.rglob("*.pt"))
            + sorted(out_dir.rglob("*.pth"))
            + sorted(out_dir.rglob("*.ckpt"))
            + sorted(external_dataset_root.rglob("*.pt"))
            + sorted(external_dataset_root.rglob("*.pth"))
            + sorted(external_dataset_root.rglob("*.ckpt"))
        )
        if candidates:
            ckpt = str(candidates[0])

    if not ckpt:
        raise FileNotFoundError("external backend finished but no checkpoint found")

    return {
        "status": "trained",
        "trainer_backend": "external",
        "dataset_root": str(external_dataset_root),
        "external_dataset_summary_json": str(external_summary),
        "best_ckpt": ckpt,
        "ckpt_compatible_with_stage1": True,
        "note": "external backend 已调用真实 stage1 训练脚本，并产出兼容前四层 stage1 的权重。",
    }



def _run_tiny_unet_backend(cfg: dict, manifest_csv: Path, out_dir: Path) -> dict:
    set_seed(int(cfg.get("seed", 42)))

    raw_manifest = pd.read_csv(manifest_csv)
    usable = raw_manifest[raw_manifest["has_mask"] == True].copy()  # noqa

    if len(usable) == 0:
        return {
            "status": "skipped_no_masks",
            "reason": "pseudo_dataset 中没有可用 mask_sem.tif，训练已跳过。",
            "manifest_csv": str(manifest_csv),
            "num_total_samples": int(len(raw_manifest)),
            "num_usable_samples": 0,
            "trainer_backend": "tiny_unet",
            "ckpt_compatible_with_stage1": False,
        }

    ds = RoiMaskDataset(str(manifest_csv), patch_size=int(cfg.get("train_patch_size", 256)))

    val_ratio = float(cfg.get("val_ratio", 0.2))
    if len(ds) == 1:
        train_ds = ds
        val_ds = ds
        train_len = 1
        val_len = 1
    else:
        val_len = max(1, int(len(ds) * val_ratio))
        train_len = max(1, len(ds) - val_len)
        if train_len + val_len > len(ds):
            val_len = len(ds) - train_len
        train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet().to(device)

    pos_weight = torch.tensor([float(cfg.get("pos_weight", 2.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

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

    return {
        "status": "trained",
        "trainer_backend": "tiny_unet",
        "num_total_samples": int(len(raw_manifest)),
        "num_usable_samples": int(len(ds)),
        "train_len": train_len,
        "val_len": val_len,
        "best_val_loss": best_val,
        "best_ckpt": str(best_ckpt),
        "history": history,
        "ckpt_compatible_with_stage1": False,
        "note": "tiny_unet 是仓库内轻量兜底训练器，不与外部 stage1 模型结构强绑定。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["_finetune_config_path"] = args.config

    out_dir = Path(cfg["output_dir"]) / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = Path(cfg["output_dir"]) / "pseudo_dataset" / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest.csv 不存在: {manifest_csv}")

    backend = str(cfg.get("trainer_backend", "tiny_unet")).strip().lower()

    if backend == "external":
        summary = _run_external_backend(cfg, manifest_csv, out_dir)
    elif backend == "tiny_unet":
        summary = _run_tiny_unet_backend(cfg, manifest_csv, out_dir)
    else:
        raise ValueError(f"未知 trainer_backend: {backend}")

    dump_json(summary, out_dir / "train_summary.json")
    print(f"[OK] training stage done: backend={backend}")


if __name__ == "__main__":
    main()