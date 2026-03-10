from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from finetune_layer.io_utils import dump_json, load_yaml


class DummySegDataset(Dataset):
    """
    第一版最小骨架：
    先不依赖你完整的 stage1 训练器，保证流程可跑。
    后续你可把这里换成真实的 TIFF 读取与 mask 监督。
    """

    def __init__(self, meta_dir: str):
        self.files = sorted(Path(meta_dir).glob("*.json"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # 这里先生成占位数据，保证训练流程通
        # 后续换成真实 image / mask 读取
        x = torch.randn(3, 256, 256).float()
        y = (torch.rand(1, 256, 256) > 0.6).float()
        return x, y


class TinyHeadOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_backbone(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False


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

    meta_dir = Path(cfg["output_dir"]) / "pseudo_dataset" / "meta"
    ds = DummySegDataset(str(meta_dir))
    if len(ds) == 0:
        raise RuntimeError("pseudo_dataset/meta 下没有样本，无法训练。")

    val_ratio = float(cfg.get("val_ratio", 0.2))
    val_len = max(1, int(len(ds) * val_ratio))
    train_len = max(1, len(ds) - val_len)
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyHeadOnlyModel().to(device)

    if bool(cfg.get("freeze_backbone", True)):
        freeze_backbone(model)

    pos_weight = torch.tensor([float(cfg.get("pos_weight", 2.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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

    summary = {
        "train_mode": cfg.get("train_mode", "head_only"),
        "num_samples": len(ds),
        "train_len": train_len,
        "val_len": val_len,
        "best_val_loss": best_val,
        "best_ckpt": str(best_ckpt),
        "history": history,
        "note": "当前为最小可运行骨架。后续可替换为真实 stage1 训练器与真实 TIFF 监督。",
    }
    dump_json(summary, out_dir / "train_summary.json")
    print(f"[OK] training done: {best_ckpt}")


if __name__ == "__main__":
    main()