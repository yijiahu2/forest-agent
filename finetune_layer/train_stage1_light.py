from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from finetune_layer.io_utils import dump_json, load_yaml


# -----------------------------
# tiny fallback backend
# -----------------------------
class DummySegDataset(Dataset):
    def __init__(self, meta_dir: str):
        self.files = sorted(Path(meta_dir).glob("*.json"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
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
    if torch.cuda.is_available():
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


def run_tiny_backend(cfg: dict) -> Path:
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
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": cfg}, best_ckpt)

    summary = {
        "backend": "tiny_unet",
        "num_samples": len(ds),
        "train_len": train_len,
        "val_len": val_len,
        "best_val_loss": best_val,
        "best_ckpt": str(best_ckpt),
        "history": history,
    }
    dump_json(summary, out_dir / "train_summary.json")
    print(f"[OK] training done: {best_ckpt}")
    return best_ckpt


# -----------------------------
# external backend
# -----------------------------
def _ensure_external_dataset(args_config: str, cfg: dict) -> Path:
    output_dir = Path(cfg["output_dir"])
    dataset_dir = output_dir / cfg.get("external_dataset_dirname", "external_stage1_dataset")
    force_rebuild = bool(cfg.get("external_dataset_force_rebuild", False))

    if dataset_dir.exists() and not force_rebuild:
        return dataset_dir

    if dataset_dir.exists() and force_rebuild:
        print(f"[INFO] removing existing external dataset dir: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    cmd = [
        sys.executable,
        "-m",
        "finetune_layer.prepare_stage1_external_dataset",
        "--config",
        args_config,
    ]
    print("[RUN external dataset prepare]")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("prepare_stage1_external_dataset failed")

    if not dataset_dir.exists():
        raise RuntimeError(f"external dataset 目录不存在: {dataset_dir}")

    return dataset_dir


def _write_shell_script(script_path: Path, content: str) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            content.strip(),
            "",
        ]
    )
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)


def _find_best_ckpt(search_root: Path) -> Path | None:
    patterns = ["*.ckpt", "*.pt", "*.pth", "*.bin"]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(search_root.rglob(pat))

    if not candidates:
        return None

    def score(p: Path):
        name = p.name.lower()
        priority = 0
        if "best" in name:
            priority += 100
        if "last" in name:
            priority += 20
        if "epoch" in name:
            priority += 10
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0
        return (priority, mtime)

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def run_external_backend(args_config: str, cfg: dict) -> Path:
    output_dir = Path(cfg["output_dir"])
    training_dir = output_dir / cfg.get("external_train_out_dirname", "training")
    training_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = _ensure_external_dataset(args_config, cfg)
    trainer_output_dir = training_dir / "external_trainer"
    trainer_output_dir.mkdir(parents=True, exist_ok=True)

    template = cfg.get("train_command_template")
    if not template:
        raise RuntimeError("trainer_backend=external 但缺少 train_command_template")

    formatted_cmd = template.format(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        trainer_output_dir=str(trainer_output_dir),
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        epochs=cfg.get("epochs", 5),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 42),
        bsize=cfg.get("bsize", 256),
        weight_decay=cfg.get("weight_decay", 1.0e-4),
    )

    script_path = training_dir / "run_external_trainer.sh"
    _write_shell_script(script_path, formatted_cmd)

    print("[RUN external trainer]")
    print(script_path.read_text(encoding="utf-8"))

    result = subprocess.run(["bash", str(script_path)])

    ckpt = _find_best_ckpt(trainer_output_dir)
    if ckpt is None:
        ckpt = _find_best_ckpt(training_dir)
    if ckpt is None:
        ckpt = _find_best_ckpt(output_dir)

    if result.returncode != 0 and ckpt is None:
        raise RuntimeError("train_stage1_light external backend failed")

    if ckpt is None:
        raise RuntimeError("external trainer 运行结束，但未找到 ckpt 文件")

    if result.returncode != 0:
        print(
            "[warn] external trainer exited with non-zero status, "
            "but a checkpoint was found. Continuing with the recovered ckpt."
        )

    summary = {
        "backend": "external",
        "dataset_dir": str(dataset_dir),
        "trainer_output_dir": str(trainer_output_dir),
        "script_path": str(script_path),
        "trainer_returncode": int(result.returncode),
        "trainer_failed_but_ckpt_recovered": bool(result.returncode != 0),
        "best_ckpt": str(ckpt),
    }
    dump_json(summary, training_dir / "train_summary.json")
    print(f"[OK] external training done: {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    backend = str(cfg.get("trainer_backend", "tiny_unet")).lower()

    if backend == "external":
        run_external_backend(args.config, cfg)
    else:
        run_tiny_backend(cfg)


if __name__ == "__main__":
    main()
