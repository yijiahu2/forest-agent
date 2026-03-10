from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class PseudoCandidate:
    xbh: str
    split: str                  # pseudo / hard / replay_good
    tree_count_error_ratio: float
    mean_crown_width_error_ratio: float
    closure_error_abs: float
    density_error_abs: float | None = None
    mean_slope: float | None = None
    relief_elev: float | None = None
    dominant_aspect_class: str | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoiSample:
    roi_id: str
    xbh: str
    split: str
    image_path: str
    mask_sem_path: str
    dem_path: str | None = None
    slope_path: str | None = None
    aspect_path: str | None = None
    meta_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    freeze_backbone: bool
    pos_weight: float
    val_ratio: float
    seed: int
    train_mode: str = "head_only"


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p