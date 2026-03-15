from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def assert_exists(path: str | Path, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在: {p}")
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取 CSV 失败: {path}\n最后错误: {last_err}")


def dump_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = ensure_parent(path)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def run_cmd(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True)


# ==============================
# details.csv 字段标准化
# ==============================

COLUMN_CANDIDATES = {
    "XBH": [
        "XBH", "xbh", "xiaoban_id", "plot_id", "patch_id", "compartment_id"
    ],
    "tree_count_error_ratio": [
        "tree_count_error_ratio",
        "tree_error_ratio",
        "count_error_ratio",
        "tree_cnt_error_ratio",
        "count_rel_error",
        "tree_count_rel_error",
    ],
    "mean_crown_width_error_ratio": [
        "mean_crown_width_error_ratio",
        "crown_error_ratio",
        "crown_width_error_ratio",
        "mean_crown_error_ratio",
        "crown_rel_error",
        "pjgf_error_ratio",
    ],
    "closure_error_abs": [
        "closure_error_abs",
        "closure_abs_error",
        "canopy_closure_error_abs",
        "ybd_error_abs",
    ],
    "density_error_abs": [
        "density_error_abs",
        "density_abs_error",
    ],
    "mean_slope": [
        "mean_slope",
        "slope_mean",
    ],
    "relief_elev": [
        "relief_elev",
        "elev_relief",
        "relief",
    ],
    "dominant_aspect_class": [
        "dominant_aspect_class",
        "aspect_class",
        "main_aspect_class",
    ],
    "landform_type": [
        "landform_type",
        "terrain_landform_type",
    ],
    "slope_class": [
        "slope_class",
        "terrain_slope_class",
    ],
    "aspect_class": [
        "aspect_class",
        "terrain_aspect_class",
    ],
    "slope_position_class": [
        "slope_position_class",
        "terrain_slope_position_class",
    ],
}

PRED_TREE_CANDIDATES = [
    "pred_tree_count", "tree_count_pred", "n_pred", "pred_count", "tree_count_hat",
    "pred_n_trees", "tree_count", "n_instances", "instance_count", "tree_num_pred",
]
PRED_CROWN_CANDIDATES = [
    "pred_mean_crown_width", "mean_crown_width_pred", "crown_pred", "mean_crown_hat",
    "avg_crown_width_pred", "pred_pjgf", "mean_crown_width", "crown_mean_pred",
]
PRED_CLOSURE_CANDIDATES = [
    "pred_closure", "closure_pred", "ybd_pred", "closure_hat",
    "canopy_closure_pred", "pred_ybd",
]
PRED_DENSITY_CANDIDATES = [
    "pred_density", "density_pred", "density_hat", "stand_density_pred",
]


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_rel_err(pred: pd.Series, gt: pd.Series) -> pd.Series:
    gt = pd.to_numeric(gt, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    denom = gt.abs().clip(lower=1e-6)
    return (pred - gt).abs() / denom


def _safe_abs_err(pred: pd.Series, gt: pd.Series) -> pd.Series:
    gt = pd.to_numeric(gt, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    return (pred - gt).abs()


def normalize_details_df(
    raw_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str], list[str], dict[str, Any]]:
    """
    标准化 details.csv。
    与旧版不同：
    - 不再强制 tree/crown/closure 三项都必须存在
    - 优先从已有误差列映射
    - 不存在则尝试从 GT 列 + Pred 列自动计算
    - 最低只要求 XBH + 至少一个误差指标存在
    """
    df = raw_df.copy()
    rename_map: dict[str, str] = {}
    debug_info: dict[str, Any] = {
        "raw_columns": raw_df.columns.tolist(),
        "direct_mapped": {},
        "computed": {},
        "missing_after_normalize": [],
    }

    # 1) 直接同义列映射
    for std_name, cands in COLUMN_CANDIDATES.items():
        found = _find_first_existing_column(df, cands)
        if found is not None and found != std_name:
            rename_map[found] = std_name
        elif found == std_name:
            rename_map[found] = std_name

    df = df.rename(columns=rename_map).copy()
    debug_info["direct_mapped"] = rename_map

    # 2) 保证 XBH
    xbh_field = cfg.get("xiaoban_id_field", "XBH")
    if "XBH" not in df.columns and xbh_field in df.columns:
        df["XBH"] = df[xbh_field]
        debug_info["computed"]["XBH"] = f"copied_from:{xbh_field}"

    # 3) 尝试由 GT + Pred 自动推导误差列
    gt_tree = cfg.get("tree_count_field")
    gt_crown = cfg.get("crown_field")
    gt_closure = cfg.get("closure_field")
    gt_density = cfg.get("density_field")

    pred_tree = _find_first_existing_column(df, PRED_TREE_CANDIDATES)
    pred_crown = _find_first_existing_column(df, PRED_CROWN_CANDIDATES)
    pred_closure = _find_first_existing_column(df, PRED_CLOSURE_CANDIDATES)
    pred_density = _find_first_existing_column(df, PRED_DENSITY_CANDIDATES)

    if "tree_count_error_ratio" not in df.columns and gt_tree in df.columns and pred_tree in df.columns:
        df["tree_count_error_ratio"] = _safe_rel_err(df[pred_tree], df[gt_tree])
        debug_info["computed"]["tree_count_error_ratio"] = f"{pred_tree} vs {gt_tree}"

    if "mean_crown_width_error_ratio" not in df.columns and gt_crown in df.columns and pred_crown in df.columns:
        df["mean_crown_width_error_ratio"] = _safe_rel_err(df[pred_crown], df[gt_crown])
        debug_info["computed"]["mean_crown_width_error_ratio"] = f"{pred_crown} vs {gt_crown}"

    if "closure_error_abs" not in df.columns and gt_closure in df.columns and pred_closure in df.columns:
        df["closure_error_abs"] = _safe_abs_err(df[pred_closure], df[gt_closure])
        debug_info["computed"]["closure_error_abs"] = f"{pred_closure} vs {gt_closure}"

    if "density_error_abs" not in df.columns and gt_density and gt_density in df.columns and pred_density in df.columns:
        df["density_error_abs"] = _safe_abs_err(df[pred_density], df[gt_density])
        debug_info["computed"]["density_error_abs"] = f"{pred_density} vs {gt_density}"

    # 4) 补可选列
    if "density_error_abs" not in df.columns:
        df["density_error_abs"] = np.nan
    if "mean_slope" not in df.columns:
        df["mean_slope"] = 0.0
    if "relief_elev" not in df.columns:
        df["relief_elev"] = 0.0
    if "dominant_aspect_class" not in df.columns:
        df["dominant_aspect_class"] = None
    if "landform_type" not in df.columns:
        df["landform_type"] = None
    if "slope_class" not in df.columns:
        df["slope_class"] = None
    if "aspect_class" not in df.columns:
        df["aspect_class"] = None
    if "slope_position_class" not in df.columns:
        df["slope_position_class"] = None

    # 5) 最低要求：XBH + 至少一个误差列
    error_cols = [
        "tree_count_error_ratio",
        "mean_crown_width_error_ratio",
        "closure_error_abs",
        "density_error_abs",
    ]
    available_error_cols = [c for c in error_cols if c in df.columns and df[c].notna().any()]

    missing = []
    if "XBH" not in df.columns:
        missing.append("XBH")
    if len(available_error_cols) == 0:
        missing.append("at_least_one_error_metric")

    debug_info["available_error_cols"] = available_error_cols
    debug_info["missing_after_normalize"] = missing

    return df, rename_map, missing, debug_info