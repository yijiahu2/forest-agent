# geo_layer/terrain_constraints.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


DEFAULT_FLAT_SLOPE_THRESHOLD_DEG = 5.0
DEFAULT_LANDFORM_WINDOW_KM2 = 10.0
DEFAULT_PLAIN_RELIEF_THRESHOLD_M = 30.0
DEFAULT_RIDGE_BUFFER_DROP_M = 15.0


@dataclass
class TerrainRuleConfig:
    landform_window_km2: float = DEFAULT_LANDFORM_WINDOW_KM2
    flat_slope_threshold_deg: float = DEFAULT_FLAT_SLOPE_THRESHOLD_DEG
    plain_relief_threshold_m: float = DEFAULT_PLAIN_RELIEF_THRESHOLD_M
    ridge_buffer_drop_m: float = DEFAULT_RIDGE_BUFFER_DROP_M


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def classify_slope_class(slope_deg: Optional[float]) -> Optional[str]:
    v = safe_float(slope_deg)
    if v is None or not np.isfinite(v):
        return None
    if v <= 5:
        return "I_flat"
    if v <= 15:
        return "II_gentle"
    if v <= 25:
        return "III_inclined"
    if v <= 35:
        return "IV_steep"
    if v <= 45:
        return "V_very_steep"
    return "VI_dangerous"


def classify_slope_class_cn(slope_deg: Optional[float]) -> Optional[str]:
    m = {
        "I_flat": "I级 平坡",
        "II_gentle": "II级 缓坡",
        "III_inclined": "III级 斜坡",
        "IV_steep": "IV级 陡坡",
        "V_very_steep": "V级 急坡",
        "VI_dangerous": "VI级 险坡",
    }
    k = classify_slope_class(slope_deg)
    return m.get(k)


def normalize_aspect_deg(aspect_deg: Optional[float]) -> Optional[float]:
    v = safe_float(aspect_deg)
    if v is None or not np.isfinite(v):
        return None
    return float(v % 360.0)


def classify_aspect_class(
    aspect_deg: Optional[float],
    slope_deg: Optional[float],
    flat_slope_threshold_deg: float = DEFAULT_FLAT_SLOPE_THRESHOLD_DEG,
) -> Optional[str]:
    s = safe_float(slope_deg)
    if s is not None and np.isfinite(s) and s < flat_slope_threshold_deg:
        return "flat_no_aspect"

    a = normalize_aspect_deg(aspect_deg)
    if a is None:
        return None

    if a >= 338 or a <= 22:
        return "north"
    if 23 <= a <= 67:
        return "northeast"
    if 68 <= a <= 112:
        return "east"
    if 113 <= a <= 157:
        return "southeast"
    if 158 <= a <= 202:
        return "south"
    if 203 <= a <= 247:
        return "southwest"
    if 248 <= a <= 292:
        return "west"
    if 293 <= a <= 337:
        return "northwest"
    return None


def classify_aspect_class_cn(
    aspect_deg: Optional[float],
    slope_deg: Optional[float],
    flat_slope_threshold_deg: float = DEFAULT_FLAT_SLOPE_THRESHOLD_DEG,
) -> Optional[str]:
    m = {
        "north": "北坡",
        "northeast": "东北坡",
        "east": "东坡",
        "southeast": "东南坡",
        "south": "南坡",
        "southwest": "西南坡",
        "west": "西坡",
        "northwest": "西北坡",
        "flat_no_aspect": "无坡向",
    }
    k = classify_aspect_class(
        aspect_deg=aspect_deg,
        slope_deg=slope_deg,
        flat_slope_threshold_deg=flat_slope_threshold_deg,
    )
    return m.get(k)


def classify_landform_type(
    elevation_mean_m: Optional[float],
    relief_10km_m: Optional[float],
    plain_relief_threshold_m: float = DEFAULT_PLAIN_RELIEF_THRESHOLD_M,
) -> Optional[str]:
    elev = safe_float(elevation_mean_m)
    relief = safe_float(relief_10km_m)

    if elev is None or not np.isfinite(elev):
        return None

    if relief is not None and np.isfinite(relief) and relief <= plain_relief_threshold_m:
        return "plain"

    if 1000 <= elev <= 3499:
        return "mountain_middle"
    if 500 <= elev <= 999:
        return "mountain_low"
    if 250 <= elev <= 499:
        return "hill_high"
    if 100 <= elev <= 249:
        return "hill_middle"
    if elev < 100:
        return "hill_low"
    return None


def classify_landform_type_cn(
    elevation_mean_m: Optional[float],
    relief_10km_m: Optional[float],
    plain_relief_threshold_m: float = DEFAULT_PLAIN_RELIEF_THRESHOLD_M,
) -> Optional[str]:
    m = {
        "mountain_middle": "中山",
        "mountain_low": "低山",
        "hill_high": "高丘",
        "hill_middle": "中丘",
        "hill_low": "低丘",
        "plain": "平原",
    }
    k = classify_landform_type(
        elevation_mean_m=elevation_mean_m,
        relief_10km_m=relief_10km_m,
        plain_relief_threshold_m=plain_relief_threshold_m,
    )
    return m.get(k)


def classify_slope_position_class(
    slope_deg: Optional[float],
    relative_elevation_norm: Optional[float],
    tpi_local: Optional[float] = None,
    flow_accumulation_proxy: Optional[float] = None,
    flat_slope_threshold_deg: float = DEFAULT_FLAT_SLOPE_THRESHOLD_DEG,
    whole_slope_span_threshold: float = 0.8,
) -> Optional[str]:
    """
    轻量化坡位判定规则：
    1) 平地：坡度<5°
    2) 谷：flow accumulation 高或 tpi_local 显著负
    3) 脊：tpi_local 显著正且相对高程高
    4) 其他按 relative_elevation_norm 三等分
    5) 若对象跨越很大（relative_elevation_norm 缺失时无法判），返回 None
    """
    s = safe_float(slope_deg)
    rel = safe_float(relative_elevation_norm)
    tpi = safe_float(tpi_local)
    fa = safe_float(flow_accumulation_proxy)

    if s is not None and np.isfinite(s) and s < flat_slope_threshold_deg:
        return "flatland"

    if fa is not None and np.isfinite(fa) and fa >= 0.8:
        return "valley"

    if tpi is not None and np.isfinite(tpi):
        if tpi <= -0.3:
            return "valley"
        if tpi >= 0.3 and rel is not None and rel >= 0.8:
            return "ridge"

    if rel is None or not np.isfinite(rel):
        return None

    rel = min(max(rel, 0.0), 1.0)

    if rel >= whole_slope_span_threshold:
        return "ridge"
    if rel >= 2.0 / 3.0:
        return "upper"
    if rel >= 1.0 / 3.0:
        return "middle"
    return "lower"


def classify_slope_position_class_cn(
    slope_deg: Optional[float],
    relative_elevation_norm: Optional[float],
    tpi_local: Optional[float] = None,
    flow_accumulation_proxy: Optional[float] = None,
    flat_slope_threshold_deg: float = DEFAULT_FLAT_SLOPE_THRESHOLD_DEG,
    whole_slope_span_threshold: float = 0.8,
) -> Optional[str]:
    m = {
        "ridge": "脊",
        "upper": "上",
        "middle": "中",
        "lower": "下",
        "valley": "谷",
        "flatland": "平地",
        "whole_slope": "全坡",
    }
    k = classify_slope_position_class(
        slope_deg=slope_deg,
        relative_elevation_norm=relative_elevation_norm,
        tpi_local=tpi_local,
        flow_accumulation_proxy=flow_accumulation_proxy,
        flat_slope_threshold_deg=flat_slope_threshold_deg,
        whole_slope_span_threshold=whole_slope_span_threshold,
    )
    return m.get(k)


def circular_mean_deg(values: np.ndarray) -> Optional[float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    rad = np.deg2rad(vals)
    sin_m = np.mean(np.sin(rad))
    cos_m = np.mean(np.cos(rad))
    ang = np.rad2deg(np.arctan2(sin_m, cos_m))
    return float((ang + 360.0) % 360.0)


def dominant_class(values: list[str | None]) -> Optional[str]:
    vals = [v for v in values if v]
    if not vals:
        return None
    uniq, counts = np.unique(vals, return_counts=True)
    return str(uniq[np.argmax(counts)])


def summarize_terrain_classes(
    elevation_mean_m: Optional[float],
    relief_10km_m: Optional[float],
    slope_mean_deg: Optional[float],
    aspect_mean_deg: Optional[float],
    relative_elevation_norm: Optional[float] = None,
    tpi_local: Optional[float] = None,
    flow_accumulation_proxy: Optional[float] = None,
    rule_cfg: Optional[TerrainRuleConfig] = None,
) -> Dict[str, Any]:
    cfg = rule_cfg or TerrainRuleConfig()

    landform_type = classify_landform_type(
        elevation_mean_m=elevation_mean_m,
        relief_10km_m=relief_10km_m,
        plain_relief_threshold_m=cfg.plain_relief_threshold_m,
    )
    slope_class = classify_slope_class(slope_mean_deg)
    aspect_class = classify_aspect_class(
        aspect_deg=aspect_mean_deg,
        slope_deg=slope_mean_deg,
        flat_slope_threshold_deg=cfg.flat_slope_threshold_deg,
    )
    slope_position_class = classify_slope_position_class(
        slope_deg=slope_mean_deg,
        relative_elevation_norm=relative_elevation_norm,
        tpi_local=tpi_local,
        flow_accumulation_proxy=flow_accumulation_proxy,
        flat_slope_threshold_deg=cfg.flat_slope_threshold_deg,
    )

    return {
        "landform_type": landform_type,
        "landform_type_cn": classify_landform_type_cn(
            elevation_mean_m=elevation_mean_m,
            relief_10km_m=relief_10km_m,
            plain_relief_threshold_m=cfg.plain_relief_threshold_m,
        ),
        "slope_class": slope_class,
        "slope_class_cn": classify_slope_class_cn(slope_mean_deg),
        "aspect_class": aspect_class,
        "aspect_class_cn": classify_aspect_class_cn(
            aspect_deg=aspect_mean_deg,
            slope_deg=slope_mean_deg,
            flat_slope_threshold_deg=cfg.flat_slope_threshold_deg,
        ),
        "slope_position_class": slope_position_class,
        "slope_position_class_cn": classify_slope_position_class_cn(
            slope_deg=slope_mean_deg,
            relative_elevation_norm=relative_elevation_norm,
            tpi_local=tpi_local,
            flow_accumulation_proxy=flow_accumulation_proxy,
            flat_slope_threshold_deg=cfg.flat_slope_threshold_deg,
        ),
    }


def encode_class_to_int(class_name: Optional[str], mapping: Dict[str, int], nodata: int = 0) -> int:
    if class_name is None:
        return nodata
    return int(mapping.get(class_name, nodata))


LANDFORM_CODE = {
    "plain": 1,
    "hill_low": 2,
    "hill_middle": 3,
    "hill_high": 4,
    "mountain_low": 5,
    "mountain_middle": 6,
}

SLOPE_POSITION_CODE = {
    "flatland": 1,
    "valley": 2,
    "lower": 3,
    "middle": 4,
    "upper": 5,
    "ridge": 6,
    "whole_slope": 7,
}