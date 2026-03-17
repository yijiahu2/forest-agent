from __future__ import annotations

import math
from typing import Any

import pandas as pd


def safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def equivalent_crown_width(area_m2: float | None) -> float:
    if area_m2 is None or area_m2 <= 0:
        return 0.0
    return 2.0 * math.sqrt(area_m2 / math.pi)


def inventory_mean_crown_width_from_geometry(geom) -> float:
    """
    小班调查冠幅口径：(东西向冠幅 + 南北向冠幅) / 2。
    分割树冠没有单独的东西/南北量测值，因此用投影坐标系下
    geometry 包围盒的 x/y 方向跨度近似为东西/南北冠幅。
    """
    if geom is None or geom.is_empty:
        return 0.0
    minx, miny, maxx, maxy = geom.bounds
    ew = max(float(maxx - minx), 0.0)
    ns = max(float(maxy - miny), 0.0)
    return (ew + ns) / 2.0


def standardize_inventory_crown_width(value: Any) -> float | None:
    """
    当前项目中，小班平均冠幅字段已经定义为调查口径：
    (东西向冠幅 + 南北向冠幅) / 2，单位米。
    因缺少双轴原始值，无法可靠反推面积等效直径；这里统一为
    标准化后的调查口径数值，供 planner / evaluation 使用。
    """
    return safe_float(value, None)
