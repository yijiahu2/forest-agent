from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _normalize_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).replace("\ufeff", "").strip(): v for k, v in row.items()}


def _normalize_aspect(v: Any) -> str:
    s = _safe_str(v, "").lower()
    mapping = {
        "n": "north",
        "ne": "northeast",
        "e": "east",
        "se": "southeast",
        "s": "south",
        "sw": "southwest",
        "w": "west",
        "nw": "northwest",
    }
    return mapping.get(s, s)


def _terrain_context(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_slope_class = _safe_str(row.get("slope_class"), "").lower()
    mean_slope = _safe_float(row.get("mean_slope") or row.get("slope_mean_deg"), 0.0)
    relief = _safe_float(row.get("relief_elev") or row.get("relief_10km_m"), 0.0)
    landform = _safe_str(row.get("landform_type"), "").lower()
    slope_position = _safe_str(row.get("slope_position_class"), "").lower()
    aspect = _normalize_aspect(row.get("aspect_class") or row.get("dominant_aspect_class"))

    is_steep = raw_slope_class.startswith(("iv", "v", "vi")) or mean_slope >= 25.0
    is_high_relief = relief >= 20.0
    is_complex_landform = landform in {"mountain_middle", "mountain_low", "hill_high", "hill_middle"}
    is_ridge_valley = slope_position in {"ridge", "valley"}
    is_shaded = aspect in {"north", "northeast", "northwest"}

    return {
        "is_steep": is_steep,
        "is_high_relief": is_high_relief,
        "is_complex_landform": is_complex_landform,
        "is_ridge_valley": is_ridge_valley,
        "is_shaded": is_shaded,
    }


def _row_metric_values(row: Dict[str, Any]) -> Dict[str, float]:
    expected_tree_count = max(_safe_float(row.get("expected_tree_count"), 0.0), 1.0)
    expected_crown = max(_safe_float(row.get("expected_mean_crown_width"), 0.0), 1e-6)

    return {
        "tree": _safe_float(row.get("tree_count_error_abs"), 0.0) / expected_tree_count,
        "crown": _safe_float(row.get("mean_crown_width_error_abs"), 0.0) / expected_crown,
        "closure": _safe_float(row.get("closure_error_abs"), 0.0),
        "density": _safe_float(row.get("density_error_abs"), 0.0) / 1000.0,
    }


def compute_terrain_penalties(details_csv: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(details_csv)
    if not path.exists():
        return {
            "penalties": {"tree": 0.0, "crown": 0.0, "closure": 0.0, "density": 0.0},
            "groups": [],
        }

    with open(path, newline="", encoding="utf-8") as f:
        rows = [_normalize_row_keys(row) for row in csv.DictReader(f)]

    if not rows:
        return {
            "penalties": {"tree": 0.0, "crown": 0.0, "closure": 0.0, "density": 0.0},
            "groups": [],
        }

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        ctx = _terrain_context(row)
        vals = _row_metric_values(row)
        enriched.append({**row, **ctx, "_metric_values": vals})

    subgroup_specs = [
        (
            "steep_complex",
            0.40,
            lambda r: r["is_steep"] or r["is_high_relief"] or r["is_complex_landform"],
        ),
        (
            "ridge_valley",
            0.30,
            lambda r: r["is_ridge_valley"],
        ),
        (
            "shaded_steep",
            0.25,
            lambda r: r["is_shaded"] and (r["is_steep"] or r["is_high_relief"]),
        ),
    ]

    global_baseline = {
        "tree": float(metrics.get("tree_count_error_ratio", 0.0)),
        "crown": float(metrics.get("mean_crown_width_error_ratio", 0.0)),
        "closure": float(metrics.get("closure_error_abs", 0.0)),
        "density": float(metrics.get("density_error_abs", 0.0)) / 1000.0,
    }

    penalties = {"tree": 0.0, "crown": 0.0, "closure": 0.0, "density": 0.0}
    groups: List[Dict[str, Any]] = []
    total_n = max(len(enriched), 1)

    for name, base_weight, predicate in subgroup_specs:
        sub = [row for row in enriched if predicate(row)]
        if not sub:
            continue

        support = min(len(sub) / total_n, 1.0)
        subgroup_means = {}
        for metric_name in ["tree", "crown", "closure", "density"]:
            vals = [row["_metric_values"][metric_name] for row in sub]
            subgroup_means[metric_name] = sum(vals) / len(vals)

        excess = {
            metric_name: max(subgroup_means[metric_name] - global_baseline[metric_name], 0.0)
            for metric_name in subgroup_means
        }

        penalties["tree"] += excess["tree"] * base_weight * support
        penalties["crown"] += excess["crown"] * base_weight * support
        penalties["closure"] += excess["closure"] * base_weight * support
        penalties["density"] += excess["density"] * base_weight * support

        groups.append(
            {
                "name": name,
                "num_rows": len(sub),
                "support": support,
                "base_weight": base_weight,
                "means": subgroup_means,
                "excess": excess,
            }
        )

    return {"penalties": penalties, "groups": groups}
