import sys
from pathlib import Path
from geo_layer.terrain_constraints import (
    TerrainRuleConfig,
    summarize_terrain_classes,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from geo_layer.crown_metrics import (
    equivalent_crown_width,
    inventory_mean_crown_width_from_geometry,
    safe_float,
    standardize_inventory_crown_width,
)
from geo_layer.instance_ops import assign_instances_to_polygons
from geo_layer.terrain_features import generate_terrain_products


def normalize_closure(v):
    """
    郁闭度统一到 0~1
    若原值 > 1.5，则按百分数处理，除以100
    """
    x = safe_float(v)
    if x is None:
        return None
    if x > 1.5:
        x = x / 100.0
    return x


def union_area(geoms):
    """
    兼容不同版本 shapely/geopandas 的 union 面积计算
    """
    if len(geoms) == 0:
        return 0.0
    try:
        return geoms.union_all().area
    except Exception:
        try:
            return geoms.unary_union.area
        except Exception:
            return 0.0


def get_patch_polygon_from_raster(raster_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    patch = gpd.GeoDataFrame(
        {"patch_id": [Path(raster_path).stem]},
        geometry=[geom],
        crs=crs
    )
    return patch


def ensure_projected_metric_crs(gdf, fallback_crs=None):
    """
    若输入为地理坐标系（经纬度），则自动投影到 UTM
    """
    if gdf.crs is None:
        if fallback_crs is None:
            raise ValueError("Input CRS is None and no fallback CRS provided.")
        gdf = gdf.set_crs(fallback_crs)

    if gdf.crs.is_projected:
        return gdf

    try:
        utm_crs = gdf.estimate_utm_crs()
        return gdf.to_crs(utm_crs)
    except Exception as e:
        raise ValueError(f"Failed to estimate projected CRS: {e}")


def validate_field_exists(gdf, field_name, required=True):
    if field_name is None:
        return
    if field_name not in gdf.columns:
        if required:
            raise ValueError(f"Field '{field_name}' not found. Available fields: {list(gdf.columns)}")


def overlay_patch_xiaoban(patch_gdf, xiaoban_gdf, id_field):
    """
    计算 patch 与小班重叠，返回 clipped xiaoban
    """
    inter = gpd.overlay(xiaoban_gdf, patch_gdf, how="intersection")
    inter = inter[inter.geometry.notnull() & (~inter.geometry.is_empty)].copy()

    if len(inter) == 0:
        return inter

    inter["clip_area_m2"] = inter.geometry.area

    xiaoban_area = xiaoban_gdf[[id_field, "geometry"]].copy()
    xiaoban_area["xiaoban_area_m2"] = xiaoban_area.geometry.area
    xiaoban_area = xiaoban_area[[id_field, "xiaoban_area_m2"]]

    inter = inter.merge(xiaoban_area, on=id_field, how="left")

    total_clip_area = inter["clip_area_m2"].sum()
    if total_clip_area > 0:
        inter["overlap_ratio_in_patch"] = inter["clip_area_m2"] / total_clip_area
    else:
        inter["overlap_ratio_in_patch"] = 0.0

    inter["overlap_ratio_in_xiaoban"] = inter["clip_area_m2"] / inter["xiaoban_area_m2"].replace(0, pd.NA)
    inter["overlap_ratio_in_xiaoban"] = inter["overlap_ratio_in_xiaoban"].fillna(0.0)

    return inter


# =========================
# terrain 工具
# =========================

ASPECT_CLASS_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def classify_aspect_deg(aspect_deg: Optional[float]) -> Optional[str]:
    if aspect_deg is None or pd.isna(aspect_deg):
        return None
    x = float(aspect_deg) % 360.0
    bins = [
        (337.5, 360.0, "N"),
        (0.0, 22.5, "N"),
        (22.5, 67.5, "NE"),
        (67.5, 112.5, "E"),
        (112.5, 157.5, "SE"),
        (157.5, 202.5, "S"),
        (202.5, 247.5, "SW"),
        (247.5, 292.5, "W"),
        (292.5, 337.5, "NW"),
    ]
    for lo, hi, label in bins:
        if lo <= x < hi:
            return label
    return "N"


def circular_mean_deg(values: np.ndarray) -> Optional[float]:
    if values is None or len(values) == 0:
        return None
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None

    rad = np.deg2rad(vals)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))

    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return None

    ang = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    return float((ang + 360.0) % 360.0)


def dominant_aspect_class(values: np.ndarray) -> Optional[str]:
    if values is None or len(values) == 0:
        return None
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None

    classes = [classify_aspect_deg(v) for v in vals]
    classes = [c for c in classes if c is not None]
    if not classes:
        return None

    s = pd.Series(classes)
    vc = s.value_counts()
    return str(vc.index[0])


def _masked_values_from_geom(raster_path: str, geom_gdf: gpd.GeoDataFrame) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        geom = geom_gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in geom.geometry if g is not None and not g.is_empty]
        if not geoms:
            return np.array([], dtype=np.float32)

        out_image, _ = mask(src, geoms, crop=True, filled=False)
        band = out_image[0]

        if np.ma.isMaskedArray(band):
            vals = band.compressed()
        else:
            vals = band.reshape(-1)

        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]

        nodata = src.nodata
        if nodata is not None:
            vals = vals[~np.isclose(vals, nodata)]

        return vals


def raster_stats_for_geom(raster_path: str, geom_gdf: gpd.GeoDataFrame) -> Dict[str, Optional[float]]:
    vals = _masked_values_from_geom(raster_path, geom_gdf)

    if len(vals) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "relief": None,
            "count": 0,
        }

    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "relief": float(np.max(vals) - np.min(vals)),
        "count": int(len(vals)),
    }


def aspect_stats_for_geom(aspect_raster_path: str, geom_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    vals = _masked_values_from_geom(aspect_raster_path, geom_gdf)
    if len(vals) == 0:
        return {
            "mean_aspect_deg": None,
            "dominant_aspect_class": None,
            "aspect_count": 0,
        }

    return {
        "mean_aspect_deg": circular_mean_deg(vals),
        "dominant_aspect_class": dominant_aspect_class(vals),
        "aspect_count": int(len(vals)),
    }


def prepare_terrain_inputs(
    dem_tif: Optional[str],
    slope_tif: Optional[str],
    aspect_tif: Optional[str],
    work_dir: str,
) -> Dict[str, Any]:
    result = {
        "dem_tif": dem_tif,
        "slope_tif": slope_tif,
        "aspect_tif": aspect_tif,
        "landform_tif": None,
        "slope_position_tif": None,
        "terrain_generated": False,
    }

    if dem_tif is None:
        return result

    terrain_dir = Path(work_dir) / "terrain_cache"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    auto_slope = terrain_dir / f"{Path(dem_tif).stem}_slope.tif"
    auto_aspect = terrain_dir / f"{Path(dem_tif).stem}_aspect.tif"
    auto_landform = terrain_dir / f"{Path(dem_tif).stem}_landform.tif"
    auto_slope_position = terrain_dir / f"{Path(dem_tif).stem}_slope_position.tif"

    if slope_tif is not None and aspect_tif is not None and auto_landform.exists() and auto_slope_position.exists():
        result["landform_tif"] = str(auto_landform)
        result["slope_position_tif"] = str(auto_slope_position)
        return result

    generate_terrain_products(
        dem_tif=dem_tif,
        slope_tif=str(auto_slope),
        aspect_tif=str(auto_aspect),
        landform_tif=str(auto_landform),
        slope_position_tif=str(auto_slope_position),
        z_factor=1.0,
    )

    result["slope_tif"] = str(auto_slope)
    result["aspect_tif"] = str(auto_aspect)
    result["landform_tif"] = str(auto_landform)
    result["slope_position_tif"] = str(auto_slope_position)
    result["terrain_generated"] = True
    return result


def summarize_stratified_errors(details_df: pd.DataFrame) -> Dict[str, Any]:
    if details_df is None or len(details_df) == 0:
        return {}

    metric_cols = [
        "tree_count_error_ratio",
        "mean_crown_width_error_ratio",
        "closure_error_abs",
        "density_error_abs",
    ]
    out: Dict[str, Any] = {}
    for col in ["landform_type", "slope_class", "aspect_class", "slope_position_class"]:
        if col not in details_df.columns:
            continue
        rows = []
        for key, sub in details_df.groupby(col, dropna=False):
            rec: Dict[str, Any] = {
                col: None if pd.isna(key) else key,
                "num_samples": int(len(sub)),
            }
            for metric_col in metric_cols:
                if metric_col in sub.columns:
                    vals = pd.to_numeric(sub[metric_col], errors="coerce").dropna()
                    rec[f"{metric_col}_mean"] = float(vals.mean()) if len(vals) > 0 else None
            rows.append(rec)
        out[f"by_{col}"] = rows
    return out


def attach_terrain_stats_to_xiaoban_clip(
    xiaoban_clip_gdf: gpd.GeoDataFrame,
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    flat_slope_threshold_deg: float = 5.0,
    plain_relief_threshold_m: float = 30.0,
) -> gpd.GeoDataFrame:
    """
    给每个 clipped 小班附加 terrain 统计字段，并规范输出 DEM 四元组：
    - elevation_mean_m
    - relief_10km_m
    - slope_mean_deg
    - aspect_mean_deg
    - landform_type
    - slope_class
    - aspect_class
    - slope_position_class
    """
    out = xiaoban_clip_gdf.copy()
    rule_cfg = TerrainRuleConfig(
        flat_slope_threshold_deg=flat_slope_threshold_deg,
        plain_relief_threshold_m=plain_relief_threshold_m,
    )

    mean_elev_list = []
    std_elev_list = []
    relief_elev_list = []
    mean_slope_list = []
    std_slope_list = []
    mean_aspect_list = []

    landform_type_list = []
    landform_type_cn_list = []
    slope_class_list = []
    slope_class_cn_list = []
    aspect_class_list = []
    aspect_class_cn_list = []
    slope_position_class_list = []
    slope_position_class_cn_list = []
    rel_norm_list = []

    for _, row in out.iterrows():
        geom_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[row.geometry], crs=out.crs)

        dem_st = raster_stats_for_geom(dem_tif, geom_gdf) if dem_tif is not None else {}
        slope_st = raster_stats_for_geom(slope_tif, geom_gdf) if slope_tif is not None else {}
        aspect_st = aspect_stats_for_geom(aspect_tif, geom_gdf) if aspect_tif is not None else {}

        mean_elev = dem_st.get("mean")
        std_elev = dem_st.get("std")
        relief_elev = dem_st.get("relief")
        mean_slope = slope_st.get("mean")
        std_slope = slope_st.get("std")
        mean_aspect = aspect_st.get("mean_aspect_deg")

        rel_norm = None
        dem_min = dem_st.get("min")
        dem_max = dem_st.get("max")
        if mean_elev is not None and dem_min is not None and dem_max is not None and dem_max > dem_min:
            rel_norm = float((mean_elev - dem_min) / (dem_max - dem_min))
        elif mean_elev is not None:
            rel_norm = 0.5

        terrain_summary = summarize_terrain_classes(
            elevation_mean_m=mean_elev,
            relief_10km_m=relief_elev,
            slope_mean_deg=mean_slope,
            aspect_mean_deg=mean_aspect,
            relative_elevation_norm=rel_norm,
            tpi_local=None,
            flow_accumulation_proxy=None,
            rule_cfg=rule_cfg,
        )

        mean_elev_list.append(mean_elev)
        std_elev_list.append(std_elev)
        relief_elev_list.append(relief_elev)
        mean_slope_list.append(mean_slope)
        std_slope_list.append(std_slope)
        mean_aspect_list.append(mean_aspect)
        rel_norm_list.append(rel_norm)

        landform_type_list.append(terrain_summary["landform_type"])
        landform_type_cn_list.append(terrain_summary["landform_type_cn"])
        slope_class_list.append(terrain_summary["slope_class"])
        slope_class_cn_list.append(terrain_summary["slope_class_cn"])
        aspect_class_list.append(terrain_summary["aspect_class"])
        aspect_class_cn_list.append(terrain_summary["aspect_class_cn"])
        slope_position_class_list.append(terrain_summary["slope_position_class"])
        slope_position_class_cn_list.append(terrain_summary["slope_position_class_cn"])

    out["elevation_mean_m"] = mean_elev_list
    out["elevation_std_m"] = std_elev_list
    out["relief_10km_m"] = relief_elev_list
    out["slope_mean_deg"] = mean_slope_list
    out["slope_std_deg"] = std_slope_list
    out["aspect_mean_deg"] = mean_aspect_list
    out["relative_elevation_norm"] = rel_norm_list

    out["landform_type"] = landform_type_list
    out["landform_type_cn"] = landform_type_cn_list
    out["slope_class"] = slope_class_list
    out["slope_class_cn"] = slope_class_cn_list
    out["aspect_class"] = aspect_class_list
    out["aspect_class_cn"] = aspect_class_cn_list
    out["slope_position_class"] = slope_position_class_list
    out["slope_position_class_cn"] = slope_position_class_cn_list

    # 兼容旧字段，避免影响下游
    out["mean_elev"] = out["elevation_mean_m"]
    out["std_elev"] = out["elevation_std_m"]
    out["relief_elev"] = out["relief_10km_m"]
    out["mean_slope"] = out["slope_mean_deg"]
    out["std_slope"] = out["slope_std_deg"]
    out["mean_aspect_deg"] = out["aspect_mean_deg"]
    out["dominant_aspect_class"] = out["aspect_class"]

    return out

def compute_patch_terrain_metrics(
    patch_gdf: gpd.GeoDataFrame,
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    flat_slope_threshold_deg: float = 5.0,
    plain_relief_threshold_m: float = 30.0,
) -> Dict[str, Any]:
    metrics = {}
    dem_st = raster_stats_for_geom(dem_tif, patch_gdf) if dem_tif is not None else {}
    slope_st = raster_stats_for_geom(slope_tif, patch_gdf) if slope_tif is not None else {}
    aspect_st = aspect_stats_for_geom(aspect_tif, patch_gdf) if aspect_tif is not None else {}

    metrics["patch_mean_elev"] = dem_st.get("mean")
    metrics["patch_std_elev"] = dem_st.get("std")
    metrics["patch_relief_elev"] = dem_st.get("relief")
    metrics["patch_mean_slope"] = slope_st.get("mean")
    metrics["patch_std_slope"] = slope_st.get("std")
    metrics["patch_mean_aspect_deg"] = aspect_st.get("mean_aspect_deg")

    rel_norm = None
    if dem_st.get("mean") is not None and dem_st.get("min") is not None and dem_st.get("max") is not None:
        if dem_st["max"] > dem_st["min"]:
            rel_norm = float((dem_st["mean"] - dem_st["min"]) / (dem_st["max"] - dem_st["min"]))
        else:
            rel_norm = 0.5

    terrain_summary = summarize_terrain_classes(
        elevation_mean_m=dem_st.get("mean"),
        relief_10km_m=dem_st.get("relief"),
        slope_mean_deg=slope_st.get("mean"),
        aspect_mean_deg=aspect_st.get("mean_aspect_deg"),
        relative_elevation_norm=rel_norm,
        tpi_local=None,
        flow_accumulation_proxy=None,
        rule_cfg=TerrainRuleConfig(
            flat_slope_threshold_deg=flat_slope_threshold_deg,
            plain_relief_threshold_m=plain_relief_threshold_m,
        ),
    )

    metrics.update(
        {
            "patch_landform_type": terrain_summary["landform_type"],
            "patch_landform_type_cn": terrain_summary["landform_type_cn"],
            "patch_slope_class": terrain_summary["slope_class"],
            "patch_slope_class_cn": terrain_summary["slope_class_cn"],
            "patch_aspect_class": terrain_summary["aspect_class"],
            "patch_aspect_class_cn": terrain_summary["aspect_class_cn"],
            "patch_slope_position_class": terrain_summary["slope_position_class"],
            "patch_slope_position_class_cn": terrain_summary["slope_position_class_cn"],
        }
    )
    return metrics


def compute_patch_level_metrics(
    inst_gdf,
    xiaoban_clip_gdf,
    patch_area_m2,
    tree_count_field,
    crown_field,
    closure_field,
    density_field,
    area_ha_field,
    id_field
):
    """
    patch 级指标：
    - 预测树数
    - 期望树数（由小班树木总数按重叠比例缩放）
    - 预测平均冠幅
    - 期望平均冠幅（面积加权）
    - 预测覆盖率
    - 期望郁闭度（面积加权）
    - 预测密度（株/公顷）
    - 期望密度（优先 density_field，否则由 林木数量 / 面积/公顷 推导）
    """
    pred_tree_count = int(len(inst_gdf))

    if pred_tree_count > 0:
        pred_mean_crown_width = float(inst_gdf["inventory_crown_width_m"].mean())
    else:
        pred_mean_crown_width = 0.0

    if pred_tree_count > 0 and patch_area_m2 > 0:
        pred_cover_ratio = float(union_area(inst_gdf.geometry) / patch_area_m2)
    else:
        pred_cover_ratio = 0.0

    pred_density_trees_per_ha = float(pred_tree_count / (patch_area_m2 / 10000.0)) if patch_area_m2 > 0 else 0.0

    expected_tree_count = None
    if tree_count_field and tree_count_field in xiaoban_clip_gdf.columns:
        vals = []
        for _, row in xiaoban_clip_gdf.iterrows():
            total_count = safe_float(row.get(tree_count_field))
            r = safe_float(row.get("overlap_ratio_in_xiaoban"))
            if total_count is not None and r is not None:
                vals.append(total_count * r)
        if len(vals) > 0:
            expected_tree_count = float(sum(vals))

    expected_mean_crown_width = None
    expected_crown_col = "inventory_crown_width_m" if "inventory_crown_width_m" in xiaoban_clip_gdf.columns else crown_field
    if expected_crown_col and expected_crown_col in xiaoban_clip_gdf.columns:
        tmp = xiaoban_clip_gdf[[expected_crown_col, "clip_area_m2"]].copy()
        tmp[expected_crown_col] = tmp[expected_crown_col].apply(standardize_inventory_crown_width)
        tmp = tmp.dropna(subset=[expected_crown_col])
        if len(tmp) > 0 and tmp["clip_area_m2"].sum() > 0:
            expected_mean_crown_width = float(
                (tmp[expected_crown_col] * tmp["clip_area_m2"]).sum() / tmp["clip_area_m2"].sum()
            )

    expected_closure = None
    if closure_field and closure_field in xiaoban_clip_gdf.columns:
        tmp = xiaoban_clip_gdf[[closure_field, "clip_area_m2"]].copy()
        tmp[closure_field] = tmp[closure_field].apply(normalize_closure)
        tmp = tmp.dropna(subset=[closure_field])
        if len(tmp) > 0 and tmp["clip_area_m2"].sum() > 0:
            expected_closure = float(
                (tmp[closure_field] * tmp["clip_area_m2"]).sum() / tmp["clip_area_m2"].sum()
            )

    expected_density = None

    # 优先显式密度字段
    if density_field and density_field in xiaoban_clip_gdf.columns:
        tmp = xiaoban_clip_gdf[[density_field, "clip_area_m2"]].copy()
        tmp[density_field] = pd.to_numeric(tmp[density_field], errors="coerce")
        tmp = tmp.dropna(subset=[density_field])
        if len(tmp) > 0 and tmp["clip_area_m2"].sum() > 0:
            expected_density = float(
                (tmp[density_field] * tmp["clip_area_m2"]).sum() / tmp["clip_area_m2"].sum()
            )

    # 否则由 林木数量 / 面积/公顷 推导
    elif (
        tree_count_field and tree_count_field in xiaoban_clip_gdf.columns and
        area_ha_field and area_ha_field in xiaoban_clip_gdf.columns
    ):
        tmp = xiaoban_clip_gdf[[tree_count_field, area_ha_field, "clip_area_m2"]].copy()
        tmp[tree_count_field] = pd.to_numeric(tmp[tree_count_field], errors="coerce")
        tmp[area_ha_field] = pd.to_numeric(tmp[area_ha_field], errors="coerce")
        tmp = tmp.dropna(subset=[tree_count_field, area_ha_field])
        tmp = tmp[tmp[area_ha_field] > 0]
        if len(tmp) > 0 and tmp["clip_area_m2"].sum() > 0:
            tmp["derived_density"] = tmp[tree_count_field] / tmp[area_ha_field]
            expected_density = float(
                (tmp["derived_density"] * tmp["clip_area_m2"]).sum() / tmp["clip_area_m2"].sum()
            )

    dominant_row = xiaoban_clip_gdf.sort_values("clip_area_m2", ascending=False).iloc[0]
    dominant_xiaoban_id = str(dominant_row[id_field])
    dominant_xiaoban_ratio = float(dominant_row["overlap_ratio_in_patch"])

    metrics = {
        "num_overlapping_xiaoban": int(len(xiaoban_clip_gdf)),
        "dominant_xiaoban_ratio": dominant_xiaoban_ratio,
        "pred_tree_count": pred_tree_count,
        "pred_mean_crown_width": pred_mean_crown_width,
        "pred_cover_ratio": pred_cover_ratio,
        "pred_density_trees_per_ha": pred_density_trees_per_ha,
        "dominant_xiaoban_id": dominant_xiaoban_id,
    }

    if expected_tree_count is not None:
        metrics["expected_tree_count"] = expected_tree_count
        metrics["tree_count_error_abs"] = abs(pred_tree_count - expected_tree_count)
        metrics["tree_count_error_ratio"] = abs(pred_tree_count - expected_tree_count) / max(expected_tree_count, 1e-6)

    if expected_mean_crown_width is not None:
        metrics["expected_mean_crown_width"] = expected_mean_crown_width
        metrics["mean_crown_width_error_abs"] = abs(pred_mean_crown_width - expected_mean_crown_width)
        metrics["mean_crown_width_error_ratio"] = abs(pred_mean_crown_width - expected_mean_crown_width) / max(expected_mean_crown_width, 1e-6)

    if expected_closure is not None:
        metrics["expected_closure"] = expected_closure
        metrics["closure_error_abs"] = abs(pred_cover_ratio - expected_closure)

    if expected_density is not None:
        metrics["expected_density"] = expected_density
        metrics["density_error_abs"] = abs(pred_density_trees_per_ha - expected_density)

    return metrics


def compute_xiaoban_level_details(
    inst_assigned_gdf,
    xiaoban_clip_gdf,
    id_field,
    crown_field,
    closure_field,
    tree_count_field,
    area_ha_field,
    density_field
):
    """
    输出每个小班级别的明细表
    """
    rows = []

    for _, xb in xiaoban_clip_gdf.iterrows():
        xb_id = xb[id_field]
        xb_geom = xb.geometry
        xb_area_m2 = safe_float(xb["clip_area_m2"]) or 0.0

        sub = inst_assigned_gdf[inst_assigned_gdf[id_field] == xb_id].copy()
        pred_tree_count = int(len(sub))

        if pred_tree_count > 0 and xb_area_m2 > 0:
            pred_mean_crown_width = float(sub["inventory_crown_width_m"].mean())
            clipped_geoms = sub.geometry.intersection(xb_geom)
            pred_cover_ratio = float(union_area(clipped_geoms) / xb_area_m2)
        else:
            pred_mean_crown_width = 0.0
            pred_cover_ratio = 0.0

        row = {
            "xiaoban_id": xb_id,
            "clip_area_m2": xb_area_m2,
            "pred_tree_count": pred_tree_count,
            "pred_mean_crown_width": pred_mean_crown_width,
            "pred_cover_ratio": pred_cover_ratio,
        }

        if xb_area_m2 > 0:
            row["pred_density_trees_per_ha"] = pred_tree_count / (xb_area_m2 / 10000.0)
        else:
            row["pred_density_trees_per_ha"] = 0.0

        # 平均冠幅
        crown_value = xb.get("inventory_crown_width_m") if "inventory_crown_width_m" in xb.index else xb.get(crown_field)
        if crown_value is not None:
            row["expected_mean_crown_width"] = standardize_inventory_crown_width(crown_value)
            if row["expected_mean_crown_width"] is not None:
                row["mean_crown_width_error_abs"] = abs(pred_mean_crown_width - row["expected_mean_crown_width"])

        # 郁闭度
        if closure_field and closure_field in xb.index:
            row["expected_closure"] = normalize_closure(xb[closure_field])
            if row["expected_closure"] is not None:
                row["closure_error_abs"] = abs(pred_cover_ratio - row["expected_closure"])

        # 树木数量
        if tree_count_field and tree_count_field in xb.index:
            total_count = safe_float(xb[tree_count_field])
            overlap_ratio_in_xiaoban = safe_float(xb["overlap_ratio_in_xiaoban"])
            if total_count is not None and overlap_ratio_in_xiaoban is not None:
                exp_count = total_count * overlap_ratio_in_xiaoban
                row["expected_tree_count"] = exp_count
                row["tree_count_error_abs"] = abs(pred_tree_count - exp_count)

        # 密度：优先显式字段，否则由数量/面积推导
        exp_density = None
        if density_field and density_field in xb.index:
            exp_density = safe_float(xb[density_field])
        elif tree_count_field and tree_count_field in xb.index and area_ha_field and area_ha_field in xb.index:
            total_count = safe_float(xb[tree_count_field])
            area_ha = safe_float(xb[area_ha_field])
            if total_count is not None and area_ha is not None and area_ha > 0:
                exp_density = total_count / area_ha

        if exp_density is not None:
            row["expected_density"] = exp_density
            row["density_error_abs"] = abs(row["pred_density_trees_per_ha"] - exp_density)

        # terrain 字段（如果已附加到 xiaoban_clip_gdf）
        if "mean_elev" in xb.index:
            row["mean_elev"] = safe_float(xb["mean_elev"])
        if "std_elev" in xb.index:
            row["std_elev"] = safe_float(xb["std_elev"])
        if "relief_elev" in xb.index:
            row["relief_elev"] = safe_float(xb["relief_elev"])
        if "mean_slope" in xb.index:
            row["mean_slope"] = safe_float(xb["mean_slope"])
        if "std_slope" in xb.index:
            row["std_slope"] = safe_float(xb["std_slope"])
        if "mean_aspect_deg" in xb.index:
            row["mean_aspect_deg"] = safe_float(xb["mean_aspect_deg"])
        if "dominant_aspect_class" in xb.index:
            row["dominant_aspect_class"] = xb["dominant_aspect_class"]
        if "elevation_mean_m" in xb.index:
            row["elevation_mean_m"] = safe_float(xb["elevation_mean_m"])
        if "elevation_std_m" in xb.index:
            row["elevation_std_m"] = safe_float(xb["elevation_std_m"])
        if "relief_10km_m" in xb.index:
            row["relief_10km_m"] = safe_float(xb["relief_10km_m"])
        if "slope_mean_deg" in xb.index:
            row["slope_mean_deg"] = safe_float(xb["slope_mean_deg"])
        if "slope_std_deg" in xb.index:
            row["slope_std_deg"] = safe_float(xb["slope_std_deg"])
        if "aspect_mean_deg" in xb.index:
            row["aspect_mean_deg"] = safe_float(xb["aspect_mean_deg"])
        if "relative_elevation_norm" in xb.index:
            row["relative_elevation_norm"] = safe_float(xb["relative_elevation_norm"])
        if "landform_type" in xb.index:
            row["landform_type"] = xb["landform_type"]
        if "landform_type_cn" in xb.index:
            row["landform_type_cn"] = xb["landform_type_cn"]
        if "slope_class" in xb.index:
            row["slope_class"] = xb["slope_class"]
        if "slope_class_cn" in xb.index:
            row["slope_class_cn"] = xb["slope_class_cn"]
        if "aspect_class" in xb.index:
            row["aspect_class"] = xb["aspect_class"]
        if "aspect_class_cn" in xb.index:
            row["aspect_class_cn"] = xb["aspect_class_cn"]
        if "slope_position_class" in xb.index:
            row["slope_position_class"] = xb["slope_position_class"]
        if "slope_position_class_cn" in xb.index:
            row["slope_position_class_cn"] = xb["slope_position_class_cn"]

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst_shp", required=True, help="Path to Y_inst.shp")
    parser.add_argument("--patch_raster", required=True, help="Path to patch tif / original input tif")
    parser.add_argument("--xiaoban_shp", required=True, help="Path to xiaoban polygon shp/gpkg")
    parser.add_argument("--out_json", required=True, help="Output metrics json")
    parser.add_argument("--out_csv", required=True, help="Output detailed csv")

    parser.add_argument("--id_field", default="XBH", help="小班ID字段名")
    parser.add_argument("--tree_count_field", default="LMSL", help="林木数量字段")
    parser.add_argument("--crown_field", default="PJGF", help="平均冠幅字段")
    parser.add_argument("--closure_field", default="YBD", help="郁闭度字段")
    parser.add_argument("--density_field", default=None, help="林分密度字段（可空）")
    parser.add_argument("--area_ha_field", default="MJ_hm2", help="面积/公顷字段")
    parser.add_argument("--assign_method", default="max_overlap", choices=["max_overlap", "centroid"])

    # terrain 输入
    parser.add_argument("--dem_tif", default=None, help="Optional DEM tif")
    parser.add_argument("--slope_tif", default=None, help="Optional precomputed slope tif")
    parser.add_argument("--aspect_tif", default=None, help="Optional precomputed aspect tif")
    parser.add_argument("--flat_slope_threshold_deg", type=float, default=5.0, help="坡向无向阈值，默认5°")
    parser.add_argument("--plain_relief_threshold_m", type=float, default=30.0, help="平原起伏阈值，默认30m")


    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_work_dir = str(out_json.parent)

    terrain_info = prepare_terrain_inputs(
        dem_tif=args.dem_tif,
        slope_tif=args.slope_tif,
        aspect_tif=args.aspect_tif,
        work_dir=eval_work_dir,
    )

    # 1. patch polygon
    patch = get_patch_polygon_from_raster(args.patch_raster)
    patch = ensure_projected_metric_crs(patch)

    # 2. xiaoban
    xiaoban = gpd.read_file(args.xiaoban_shp)
    if xiaoban.crs is None:
        raise ValueError("Xiaoban shapefile has no CRS.")

    validate_field_exists(xiaoban, args.id_field, required=True)
    validate_field_exists(xiaoban, args.tree_count_field, required=False)
    validate_field_exists(xiaoban, args.crown_field, required=False)
    validate_field_exists(xiaoban, args.closure_field, required=False)
    validate_field_exists(xiaoban, args.area_ha_field, required=False)
    if args.density_field is not None:
        validate_field_exists(xiaoban, args.density_field, required=False)

    xiaoban = xiaoban.to_crs(patch.crs)
    xiaoban = xiaoban[xiaoban.geometry.notnull() & (~xiaoban.geometry.is_empty)].copy()

    xiaoban_clip = overlay_patch_xiaoban(patch, xiaoban, args.id_field)
    if len(xiaoban_clip) == 0:
        raise ValueError("No overlapping xiaoban found for this patch.")

    # 2.1 terrain attach to clipped xiaoban
    xiaoban_clip = attach_terrain_stats_to_xiaoban_clip(
        xiaoban_clip_gdf=xiaoban_clip,
        dem_tif=terrain_info["dem_tif"],
        slope_tif=terrain_info["slope_tif"],
        aspect_tif=terrain_info["aspect_tif"],
        flat_slope_threshold_deg=args.flat_slope_threshold_deg,
        plain_relief_threshold_m=args.plain_relief_threshold_m,
    )

    patch_terrain_metrics = compute_patch_terrain_metrics(
        patch_gdf=patch,
        dem_tif=terrain_info["dem_tif"],
        slope_tif=terrain_info["slope_tif"],
        aspect_tif=terrain_info["aspect_tif"],
        flat_slope_threshold_deg=args.flat_slope_threshold_deg,
        plain_relief_threshold_m=args.plain_relief_threshold_m,
    )

    # 3. instances
    inst = gpd.read_file(args.inst_shp)
    if inst.crs is None:
        raise ValueError("Instance shapefile has no CRS.")

    inst = inst.to_crs(patch.crs)
    inst = inst[inst.geometry.notnull() & (~inst.geometry.is_empty)].copy()

    # 裁剪到 patch
    inst = gpd.overlay(inst, patch, how="intersection")
    inst = inst[inst.geometry.notnull() & (~inst.geometry.is_empty)].copy()

    patch_area_m2 = float(patch.geometry.area.iloc[0])

    if len(inst) == 0:
        metrics = compute_patch_level_metrics(
            inst,
            xiaoban_clip,
            patch_area_m2,
            args.tree_count_field,
            args.crown_field,
            args.closure_field,
            args.density_field,
            args.area_ha_field,
            args.id_field
        )
        metrics["num_assigned_instances"] = 0
        metrics["num_boundary_instances"] = 0
        metrics["terrain_info"] = terrain_info
        metrics["terrain_dem_tif"] = terrain_info.get("dem_tif")
        metrics["terrain_slope_tif"] = terrain_info.get("slope_tif")
        metrics["terrain_aspect_tif"] = terrain_info.get("aspect_tif")
        metrics.update(patch_terrain_metrics)

        out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Saved metrics to {out_json}")
        print(f"Saved details to {out_csv}")
        return

    inst["inst_area_m2"] = inst.geometry.area
    inst["eq_crown_width_m"] = inst["inst_area_m2"].apply(equivalent_crown_width)
    inst["inventory_crown_width_m"] = inst.geometry.apply(inventory_mean_crown_width_from_geometry)

    # 4. assign instances to xiaoban
    inst_assigned = assign_instances_to_polygons(
        inst,
        xiaoban_clip,
        args.id_field,
        method=args.assign_method
    )

    num_boundary_instances = int(inst_assigned[args.id_field].isna().sum())
    inst_assigned_valid = inst_assigned.dropna(subset=[args.id_field]).copy()

    # 5. patch-level metrics
    metrics = compute_patch_level_metrics(
        inst_assigned_valid,
        xiaoban_clip,
        patch_area_m2,
        args.tree_count_field,
        args.crown_field,
        args.closure_field,
        args.density_field,
        args.area_ha_field,
        args.id_field
    )
    metrics["num_assigned_instances"] = int(len(inst_assigned_valid))
    metrics["num_boundary_instances"] = num_boundary_instances
    metrics["terrain_info"] = terrain_info
    metrics["terrain_dem_tif"] = terrain_info.get("dem_tif")
    metrics["terrain_slope_tif"] = terrain_info.get("slope_tif")
    metrics["terrain_aspect_tif"] = terrain_info.get("aspect_tif")
    metrics.update(patch_terrain_metrics)

    # 6. xiaoban-level detail
    details_df = compute_xiaoban_level_details(
        inst_assigned_valid,
        xiaoban_clip,
        args.id_field,
        args.crown_field,
        args.closure_field,
        args.tree_count_field,
        args.area_ha_field,
        args.density_field
    )
    metrics["terrain_stratified_error_summary"] = summarize_stratified_errors(details_df)

    # 7. save
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    details_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved metrics to {out_json}")
    print(f"Saved details to {out_csv}")


if __name__ == "__main__":
    main()
