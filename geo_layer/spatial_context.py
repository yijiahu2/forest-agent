# spatial_context.py

# 这是场景装配层 / 主流程接入层。
# 它负责把第四层结果真正接到主线 ROI 流程里，例如：

# 裁剪 DOM 同范围 DE
# 调用 terrain_features.py 生成 slope/aspect/landform/slope_position
# 裁剪小班矢量
# 把 DEM 四元组和连续统计回写进 context_xiaoban.gpkg
# 返回给主程序统一消费

# 所以它解决的是：
# 怎么接进主流程

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from geo_layer.crown_metrics import standardize_inventory_crown_width
from geo_layer.terrain_features import generate_terrain_products
from geo_layer.terrain_constraints import summarize_terrain_classes, TerrainRuleConfig


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_vector_auto(gdf: gpd.GeoDataFrame, out_path: str | Path, layer: Optional[str] = None) -> str:
    out_path = Path(out_path)
    ensure_parent(out_path)

    suffix = out_path.suffix.lower()
    if suffix == ".gpkg":
        out_layer = layer or out_path.stem
        gdf.to_file(out_path, driver="GPKG", layer=out_layer)
    elif suffix == ".shp":
        gdf.to_file(out_path)
    else:
        raise ValueError(f"Unsupported vector output format: {out_path}")
    return str(out_path)


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def load_dom_bounds(dom_tif: str | Path) -> Tuple[Any, Any, Dict[str, float]]:
    with rasterio.open(dom_tif) as src:
        crs = src.crs
        if crs is None:
            raise ValueError(f"DOM has no CRS: {dom_tif}")
        bounds = src.bounds
        profile = src.profile.copy()
        bounds_dict = {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top),
        }
    return crs, profile, bounds_dict


def build_bounds_gdf(bounds_dict: Dict[str, float], crs) -> gpd.GeoDataFrame:
    geom = box(
        bounds_dict["left"],
        bounds_dict["bottom"],
        bounds_dict["right"],
        bounds_dict["top"],
    )
    return gpd.GeoDataFrame({"_id": [1]}, geometry=[geom], crs=crs)


def crop_raster_to_geometry(
    src_raster: str | Path,
    geom_gdf: gpd.GeoDataFrame,
    out_raster: str | Path,
    all_touched: bool = False,
) -> str:
    from shapely.geometry import box as shp_box

    with rasterio.open(src_raster) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {src_raster}")

        geom_in_src = geom_gdf.to_crs(src.crs)
        geom_in_src = geom_in_src[geom_in_src.geometry.notnull() & (~geom_in_src.geometry.is_empty)].copy()
        if geom_in_src.empty:
            raise ValueError(f"No valid geometry for raster crop: {src_raster}")

        raster_bounds_geom = shp_box(*src.bounds)
        geom_union = geom_in_src.geometry.union_all()

        print("\n[spatial_context] crop check")
        print("src_raster:", src_raster)
        print("raster_crs:", src.crs)
        print("raster_bounds:", tuple(src.bounds))
        print("geom_crs_after_reproject:", geom_in_src.crs)
        print("geom_bounds_after_reproject:", tuple(geom_in_src.total_bounds))

        if geom_union.is_empty:
            raise ValueError(f"Geometry becomes empty after CRS transform for raster crop: {src_raster}")

        if not geom_union.intersects(raster_bounds_geom):
            raise ValueError(
                "\n".join(
                    [
                        f"Input shapes do not overlap raster: {src_raster}",
                        f"Raster CRS: {src.crs}",
                        f"Raster bounds: {tuple(src.bounds)}",
                        f"Geometry CRS(after reprojection): {geom_in_src.crs}",
                        f"Geometry bounds(after reprojection): {tuple(geom_in_src.total_bounds)}",
                        "Please check whether DOM and DEM truly overlap and whether their CRS/georeferencing are correct.",
                    ]
                )
            )

        geoms = [g.__geo_interface__ for g in geom_in_src.geometry if g is not None and not g.is_empty]
        out_image, out_transform = mask(
            src,
            geoms,
            crop=True,
            all_touched=all_touched,
            filled=True,
        )

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        ensure_parent(out_raster)
        with rasterio.open(out_raster, "w", **out_meta) as dst:
            dst.write(out_image)

    return str(out_raster)


def _get_metric_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS.")
    if getattr(gdf.crs, "is_projected", False):
        return gdf.crs
    try:
        utm_crs = gdf.estimate_utm_crs()
        if utm_crs is not None:
            return utm_crs
    except Exception:
        pass
    return "EPSG:3857"


def _masked_values_from_geom(raster_path: str, geom_gdf: gpd.GeoDataFrame) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        geom = geom_gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in geom.geometry if g is not None and not g.is_empty]
        if not geoms:
            return np.array([], dtype=np.float32)

        out_image, _ = mask(src, geoms, crop=True, filled=False)
        band = out_image[0]
        vals = band.compressed() if np.ma.isMaskedArray(band) else band.reshape(-1)
        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]

        nodata = src.nodata
        if nodata is not None:
            vals = vals[~np.isclose(vals, nodata)]
        return vals


def raster_stats_for_geom(raster_path: str, geom_gdf: gpd.GeoDataFrame) -> Dict[str, Optional[float]]:
    vals = _masked_values_from_geom(raster_path, geom_gdf)
    if len(vals) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "relief": None, "count": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "relief": float(np.max(vals) - np.min(vals)),
        "count": int(len(vals)),
    }


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


def aspect_stats_for_geom(aspect_raster_path: str, geom_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    vals = _masked_values_from_geom(aspect_raster_path, geom_gdf)
    if len(vals) == 0:
        return {"mean_aspect_deg": None, "aspect_count": 0}
    return {
        "mean_aspect_deg": circular_mean_deg(vals),
        "aspect_count": int(len(vals)),
    }


def _dominant_non_null(series: pd.Series) -> Optional[str]:
    vals = [str(v) for v in series.dropna().tolist() if str(v).strip()]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def summarize_xiaoban_terrain_classes(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    if gdf is None or len(gdf) == 0:
        return {}

    summary: Dict[str, Any] = {
        "num_xiaoban": int(len(gdf)),
        "dominant_landform": _dominant_non_null(gdf["landform_type"]) if "landform_type" in gdf.columns else None,
        "dominant_slope_class": _dominant_non_null(gdf["slope_class"]) if "slope_class" in gdf.columns else None,
        "dominant_aspect_class": _dominant_non_null(gdf["aspect_class"]) if "aspect_class" in gdf.columns else None,
        "dominant_slope_position_class": _dominant_non_null(gdf["slope_position_class"]) if "slope_position_class" in gdf.columns else None,
    }

    for col in ["elevation_mean_m", "slope_mean_deg", "aspect_mean_deg", "relief_10km_m"]:
        if col in gdf.columns:
            vals = pd.to_numeric(gdf[col], errors="coerce").dropna()
            summary[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else None

    return summary


def enrich_xiaoban_clip_fields(
    clipped_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    xiaoban_id_field: str,
    tree_count_field: Optional[str] = None,
    crown_field: Optional[str] = None,
    closure_field: Optional[str] = None,
    area_ha_field: Optional[str] = None,
    density_field: Optional[str] = None,
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    terrain_rule_cfg: Optional[TerrainRuleConfig] = None,
) -> gpd.GeoDataFrame:
    if xiaoban_id_field not in clipped_gdf.columns:
        raise ValueError(f"Missing xiaoban id field in clipped gdf: {xiaoban_id_field}")
    if xiaoban_id_field not in source_gdf.columns:
        raise ValueError(f"Missing xiaoban id field in source gdf: {xiaoban_id_field}")

    clipped = clipped_gdf.copy()
    source = source_gdf.copy()

    # 支持对已经富化过一次的小班再次裁剪并重算。
    # 否则 clipped/source 中已有的派生列会在 merge 时触发同名冲突，
    # 生成 *_x/*_y 后缀，后续再按标准列名读取就会报 KeyError。
    derived_fields_to_reset = [
        "clip_area_m2",
        "clip_area_ha",
        "orig_geom_area_m2",
        "inventory_area_m2",
        "overlap_ratio_geom",
        "overlap_ratio_inventory",
        "overlap_ratio_in_xiaoban",
        "est_tree_count_clip",
        "est_density_per_ha",
        "est_tree_count_by_clip_area",
        "inventory_crown_width_m",
        "elevation_mean_m",
        "relief_10km_m",
        "slope_mean_deg",
        "aspect_mean_deg",
        "relative_elevation_norm",
        "landform_type",
        "landform_type_cn",
        "slope_class",
        "slope_class_cn",
        "aspect_class",
        "aspect_class_cn",
        "slope_position_class",
        "slope_position_class_cn",
    ]
    clipped = clipped.drop(columns=[c for c in derived_fields_to_reset if c in clipped.columns], errors="ignore")

    clipped[xiaoban_id_field] = clipped[xiaoban_id_field].astype(str)
    source[xiaoban_id_field] = source[xiaoban_id_field].astype(str)

    metric_crs = _get_metric_crs(clipped)
    source_metric = source[[xiaoban_id_field, "geometry"]].copy().to_crs(metric_crs)
    source_metric["orig_geom_area_m2"] = source_metric.geometry.area.astype(float)

    if area_ha_field and area_ha_field in source.columns:
        src_area_df = source[[xiaoban_id_field, area_ha_field]].copy()
        src_area_df[area_ha_field] = pd.to_numeric(src_area_df[area_ha_field], errors="coerce")
        src_area_df["inventory_area_m2"] = src_area_df[area_ha_field] * 10000.0
        source_area_cols_df = src_area_df[[xiaoban_id_field, "inventory_area_m2"]]
    else:
        source_area_cols_df = pd.DataFrame(
            {
                xiaoban_id_field: source[xiaoban_id_field].astype(str),
                "inventory_area_m2": np.nan,
            }
        )

    source_area = source_metric[[xiaoban_id_field, "orig_geom_area_m2"]].merge(
        source_area_cols_df,
        on=xiaoban_id_field,
        how="left",
    )
    source_area = source_area.drop_duplicates(subset=[xiaoban_id_field], keep="first")

    clipped_metric = clipped.to_crs(metric_crs)
    clipped["clip_area_m2"] = clipped_metric.geometry.area.astype(float)
    clipped["clip_area_ha"] = clipped["clip_area_m2"] / 10000.0
    clipped = clipped.merge(source_area, on=xiaoban_id_field, how="left")

    clipped["overlap_ratio_geom"] = clipped["clip_area_m2"] / clipped["orig_geom_area_m2"].replace(0, pd.NA)
    clipped["overlap_ratio_inventory"] = clipped["clip_area_m2"] / clipped["inventory_area_m2"].replace(0, pd.NA)
    clipped["overlap_ratio_in_xiaoban"] = clipped["overlap_ratio_inventory"]

    overlap_numeric = pd.to_numeric(clipped["overlap_ratio_in_xiaoban"], errors="coerce")
    use_geom_mask = overlap_numeric.isna() | ~np.isfinite(overlap_numeric)
    clipped.loc[use_geom_mask, "overlap_ratio_in_xiaoban"] = clipped.loc[use_geom_mask, "overlap_ratio_geom"]
    clipped["overlap_ratio_in_xiaoban"] = clipped["overlap_ratio_in_xiaoban"].fillna(0.0).clip(lower=0.0, upper=1.0)

    if tree_count_field and tree_count_field in clipped.columns:
        clipped[tree_count_field] = pd.to_numeric(clipped[tree_count_field], errors="coerce")
        clipped["est_tree_count_clip"] = clipped[tree_count_field] * clipped["overlap_ratio_in_xiaoban"]

    if density_field and density_field in clipped.columns:
        clipped[density_field] = pd.to_numeric(clipped[density_field], errors="coerce")
        clipped["est_density_per_ha"] = clipped[density_field]
        clipped["est_tree_count_by_clip_area"] = clipped["est_density_per_ha"] * clipped["clip_area_ha"]
    elif tree_count_field and area_ha_field and tree_count_field in clipped.columns and area_ha_field in clipped.columns:
        clipped[tree_count_field] = pd.to_numeric(clipped[tree_count_field], errors="coerce")
        clipped[area_ha_field] = pd.to_numeric(clipped[area_ha_field], errors="coerce")
        clipped["est_density_per_ha"] = clipped[tree_count_field] / clipped[area_ha_field].replace(0, pd.NA)
        clipped["est_tree_count_by_clip_area"] = clipped["est_density_per_ha"] * clipped["clip_area_ha"]

    if crown_field and crown_field in clipped.columns:
        clipped[crown_field] = clipped[crown_field].apply(standardize_inventory_crown_width)
        clipped["inventory_crown_width_m"] = clipped[crown_field]
    if closure_field and closure_field in clipped.columns:
        clipped[closure_field] = pd.to_numeric(clipped[closure_field], errors="coerce")

    if dem_tif or slope_tif or aspect_tif:
        mean_elev_list = []
        relief_elev_list = []
        mean_slope_list = []
        mean_aspect_deg_list = []
        landform_type_list = []
        landform_type_cn_list = []
        slope_class_list = []
        slope_class_cn_list = []
        aspect_class_list = []
        aspect_class_cn_list = []
        rel_norm_list = []
        slope_position_class_list = []
        slope_position_class_cn_list = []

        rule_cfg = terrain_rule_cfg or TerrainRuleConfig()

        for _, row in clipped.iterrows():
            geom_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[row.geometry], crs=clipped.crs)

            dem_st = raster_stats_for_geom(dem_tif, geom_gdf) if dem_tif else {}
            slope_st = raster_stats_for_geom(slope_tif, geom_gdf) if slope_tif else {}
            aspect_st = aspect_stats_for_geom(aspect_tif, geom_gdf) if aspect_tif else {}

            mean_elev = dem_st.get("mean")
            relief_elev = dem_st.get("relief")
            mean_slope = slope_st.get("mean")
            mean_aspect_deg = aspect_st.get("mean_aspect_deg")

            rel_norm = None
            if mean_elev is not None and dem_st.get("min") is not None and dem_st.get("max") is not None:
                dem_min = dem_st.get("min")
                dem_max = dem_st.get("max")
                if dem_max is not None and dem_min is not None and dem_max > dem_min:
                    rel_norm = float((mean_elev - dem_min) / (dem_max - dem_min))
                else:
                    rel_norm = 0.5

            terrain_summary = summarize_terrain_classes(
                elevation_mean_m=mean_elev,
                relief_10km_m=relief_elev,
                slope_mean_deg=mean_slope,
                aspect_mean_deg=mean_aspect_deg,
                relative_elevation_norm=rel_norm,
                tpi_local=None,
                flow_accumulation_proxy=None,
                rule_cfg=rule_cfg,
            )

            mean_elev_list.append(mean_elev)
            relief_elev_list.append(relief_elev)
            mean_slope_list.append(mean_slope)
            mean_aspect_deg_list.append(mean_aspect_deg)
            rel_norm_list.append(rel_norm)
            landform_type_list.append(terrain_summary["landform_type"])
            landform_type_cn_list.append(terrain_summary["landform_type_cn"])
            slope_class_list.append(terrain_summary["slope_class"])
            slope_class_cn_list.append(terrain_summary["slope_class_cn"])
            aspect_class_list.append(terrain_summary["aspect_class"])
            aspect_class_cn_list.append(terrain_summary["aspect_class_cn"])
            slope_position_class_list.append(terrain_summary["slope_position_class"])
            slope_position_class_cn_list.append(terrain_summary["slope_position_class_cn"])

        clipped["elevation_mean_m"] = mean_elev_list
        clipped["relief_10km_m"] = relief_elev_list
        clipped["slope_mean_deg"] = mean_slope_list
        clipped["aspect_mean_deg"] = mean_aspect_deg_list
        clipped["relative_elevation_norm"] = rel_norm_list

        clipped["landform_type"] = landform_type_list
        clipped["landform_type_cn"] = landform_type_cn_list
        clipped["slope_class"] = slope_class_list
        clipped["slope_class_cn"] = slope_class_cn_list
        clipped["aspect_class"] = aspect_class_list
        clipped["aspect_class_cn"] = aspect_class_cn_list
        clipped["slope_position_class"] = slope_position_class_list
        clipped["slope_position_class_cn"] = slope_position_class_cn_list

    return clipped


def clip_xiaoban_to_geometry(
    xiaoban_shp: str | Path,
    geom_gdf: gpd.GeoDataFrame,
    out_vector: str | Path,
    xiaoban_id_field: str,
    tree_count_field: Optional[str] = None,
    crown_field: Optional[str] = None,
    closure_field: Optional[str] = None,
    area_ha_field: Optional[str] = None,
    density_field: Optional[str] = None,
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    flat_slope_threshold_deg: float = 5.0,
    plain_relief_threshold_m: float = 30.0,
) -> str:
    xgdf = gpd.read_file(xiaoban_shp)
    if xgdf.crs is None:
        raise ValueError(f"Xiaoban vector has no CRS: {xiaoban_shp}")

    geom_in_x = geom_gdf.to_crs(xgdf.crs)
    clipped = gpd.overlay(xgdf, geom_in_x, how="intersection")
    clipped = clipped[clipped.geometry.notnull() & (~clipped.geometry.is_empty)].copy()

    if clipped.empty:
        raise ValueError(f"Clipped xiaoban is empty: {xiaoban_shp}")

    clipped = enrich_xiaoban_clip_fields(
        clipped_gdf=clipped,
        source_gdf=xgdf,
        xiaoban_id_field=xiaoban_id_field,
        tree_count_field=tree_count_field,
        crown_field=crown_field,
        closure_field=closure_field,
        area_ha_field=area_ha_field,
        density_field=density_field,
        dem_tif=dem_tif,
        slope_tif=slope_tif,
        aspect_tif=aspect_tif,
        terrain_rule_cfg=TerrainRuleConfig(
            flat_slope_threshold_deg=flat_slope_threshold_deg,
            plain_relief_threshold_m=plain_relief_threshold_m,
        ),
    )

    return write_vector_auto(clipped, out_vector, layer=Path(out_vector).stem)


def prepare_spatial_context(
    dom_tif: str | Path,
    dem_tif: Optional[str | Path],
    xiaoban_shp: Optional[str | Path],
    out_dir: str | Path,
    xiaoban_id_field: Optional[str] = None,
    tree_count_field: Optional[str] = None,
    crown_field: Optional[str] = None,
    closure_field: Optional[str] = None,
    area_ha_field: Optional[str] = None,
    density_field: Optional[str] = None,
    all_touched: bool = False,
    flat_slope_threshold_deg: float = 5.0,
    plain_relief_threshold_m: float = 30.0,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    dom_crs, dom_profile, bounds_dict = load_dom_bounds(dom_tif)
    dom_bounds_gdf = build_bounds_gdf(bounds_dict, dom_crs)

    result = {
        "input_image": str(dom_tif),
        "dom_bounds": bounds_dict,
        "dom_crs": str(dom_crs),
        "dom_shape": [int(dom_profile["height"]), int(dom_profile["width"])],
        "global_dem_tif": str(dem_tif) if dem_tif else None,
        "global_slope_tif": None,
        "global_aspect_tif": None,
        "global_landform_tif": None,
        "global_slope_position_tif": None,
        "dem_tif": None,
        "slope_tif": None,
        "aspect_tif": None,
        "landform_tif": None,
        "slope_position_tif": None,
        "xiaoban_shp": None,
        "xiaoban_vector": None,
        "terrain_constraint_fields": {
            "landform_type": "landform_type",
            "slope_class": "slope_class",
            "aspect_class": "aspect_class",
            "slope_position_class": "slope_position_class",
        },
    }

    if dem_tif:
        global_slope = Path(dem_tif).with_name(f"{Path(dem_tif).stem}_slope.tif")
        global_aspect = Path(dem_tif).with_name(f"{Path(dem_tif).stem}_aspect.tif")
        global_landform = Path(dem_tif).with_name(f"{Path(dem_tif).stem}_landform.tif")
        global_slope_position = Path(dem_tif).with_name(f"{Path(dem_tif).stem}_slope_position.tif")
        global_summary_json = Path(dem_tif).with_name(f"{Path(dem_tif).stem}_terrain_summary.json")

        if not (
            global_slope.exists()
            and global_aspect.exists()
            and global_landform.exists()
            and global_slope_position.exists()
        ):
            generate_terrain_products(
                dem_tif=str(dem_tif),
                slope_tif=str(global_slope),
                aspect_tif=str(global_aspect),
                landform_tif=str(global_landform),
                slope_position_tif=str(global_slope_position),
                terrain_summary_json=str(global_summary_json),
                z_factor=1.0,
                flat_slope_threshold_deg=flat_slope_threshold_deg,
                plain_relief_threshold_m=plain_relief_threshold_m,
            )

        result["global_slope_tif"] = str(global_slope) if global_slope.exists() else None
        result["global_aspect_tif"] = str(global_aspect) if global_aspect.exists() else None
        result["global_landform_tif"] = str(global_landform) if global_landform.exists() else None
        result["global_slope_position_tif"] = str(global_slope_position) if global_slope_position.exists() else None

        dem_clip = out_dir / "context_dem.tif"
        slope_clip = out_dir / "context_slope.tif"
        aspect_clip = out_dir / "context_aspect.tif"
        landform_clip = out_dir / "context_landform.tif"
        slope_position_clip = out_dir / "context_slope_position.tif"
        terrain_summary_json = out_dir / "terrain_products_summary.json"

        crop_raster_to_geometry(
            src_raster=dem_tif,
            geom_gdf=dom_bounds_gdf,
            out_raster=dem_clip,
            all_touched=all_touched,
        )

        generate_terrain_products(
            dem_tif=str(dem_clip),
            slope_tif=str(slope_clip),
            aspect_tif=str(aspect_clip),
            landform_tif=str(landform_clip),
            slope_position_tif=str(slope_position_clip),
            terrain_summary_json=str(terrain_summary_json),
            z_factor=1.0,
            flat_slope_threshold_deg=flat_slope_threshold_deg,
            plain_relief_threshold_m=plain_relief_threshold_m,
        )

        result["dem_tif"] = str(dem_clip)
        result["slope_tif"] = str(slope_clip)
        result["aspect_tif"] = str(aspect_clip)
        result["landform_tif"] = str(landform_clip)
        result["slope_position_tif"] = str(slope_position_clip)
        result["terrain_summary_json"] = str(terrain_summary_json)

    if xiaoban_shp:
        if not xiaoban_id_field:
            raise ValueError("xiaoban_id_field is required when xiaoban_shp is provided.")

        xiaoban_clip = out_dir / "context_xiaoban.gpkg"

        clip_xiaoban_to_geometry(
            xiaoban_shp=xiaoban_shp,
            geom_gdf=dom_bounds_gdf,
            out_vector=xiaoban_clip,
            xiaoban_id_field=xiaoban_id_field,
            tree_count_field=tree_count_field,
            crown_field=crown_field,
            closure_field=closure_field,
            area_ha_field=area_ha_field,
            density_field=density_field,
            dem_tif=result["dem_tif"],
            slope_tif=result["slope_tif"],
            aspect_tif=result["aspect_tif"],
            flat_slope_threshold_deg=flat_slope_threshold_deg,
            plain_relief_threshold_m=plain_relief_threshold_m,
        )
        result["xiaoban_shp"] = str(xiaoban_clip)
        result["xiaoban_vector"] = str(xiaoban_clip)
        try:
            xiaoban_context = gpd.read_file(xiaoban_clip)
            result["terrain_class_summary"] = summarize_xiaoban_terrain_classes(xiaoban_context)
        except Exception:
            result["terrain_class_summary"] = {}

    summary_json = out_dir / "spatial_context_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["summary_json"] = str(summary_json)

    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DOM-aligned DEM / xiaoban spatial context.")
    parser.add_argument("--dom_tif", required=True)
    parser.add_argument("--dem_tif", default=None)
    parser.add_argument("--xiaoban_shp", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--xiaoban_id_field", default=None)
    parser.add_argument("--tree_count_field", default=None)
    parser.add_argument("--crown_field", default=None)
    parser.add_argument("--closure_field", default=None)
    parser.add_argument("--area_ha_field", default=None)
    parser.add_argument("--density_field", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--flat_slope_threshold_deg", type=float, default=5.0)
    parser.add_argument("--plain_relief_threshold_m", type=float, default=30.0)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    result = prepare_spatial_context(
        dom_tif=args.dom_tif,
        dem_tif=args.dem_tif,
        xiaoban_shp=args.xiaoban_shp,
        out_dir=args.out_dir,
        xiaoban_id_field=args.xiaoban_id_field,
        tree_count_field=args.tree_count_field,
        crown_field=args.crown_field,
        closure_field=args.closure_field,
        area_ha_field=args.area_ha_field,
        density_field=args.density_field,
        flat_slope_threshold_deg=args.flat_slope_threshold_deg,
        plain_relief_threshold_m=args.plain_relief_threshold_m,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.summary_json:
        ensure_parent(args.summary_json)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
