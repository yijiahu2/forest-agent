from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from geo_layer.terrain_features import generate_terrain_products


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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
    from shapely.geometry import box

    with rasterio.open(src_raster) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {src_raster}")

        geom_in_src = geom_gdf.to_crs(src.crs)
        geom_in_src = geom_in_src[geom_in_src.geometry.notnull() & (~geom_in_src.geometry.is_empty)].copy()
        if geom_in_src.empty:
            raise ValueError(f"No valid geometry for raster crop: {src_raster}")

        raster_bounds_geom = box(*src.bounds)
        geom_union = geom_in_src.unary_union

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

    # 保底 Web Mercator，仅作为面积近似
    return "EPSG:3857"


def enrich_xiaoban_clip_fields(
    clipped_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    xiaoban_id_field: str,
    tree_count_field: Optional[str] = None,
    crown_field: Optional[str] = None,
    closure_field: Optional[str] = None,
    area_ha_field: Optional[str] = None,
    density_field: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    对裁剪后小班补充局部字段：
    - orig_geom_area_m2
    - inventory_area_m2
    - clip_area_m2
    - clip_area_ha
    - overlap_ratio_geom
    - overlap_ratio_inventory
    - overlap_ratio_in_xiaoban  (优先 inventory)
    - est_tree_count_clip
    - est_density_per_ha
    - est_tree_count_by_clip_area
    """

    if xiaoban_id_field not in clipped_gdf.columns:
        raise ValueError(f"Missing xiaoban id field in clipped gdf: {xiaoban_id_field}")
    if xiaoban_id_field not in source_gdf.columns:
        raise ValueError(f"Missing xiaoban id field in source gdf: {xiaoban_id_field}")

    clipped = clipped_gdf.copy()
    source = source_gdf.copy()

    clipped[xiaoban_id_field] = clipped[xiaoban_id_field].astype(str)
    source[xiaoban_id_field] = source[xiaoban_id_field].astype(str)

    metric_crs = _get_metric_crs(clipped)

    source_metric = source[[xiaoban_id_field, "geometry"]].copy().to_crs(metric_crs)
    source_metric["orig_geom_area_m2"] = source_metric.geometry.area.astype(float)

    source_area_cols = [xiaoban_id_field, "orig_geom_area_m2"]
    if area_ha_field and area_ha_field in source.columns:
        src_area_df = source[[xiaoban_id_field, area_ha_field]].copy()
        src_area_df[area_ha_field] = pd.to_numeric(src_area_df[area_ha_field], errors="coerce")
        src_area_df["inventory_area_m2"] = src_area_df[area_ha_field] * 10000.0
        source_area_cols_df = src_area_df[[xiaoban_id_field, "inventory_area_m2"]]
    else:
        source_area_cols_df = pd.DataFrame({xiaoban_id_field: source[xiaoban_id_field].astype(str), "inventory_area_m2": np.nan})

    source_area = source_metric[[xiaoban_id_field, "orig_geom_area_m2"]].merge(
        source_area_cols_df,
        on=xiaoban_id_field,
        how="left",
    )

    clipped_metric = clipped.to_crs(metric_crs)
    clipped["clip_area_m2"] = clipped_metric.geometry.area.astype(float)
    clipped["clip_area_ha"] = clipped["clip_area_m2"] / 10000.0

    clipped = clipped.merge(source_area, on=xiaoban_id_field, how="left")

    clipped["overlap_ratio_geom"] = clipped["clip_area_m2"] / clipped["orig_geom_area_m2"].replace(0, pd.NA)
    clipped["overlap_ratio_inventory"] = clipped["clip_area_m2"] / clipped["inventory_area_m2"].replace(0, pd.NA)

    clipped["overlap_ratio_in_xiaoban"] = clipped["overlap_ratio_inventory"]
    use_geom_mask = clipped["overlap_ratio_in_xiaoban"].isna() | ~np.isfinite(clipped["overlap_ratio_in_xiaoban"].astype(float))
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

    # 均值型字段原则上保持原值，作为局部先验
    if crown_field and crown_field in clipped.columns:
        clipped[crown_field] = pd.to_numeric(clipped[crown_field], errors="coerce")
    if closure_field and closure_field in clipped.columns:
        clipped[closure_field] = pd.to_numeric(clipped[closure_field], errors="coerce")

    return clipped


def clip_xiaoban_to_geometry(
    xiaoban_shp: str | Path,
    geom_gdf: gpd.GeoDataFrame,
    out_shp: str | Path,
    xiaoban_id_field: str,
    tree_count_field: Optional[str] = None,
    crown_field: Optional[str] = None,
    closure_field: Optional[str] = None,
    area_ha_field: Optional[str] = None,
    density_field: Optional[str] = None,
) -> str:
    xgdf = gpd.read_file(xiaoban_shp)
    if xgdf.crs is None:
        raise ValueError(f"Xiaoban shapefile has no CRS: {xiaoban_shp}")

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
    )

    ensure_parent(out_shp)
    clipped.to_file(out_shp)
    return str(out_shp)


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
        "dem_tif": None,
        "slope_tif": None,
        "aspect_tif": None,
        "xiaoban_shp": None,
    }

    if dem_tif:
        dem_clip = out_dir / "context_dem.tif"
        slope_clip = out_dir / "context_slope.tif"
        aspect_clip = out_dir / "context_aspect.tif"

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
            z_factor=1.0,
        )

        result["dem_tif"] = str(dem_clip)
        result["slope_tif"] = str(slope_clip)
        result["aspect_tif"] = str(aspect_clip)

    if xiaoban_shp:
        if not xiaoban_id_field:
            raise ValueError("xiaoban_id_field is required when xiaoban_shp is provided.")

        xiaoban_clip = out_dir / "context_xiaoban.shp"
        clip_xiaoban_to_geometry(
            xiaoban_shp=xiaoban_shp,
            geom_gdf=dom_bounds_gdf,
            out_shp=xiaoban_clip,
            xiaoban_id_field=xiaoban_id_field,
            tree_count_field=tree_count_field,
            crown_field=crown_field,
            closure_field=closure_field,
            area_ha_field=area_ha_field,
            density_field=density_field,
        )
        result["xiaoban_shp"] = str(xiaoban_clip)

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
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.summary_json:
        ensure_parent(args.summary_json)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()