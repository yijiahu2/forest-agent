from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import pandas as pd


def assign_instances_to_polygons(
    inst_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    id_field: str,
    method: str = "max_overlap",
) -> gpd.GeoDataFrame:
    inst = inst_gdf.copy()
    inst = inst.drop(columns=[id_field], errors="ignore")
    inst["inst_id"] = range(len(inst))

    if inst.empty:
        return inst

    polygons = polygon_gdf[[id_field, "geometry"]].copy()
    polygons = polygons[polygons.geometry.notnull() & (~polygons.geometry.is_empty)].copy()
    if polygons.empty:
        inst[id_field] = pd.NA
        return inst

    inst = inst.to_crs(polygons.crs)

    if method == "centroid":
        cent = inst.copy()
        cent.geometry = cent.centroid
        joined = gpd.sjoin(cent, polygons, how="left", predicate="within")
        return inst.merge(joined[["inst_id", id_field]], on="inst_id", how="left")

    ov = gpd.overlay(inst[["inst_id", "geometry"]], polygons, how="intersection")
    ov = ov[ov.geometry.notnull() & (~ov.geometry.is_empty)].copy()

    if ov.empty:
        cent = inst.copy()
        cent.geometry = cent.centroid
        joined = gpd.sjoin(cent, polygons, how="left", predicate="within")
        return inst.merge(joined[["inst_id", id_field]], on="inst_id", how="left")

    ov["overlap_area_m2"] = ov.geometry.area
    ov = ov.sort_values(["inst_id", "overlap_area_m2"], ascending=[True, False])
    best = ov.groupby("inst_id", as_index=False).first()[["inst_id", id_field, "overlap_area_m2"]]
    return inst.merge(best, on="inst_id", how="left")


def filter_instances_to_ids_by_overlap(
    inst_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    id_field: str,
    allowed_ids: Iterable[str],
) -> gpd.GeoDataFrame:
    assigned = assign_instances_to_polygons(inst_gdf, polygon_gdf, id_field=id_field, method="max_overlap")
    allowed = {str(x) for x in allowed_ids}
    assigned[id_field] = assigned[id_field].astype(str)
    filtered = assigned[assigned[id_field].isin(allowed)].copy()
    return filtered.drop(columns=["inst_id", "overlap_area_m2"], errors="ignore")


def overlap_share_with_geom(geom, region_geom) -> float:
    if geom is None or geom.is_empty or region_geom is None or region_geom.is_empty:
        return 0.0
    area = float(getattr(geom, "area", 0.0) or 0.0)
    if area <= 0:
        return 0.0
    inter_area = float(geom.intersection(region_geom).area)
    return inter_area / area


def dedupe_instances_by_overlap(
    inst_gdf: gpd.GeoDataFrame,
    overlap_ratio_thr: float = 0.6,
) -> gpd.GeoDataFrame:
    if inst_gdf.empty:
        return inst_gdf.copy()

    ordered = inst_gdf.copy()
    ordered["_orig_idx"] = range(len(ordered))
    ordered["_area_m2"] = ordered.geometry.area.astype(float)
    ordered = ordered.sort_values("_area_m2", ascending=False).reset_index(drop=True)
    sindex = ordered.sindex
    keep = [True] * len(ordered)

    for i, geom in enumerate(ordered.geometry):
        if not keep[i] or geom is None or geom.is_empty:
            continue
        for j in sindex.intersection(geom.bounds):
            if j <= i or not keep[j]:
                continue
            other = ordered.geometry.iloc[j]
            if other is None or other.is_empty or not geom.intersects(other):
                continue
            inter_area = float(geom.intersection(other).area)
            if inter_area <= 0:
                continue
            denom = max(min(float(ordered["_area_m2"].iloc[i]), float(ordered["_area_m2"].iloc[j])), 1e-6)
            if inter_area / denom >= overlap_ratio_thr:
                keep[j] = False

    deduped = ordered[pd.Series(keep, index=ordered.index)].copy()
    deduped = deduped.sort_values("_orig_idx").drop(columns=["_orig_idx", "_area_m2"], errors="ignore")
    return gpd.GeoDataFrame(deduped, geometry="geometry", crs=inst_gdf.crs)


def suppress_small_boundary_fragments(
    inst_gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    boundary_band_m: float = 1.5,
    min_area_m2: float = 6.0,
) -> gpd.GeoDataFrame:
    if inst_gdf.empty or polygon_gdf.empty or boundary_band_m <= 0 or min_area_m2 <= 0:
        return inst_gdf.copy()

    inst = inst_gdf.to_crs(polygon_gdf.crs).copy()
    boundaries = polygon_gdf.boundary
    boundary_band = boundaries.buffer(boundary_band_m)
    try:
        band_geom = boundary_band.union_all()
    except Exception:
        band_geom = boundary_band.unary_union

    areas = inst.geometry.area.astype(float)
    near_boundary = inst.geometry.intersects(band_geom)
    small = areas < float(min_area_m2)
    kept = inst[~(near_boundary & small)].copy()
    return gpd.GeoDataFrame(kept, geometry="geometry", crs=polygon_gdf.crs)
