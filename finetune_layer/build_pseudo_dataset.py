from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import WindowError
from rasterio.features import rasterize
from rasterio.mask import mask
from shapely.geometry import box

from finetune_layer.io_utils import dump_csv, dump_json, load_csv, load_yaml


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        return str(v)
    except Exception:
        return None


def _get_raster_bounds_polygon(src: rasterio.io.DatasetReader):
    left, bottom, right, top = src.bounds
    return box(left, bottom, right, top)


def _ensure_geom_crs_matches_raster(geom, geom_crs, raster_crs):
    if geom is None or geom.is_empty:
        return geom
    if geom_crs is None or raster_crs is None or str(geom_crs) == str(raster_crs):
        return geom
    gseries = gpd.GeoSeries([geom], crs=geom_crs).to_crs(raster_crs)
    return gseries.iloc[0]


def _get_input_image_bounds_in_shp_crs(input_image: str, shp_crs):
    with rasterio.open(input_image) as src:
        raster_poly = _get_raster_bounds_polygon(src)
        raster_crs = src.crs
    if shp_crs is None or raster_crs is None or str(shp_crs) == str(raster_crs):
        return raster_poly
    return gpd.GeoSeries([raster_poly], crs=raster_crs).to_crs(shp_crs).iloc[0]


def _crop_raster_with_geom(
    src_path: str,
    geom,
    geom_crs,
    out_path: Path | None = None,
    force_singleband_uint8: bool = False,
):
    try:
        with rasterio.open(src_path) as src:
            geom2 = _ensure_geom_crs_matches_raster(geom, geom_crs, src.crs)
            if geom2 is None or geom2.is_empty:
                return False, None, None, "empty_geometry"

            raster_poly = _get_raster_bounds_polygon(src)
            if not geom2.intersects(raster_poly):
                return False, None, None, "no_overlap"

            try:
                out_img, out_transform = mask(src, [geom2], crop=True)
            except (ValueError, WindowError):
                return False, None, None, "no_overlap_after_mask"

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                }
            )

            if force_singleband_uint8:
                arr = out_img[0]
                arr = (arr > 0).astype("uint8")
                out_img = arr[np.newaxis, ...]
                out_meta.update({
                    "count": 1,
                    "dtype": "uint8",
                    "nodata": 0,
                })
                for bad_key in ["photometric", "compress", "interleave"]:
                    out_meta.pop(bad_key, None)    

            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_img)

            return True, out_meta, out_img, None
    except Exception as e:
        return False, None, None, f"exception:{type(e).__name__}:{e}"


def _save_crop(src_path: str, geom, geom_crs, out_path: Path):
    return _crop_raster_with_geom(
        src_path=src_path,
        geom=geom,
        geom_crs=geom_crs,
        out_path=out_path,
        force_singleband_uint8=False,
    )


def _save_crop_mask_from_raster(src_path: str, geom, geom_crs, out_path: Path):
    ok, _, _, reason = _crop_raster_with_geom(
        src_path=src_path,
        geom=geom,
        geom_crs=geom_crs,
        out_path=out_path,
        force_singleband_uint8=True,
    )
    return ok, reason


def _save_mask_from_inst_shp(
    inst_gdf: gpd.GeoDataFrame,
    geom,
    geom_crs,
    ref_meta: dict,
    ref_raster_crs,
    out_path: Path,
):
    if inst_gdf is None or len(inst_gdf) == 0:
        return False, "empty_instance_gdf"

    geom2 = _ensure_geom_crs_matches_raster(geom, geom_crs, ref_raster_crs)
    if geom2 is None or geom2.is_empty:
        return False, "empty_geometry"

    work = inst_gdf
    if work.crs is not None and ref_raster_crs is not None and str(work.crs) != str(ref_raster_crs):
        work = work.to_crs(ref_raster_crs)

    sub = work[work.intersects(geom2)]
    if len(sub) == 0:
        return False, "no_intersecting_instances"

    height = ref_meta["height"]
    width = ref_meta["width"]
    transform = ref_meta["transform"]

    shapes = [(g, 1) for g in sub.geometry if g is not None and not g.is_empty]
    if not shapes:
        return False, "no_valid_instance_geoms"

    arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )

    meta = ref_meta.copy()

    # 关键修复：
    # 参考影像可能继承了 nodata=256（例如原始 16-bit DOM），
    # 但这里输出的是 uint8 mask，nodata 必须落在 0~255 范围内。
    meta.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
        }
    )

    # 某些参考影像 profile 里还可能带有不适合单波段 mask 的 photometric / compress 等字段，
    # 这里尽量清理掉，避免后续再出现 profile 不兼容问题。
    for bad_key in ["photometric", "compress", "interleave"]:
        meta.pop(bad_key, None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr, 1)

    return True, None


def _filter_local_xiaoban(gdf: gpd.GeoDataFrame, xbh_field: str, input_image: str, out_dir: Path):
    input_bounds_in_shp_crs = _get_input_image_bounds_in_shp_crs(input_image, gdf.crs)
    local_gdf = gdf[gdf.intersects(input_bounds_in_shp_crs)].copy()

    local_shp = out_dir / "local_xiaoban" / "xiaoban_local.shp"
    local_shp.parent.mkdir(parents=True, exist_ok=True)
    if len(local_gdf) > 0:
        local_gdf.to_file(local_shp, encoding="utf-8")

    return local_gdf, str(local_shp), input_bounds_in_shp_crs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo_csv", required=True)
    parser.add_argument("--replay_good_csv", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"]) / "pseudo_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_image = cfg["input_image"]
    dem_tif = cfg.get("dem_tif")
    slope_tif = cfg.get("slope_tif")
    aspect_tif = cfg.get("aspect_tif")
    shp = cfg["xiaoban_shp"]
    xbh_field = cfg["xiaoban_id_field"]
    pseudo_mask_tif = cfg.get("pseudo_mask_tif")
    pseudo_inst_shp = cfg.get("pseudo_inst_shp")

    pseudo_df = load_csv(args.pseudo_csv)
    replay_df = load_csv(args.replay_good_csv)
    all_df = pd.concat([pseudo_df, replay_df], ignore_index=True)

    gdf = gpd.read_file(shp)
    if xbh_field not in gdf.columns:
        raise ValueError(f"小班 shp 中未找到字段: {xbh_field}")

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)

    local_gdf, local_shp, local_bounds_geom = _filter_local_xiaoban(
        gdf=gdf,
        xbh_field=xbh_field,
        input_image=input_image,
        out_dir=out_dir,
    )

    target_gdf = local_gdf if len(local_gdf) > 0 else gdf

    inst_gdf = None
    if pseudo_inst_shp:
        inst_gdf = gpd.read_file(pseudo_inst_shp)
        inst_gdf = inst_gdf.copy()
        inst_gdf["geometry"] = inst_gdf.geometry.buffer(0)

    records = []
    skip_records = []

    available_xbh = set(target_gdf[xbh_field].astype(str).tolist())

    for _, row in all_df.iterrows():
        xbh = str(row["XBH"])
        split = str(row["split"])

        if xbh not in available_xbh:
            skip_records.append(
                {
                    "XBH": xbh,
                    "split": split,
                    "reason": "xbh_outside_local_input_extent_or_not_found",
                }
            )
            continue

        sub = target_gdf[target_gdf[xbh_field].astype(str) == xbh]
        if len(sub) == 0:
            skip_records.append({"XBH": xbh, "split": split, "reason": "xbh_not_found_in_local_shp"})
            continue

        geom = sub.iloc[0].geometry
        geom_crs = target_gdf.crs
        roi_id = f"{split}_{xbh}"

        if geom is None or geom.is_empty:
            skip_records.append({"XBH": xbh, "split": split, "reason": "empty_geometry"})
            continue

        image_path = out_dir / "images" / f"{roi_id}.tif"
        img_ok, image_meta, _, img_reason = _save_crop(input_image, geom, geom_crs, image_path)
        if not img_ok or image_meta is None:
            skip_records.append({"XBH": xbh, "split": split, "reason": f"image_crop_failed:{img_reason}"})
            continue

        dem_path = None
        slope_path = None
        aspect_path = None
        dem_ok = False
        slope_ok = False
        aspect_ok = False
        dem_reason = None
        slope_reason = None
        aspect_reason = None

        if dem_tif:
            dem_path = out_dir / "terrain" / "dem" / f"{roi_id}.tif"
            dem_ok, _, _, dem_reason = _save_crop(dem_tif, geom, geom_crs, dem_path)
            if not dem_ok:
                dem_path = None

        if slope_tif:
            slope_path = out_dir / "terrain" / "slope" / f"{roi_id}.tif"
            slope_ok, _, _, slope_reason = _save_crop(slope_tif, geom, geom_crs, slope_path)
            if not slope_ok:
                slope_path = None

        if aspect_tif:
            aspect_path = out_dir / "terrain" / "aspect" / f"{roi_id}.tif"
            aspect_ok, _, _, aspect_reason = _save_crop(aspect_tif, geom, geom_crs, aspect_path)
            if not aspect_ok:
                aspect_path = None

        mask_sem_path = out_dir / "masks_sem" / f"{roi_id}.tif"
        has_mask = False
        mask_reason = "no_mask_source"

        if pseudo_mask_tif:
            has_mask, mask_reason = _save_crop_mask_from_raster(
                pseudo_mask_tif,
                geom,
                geom_crs,
                mask_sem_path,
            )
        elif inst_gdf is not None:
            with rasterio.open(image_path) as ref_src:
                ref_meta = ref_src.meta.copy()
                ref_raster_crs = ref_src.crs
            has_mask, mask_reason = _save_mask_from_inst_shp(
                inst_gdf=inst_gdf,
                geom=geom,
                geom_crs=geom_crs,
                ref_meta=ref_meta,
                ref_raster_crs=ref_raster_crs,
                out_path=mask_sem_path,
            )

        meta = {
            "roi_id": roi_id,
            "XBH": xbh,
            "split": split,
            "tree_count_error_ratio": _safe_float(row.get("tree_count_error_ratio")),
            "mean_crown_width_error_ratio": _safe_float(row.get("mean_crown_width_error_ratio")),
            "closure_error_abs": _safe_float(row.get("closure_error_abs")),
            "density_error_abs": _safe_float(row.get("density_error_abs")),
            "mean_slope": _safe_float(row.get("mean_slope")),
            "relief_elev": _safe_float(row.get("relief_elev")),
            "dominant_aspect_class": _safe_str(row.get("dominant_aspect_class")),
            "image_path": str(image_path),
            "mask_sem_path": str(mask_sem_path),
            "has_mask": bool(has_mask),
            "mask_reason": mask_reason,
            "dem_path": None if dem_path is None else str(dem_path),
            "dem_ok": dem_ok,
            "dem_reason": dem_reason,
            "slope_path": None if slope_path is None else str(slope_path),
            "slope_ok": slope_ok,
            "slope_reason": slope_reason,
            "aspect_path": None if aspect_path is None else str(aspect_path),
            "aspect_ok": aspect_ok,
            "aspect_reason": aspect_reason,
        }

        meta_path = out_dir / "meta" / f"{roi_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        records.append(meta)

    manifest = pd.DataFrame(records)
    skip_df = pd.DataFrame(skip_records)

    dump_csv(manifest, out_dir / "manifest.csv")
    if len(skip_df) > 0:
        dump_csv(skip_df, out_dir / "skipped_samples.csv")

    summary = {
        "num_input_candidates": int(len(all_df)),
        "num_local_xiaoban": int(len(local_gdf)),
        "num_samples": int(len(records)),
        "num_samples_with_mask": int(manifest["has_mask"].sum()) if len(manifest) else 0,
        "num_skipped": int(len(skip_records)),
        "output_dir": str(out_dir),
        "local_xiaoban_shp": local_shp,
        "pseudo_mask_tif": pseudo_mask_tif,
        "pseudo_inst_shp": pseudo_inst_shp,
        "terrain_inputs": {
            "dem_tif": dem_tif,
            "slope_tif": slope_tif,
            "aspect_tif": aspect_tif,
        },
        "outputs": {
            "manifest_csv": str(out_dir / "manifest.csv"),
            "skipped_samples_csv": str(out_dir / "skipped_samples.csv") if len(skip_df) > 0 else None,
        },
        "note": "已先做 local ROI 过滤，再对局部 xiaoban 建集；适配 shp 全区、DOM/DEM 局部块 的输入方式。",
    }
    dump_json(summary, out_dir / "pseudo_dataset_summary.json")
    print(f"[OK] pseudo dataset built: {out_dir}, n={len(records)}, skipped={len(skip_records)}")


if __name__ == "__main__":
    main()