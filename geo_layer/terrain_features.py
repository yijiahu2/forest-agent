# geo_layer/terrain_features.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.windows import Window

from geo_layer.terrain_constraints import (
    TerrainRuleConfig,
    summarize_terrain_classes,
    encode_class_to_int,
    LANDFORM_CODE,
    SLOPE_POSITION_CODE,
)

DEFAULT_NODATA_FLOAT = -9999.0
DEFAULT_NODATA_INT = 0


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_dem(dem_tif: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(dem_tif) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def _resolve_nodata(dem: np.ndarray, nodata: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    dem_nan = dem.astype(np.float32).copy()
    invalid = ~np.isfinite(dem_nan)
    if nodata is not None:
        invalid |= np.isclose(dem_nan, float(nodata))
    dem_nan[invalid] = np.nan
    valid_mask = ~np.isnan(dem_nan)
    return dem_nan, valid_mask


def _fill_nan_for_gradient(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if np.isnan(out).any():
        med = np.nanmedian(out)
        if not np.isfinite(med):
            med = 0.0
        out[np.isnan(out)] = med
    return out


def compute_slope_aspect(
    dem: np.ndarray,
    transform,
    nodata: Optional[float] = None,
    z_factor: float = 1.0,
    out_nodata: float = DEFAULT_NODATA_FLOAT,
) -> Tuple[np.ndarray, np.ndarray]:
    dem_nan, valid_mask = _resolve_nodata(dem, nodata)
    dem_filled = _fill_nan_for_gradient(dem_nan)

    xres = abs(transform.a)
    yres = abs(transform.e)
    if xres <= 0 or yres <= 0:
        raise ValueError(f"Invalid raster resolution: xres={xres}, yres={yres}")

    dem_filled = dem_filled * float(z_factor)
    dz_dy, dz_dx = np.gradient(dem_filled, yres, xres)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)
    aspect_deg = np.where(aspect_deg >= 360.0, aspect_deg - 360.0, aspect_deg).astype(np.float32)

    slope_deg[~valid_mask] = out_nodata
    aspect_deg[~valid_mask] = out_nodata
    return slope_deg, aspect_deg


def compute_tpi_like(dem: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    dem_nan, valid_mask = _resolve_nodata(dem, nodata)
    dem_filled = _fill_nan_for_gradient(dem_nan)

    up = np.roll(dem_filled, -1, axis=0)
    down = np.roll(dem_filled, 1, axis=0)
    left = np.roll(dem_filled, 1, axis=1)
    right = np.roll(dem_filled, -1, axis=1)
    ul = np.roll(up, 1, axis=1)
    ur = np.roll(up, -1, axis=1)
    dl = np.roll(down, 1, axis=1)
    dr = np.roll(down, -1, axis=1)

    neigh_mean = (up + down + left + right + ul + ur + dl + dr) / 8.0
    tpi = dem_filled - neigh_mean
    tpi = tpi.astype(np.float32)
    tpi[~valid_mask] = np.nan
    return tpi


def normalize_relative_position(dem: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    dem_nan, valid_mask = _resolve_nodata(dem, nodata)
    out = np.full_like(dem_nan, np.nan, dtype=np.float32)
    vals = dem_nan[valid_mask]
    if len(vals) == 0:
        return out
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        out[valid_mask] = 0.5
        return out
    out[valid_mask] = ((dem_nan[valid_mask] - vmin) / (vmax - vmin)).astype(np.float32)
    return out


def classify_landform_and_slope_position_rasters(
    dem: np.ndarray,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    nodata: Optional[float] = None,
    rule_cfg: Optional[TerrainRuleConfig] = None,
    out_nodata: int = DEFAULT_NODATA_INT,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = rule_cfg or TerrainRuleConfig()
    dem_nan, valid_mask = _resolve_nodata(dem, nodata)

    rel = normalize_relative_position(dem, nodata=nodata)
    tpi = compute_tpi_like(dem, nodata=nodata)

    vals = dem_nan[valid_mask]
    relief = float(np.nanmax(vals) - np.nanmin(vals)) if len(vals) > 0 else None

    landform_arr = np.full(dem.shape, out_nodata, dtype=np.uint8)
    slope_pos_arr = np.full(dem.shape, out_nodata, dtype=np.uint8)

    it = np.ndindex(dem.shape)
    for i, j in it:
        if not valid_mask[i, j]:
            continue

        summary = summarize_terrain_classes(
            elevation_mean_m=float(dem_nan[i, j]),
            relief_10km_m=relief,
            slope_mean_deg=float(slope_deg[i, j]),
            aspect_mean_deg=float(aspect_deg[i, j]),
            relative_elevation_norm=float(rel[i, j]) if np.isfinite(rel[i, j]) else None,
            tpi_local=float(tpi[i, j]) if np.isfinite(tpi[i, j]) else None,
            flow_accumulation_proxy=None,
            rule_cfg=cfg,
        )

        landform_arr[i, j] = encode_class_to_int(summary["landform_type"], LANDFORM_CODE, nodata=out_nodata)
        slope_pos_arr[i, j] = encode_class_to_int(summary["slope_position_class"], SLOPE_POSITION_CODE, nodata=out_nodata)

    return landform_arr, slope_pos_arr


def write_single_band_float(
    out_path: str | Path,
    arr: np.ndarray,
    profile: Dict[str, Any],
    nodata: float = DEFAULT_NODATA_FLOAT,
    compress: str = "lzw",
) -> None:
    ensure_parent(out_path)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=float(nodata),
        compress=compress,
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def write_single_band_uint8(
    out_path: str | Path,
    arr: np.ndarray,
    profile: Dict[str, Any],
    nodata: int = DEFAULT_NODATA_INT,
    compress: str = "lzw",
) -> None:
    ensure_parent(out_path)
    out_profile = profile.copy()
    out_profile.update(
        dtype="uint8",
        count=1,
        nodata=int(nodata),
        compress=compress,
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


def generate_terrain_products(
    dem_tif: str | Path,
    slope_tif: str | Path,
    aspect_tif: str | Path,
    z_factor: float = 1.0,
    out_nodata: float = DEFAULT_NODATA_FLOAT,
    landform_tif: Optional[str | Path] = None,
    slope_position_tif: Optional[str | Path] = None,
    terrain_summary_json: Optional[str | Path] = None,
    flat_slope_threshold_deg: float = 5.0,
    plain_relief_threshold_m: float = 30.0,
) -> dict:
    dem, profile = load_dem(dem_tif)
    nodata = profile.get("nodata", None)

    slope_deg, aspect_deg = compute_slope_aspect(
        dem=dem,
        transform=profile["transform"],
        nodata=nodata,
        z_factor=z_factor,
        out_nodata=out_nodata,
    )

    write_single_band_float(slope_tif, slope_deg, profile, nodata=out_nodata)
    write_single_band_float(aspect_tif, aspect_deg, profile, nodata=out_nodata)

    summary = {
        "dem_tif": str(dem_tif),
        "slope_tif": str(slope_tif),
        "aspect_tif": str(aspect_tif),
        "landform_tif": None,
        "slope_position_tif": None,
        "z_factor": z_factor,
        "out_nodata": out_nodata,
        "flat_slope_threshold_deg": flat_slope_threshold_deg,
        "plain_relief_threshold_m": plain_relief_threshold_m,
    }

    cfg = TerrainRuleConfig(
        flat_slope_threshold_deg=flat_slope_threshold_deg,
        plain_relief_threshold_m=plain_relief_threshold_m,
    )

    if landform_tif or slope_position_tif:
        landform_arr, slope_pos_arr = classify_landform_and_slope_position_rasters(
            dem=dem,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            nodata=nodata,
            rule_cfg=cfg,
        )
        if landform_tif:
            write_single_band_uint8(landform_tif, landform_arr, profile, nodata=DEFAULT_NODATA_INT)
            summary["landform_tif"] = str(landform_tif)
        if slope_position_tif:
            write_single_band_uint8(slope_position_tif, slope_pos_arr, profile, nodata=DEFAULT_NODATA_INT)
            summary["slope_position_tif"] = str(slope_position_tif)

    if terrain_summary_json:
        ensure_parent(terrain_summary_json)
        Path(terrain_summary_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate terrain products from DEM.")
    parser.add_argument("--dem_tif", required=True)
    parser.add_argument("--slope_tif", required=True)
    parser.add_argument("--aspect_tif", required=True)
    parser.add_argument("--landform_tif", default=None)
    parser.add_argument("--slope_position_tif", default=None)
    parser.add_argument("--terrain_summary_json", default=None)
    parser.add_argument("--z_factor", type=float, default=1.0)
    parser.add_argument("--out_nodata", type=float, default=DEFAULT_NODATA_FLOAT)
    parser.add_argument("--flat_slope_threshold_deg", type=float, default=5.0)
    parser.add_argument("--plain_relief_threshold_m", type=float, default=30.0)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    summary = generate_terrain_products(
        dem_tif=args.dem_tif,
        slope_tif=args.slope_tif,
        aspect_tif=args.aspect_tif,
        landform_tif=args.landform_tif,
        slope_position_tif=args.slope_position_tif,
        terrain_summary_json=args.terrain_summary_json,
        z_factor=args.z_factor,
        out_nodata=args.out_nodata,
        flat_slope_threshold_deg=args.flat_slope_threshold_deg,
        plain_relief_threshold_m=args.plain_relief_threshold_m,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()