from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio


DEFAULT_NODATA_FLOAT = -9999.0


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_dem(dem_path: str | Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return dem, profile


def _resolve_nodata(dem: np.ndarray, nodata: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    dem_nan = dem.astype(np.float32).copy()

    if nodata is not None:
        invalid = np.isclose(dem_nan, nodata)
    else:
        invalid = np.zeros_like(dem_nan, dtype=bool)

    invalid |= ~np.isfinite(dem_nan)
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

    # 北=0, 东=90, 顺时针
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.mod(aspect_deg + 360.0, 360.0).astype(np.float32)

    flat_mask = slope_deg < 1e-6

    slope_deg[~valid_mask] = out_nodata
    aspect_deg[~valid_mask] = out_nodata
    aspect_deg[flat_mask] = out_nodata

    return slope_deg, aspect_deg


def write_single_band_float(
    out_path: str | Path,
    arr: np.ndarray,
    profile: dict,
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


def generate_terrain_products(
    dem_tif: str | Path,
    slope_tif: str | Path,
    aspect_tif: str | Path,
    z_factor: float = 1.0,
    out_nodata: float = DEFAULT_NODATA_FLOAT,
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
        "z_factor": z_factor,
        "out_nodata": out_nodata,
        "shape": [int(profile["height"]), int(profile["width"])],
        "crs": str(profile.get("crs")),
        "transform": str(profile.get("transform")),
    }
    return summary


def auto_derive_output_paths(
    dem_tif: str | Path,
    out_dir: Optional[str | Path] = None,
):
    dem_tif = Path(dem_tif)
    if out_dir is None:
        out_dir = dem_tif.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = dem_tif.stem
    slope_tif = out_dir / f"{stem}_slope.tif"
    aspect_tif = out_dir / f"{stem}_aspect.tif"
    return slope_tif, aspect_tif


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate slope/aspect rasters from DEM.")
    parser.add_argument("--dem_tif", required=True)
    parser.add_argument("--slope_tif", default=None)
    parser.add_argument("--aspect_tif", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--z_factor", type=float, default=1.0)
    parser.add_argument("--summary_json", default=None)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.slope_tif is None or args.aspect_tif is None:
        auto_slope, auto_aspect = auto_derive_output_paths(args.dem_tif, args.out_dir)
        slope_tif = args.slope_tif or str(auto_slope)
        aspect_tif = args.aspect_tif or str(auto_aspect)
    else:
        slope_tif = args.slope_tif
        aspect_tif = args.aspect_tif

    summary = generate_terrain_products(
        dem_tif=args.dem_tif,
        slope_tif=slope_tif,
        aspect_tif=aspect_tif,
        z_factor=args.z_factor,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        ensure_parent(args.summary_json)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()