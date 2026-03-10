from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.ops import unary_union

from agent.config_builder import load_yaml, save_yaml
from geo_layer.terrain_features import generate_terrain_products


# =========================
# 基础工具
# =========================

DEFAULT_BASE_PARAMS = {
    "diam_list": "96,192,320",
    "tile": 1536,
    "overlap": 512,
    "tile_overlap": 0.35,
    "augment": True,
    "iou_merge_thr": 0.28,
    "bsize": 256,
}

SAFE_TILE = [1536, 2048]
SAFE_OVERLAP = [384, 512]
SAFE_TILE_OVERLAP = [0.25, 0.35, 0.45]
SAFE_IOU = [0.18, 0.22, 0.24, 0.28]
SAFE_DIAM = [
    "96,160,256",
    "96,192,320",
    "128,192,320",
    "128,256,320",
]


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_parent(Path(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_str(v, default=None):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return str(v)


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_BASE_PARAMS)
    if params:
        out.update(params)

    if out.get("tile") not in SAFE_TILE:
        out["tile"] = DEFAULT_BASE_PARAMS["tile"]

    if out.get("overlap") not in SAFE_OVERLAP:
        out["overlap"] = DEFAULT_BASE_PARAMS["overlap"]

    if out.get("tile_overlap") not in SAFE_TILE_OVERLAP:
        out["tile_overlap"] = DEFAULT_BASE_PARAMS["tile_overlap"]

    if out.get("iou_merge_thr") not in SAFE_IOU:
        out["iou_merge_thr"] = DEFAULT_BASE_PARAMS["iou_merge_thr"]

    if out.get("diam_list") not in SAFE_DIAM:
        out["diam_list"] = DEFAULT_BASE_PARAMS["diam_list"]

    out["augment"] = bool(out.get("augment", True))

    # 关键运行约束：强制固定
    out["bsize"] = 256
    return out


def copy_vector_dataset(src_shp: str, dst_shp: str):
    src = Path(src_shp)
    dst = Path(dst_shp)
    ensure_parent(dst)

    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        s = src.with_suffix(ext)
        if s.exists():
            shutil.copy2(s, dst.with_suffix(ext))


# =========================
# terrain 准备 / ROI terrain 裁剪
# =========================

def prepare_terrain_rasters(
    dem_tif: Optional[str],
    slope_tif: Optional[str],
    aspect_tif: Optional[str],
    work_dir: str,
) -> Dict[str, Any]:
    """
    若只提供 dem_tif，则自动生成 slope/aspect。
    """
    result = {
        "dem_tif": dem_tif,
        "slope_tif": slope_tif,
        "aspect_tif": aspect_tif,
        "terrain_generated": False,
    }

    if dem_tif is None:
        return result

    if slope_tif is not None and aspect_tif is not None:
        return result

    terrain_dir = Path(work_dir) / "terrain_cache"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    auto_slope = terrain_dir / f"{Path(dem_tif).stem}_slope.tif"
    auto_aspect = terrain_dir / f"{Path(dem_tif).stem}_aspect.tif"

    generate_terrain_products(
        dem_tif=dem_tif,
        slope_tif=str(auto_slope),
        aspect_tif=str(auto_aspect),
        z_factor=1.0,
    )

    result["slope_tif"] = str(auto_slope)
    result["aspect_tif"] = str(auto_aspect)
    result["terrain_generated"] = True
    return result


def crop_raster_to_geometry(
    src_raster: str,
    geom_gdf: gpd.GeoDataFrame,
    out_raster: str,
    all_touched: bool = False,
):
    with rasterio.open(src_raster) as src:
        geom_in_src_crs = geom_gdf.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in geom_in_src_crs.geometry if g is not None and not g.is_empty]

        if not geoms:
            raise ValueError(f"No valid geometry found when cropping raster: {src_raster}")

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

        out_path = Path(out_raster)
        ensure_parent(out_path)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)


def clip_vector_to_geometry(src_vector: str, geom_gdf: gpd.GeoDataFrame, out_vector: str):
    gdf = gpd.read_file(src_vector)
    if gdf.crs is None:
        raise ValueError(f"Vector has no CRS: {src_vector}")

    geom = geom_gdf.to_crs(gdf.crs)
    clipped = gpd.overlay(gdf, geom, how="intersection")
    clipped = clipped[clipped.geometry.notnull() & (~clipped.geometry.is_empty)].copy()

    if clipped.empty:
        raise ValueError(f"Clipped vector is empty: {src_vector}")

    out_path = Path(out_vector)
    ensure_parent(out_path)
    clipped.to_file(out_path)


def crop_roi_terrain_bundle(
    roi_geom_gdf: gpd.GeoDataFrame,
    roi_dir: str,
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    roi_dir = Path(roi_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "roi_dem_tif": None,
        "roi_slope_tif": None,
        "roi_aspect_tif": None,
    }

    if dem_tif:
        roi_dem_tif = roi_dir / "roi_dem.tif"
        crop_raster_to_geometry(dem_tif, roi_geom_gdf, str(roi_dem_tif))
        out["roi_dem_tif"] = str(roi_dem_tif)

    if slope_tif:
        roi_slope_tif = roi_dir / "roi_slope.tif"
        crop_raster_to_geometry(slope_tif, roi_geom_gdf, str(roi_slope_tif))
        out["roi_slope_tif"] = str(roi_slope_tif)

    if aspect_tif:
        roi_aspect_tif = roi_dir / "roi_aspect.tif"
        crop_raster_to_geometry(aspect_tif, roi_geom_gdf, str(roi_aspect_tif))
        out["roi_aspect_tif"] = str(roi_aspect_tif)

    return out


# =========================
# bad xiaoban 选择
# =========================

def _build_error_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["tree_count_error_abs"].fillna(0) * 1.0
        + df["mean_crown_width_error_abs"].fillna(0) * 50.0
        + df["closure_error_abs"].fillna(0) * 100.0
    )


def select_bad_xiaoban_rows(
    details_csv: str,
    tree_count_err_thr: float = 80.0,
    crown_err_thr: float = 0.40,
    closure_err_thr: float = 0.15,
    top_k: int = 3,
) -> pd.DataFrame:
    df = pd.read_csv(details_csv)

    if df.empty:
        return df

    required = [
        "xiaoban_id",
        "tree_count_error_abs",
        "mean_crown_width_error_abs",
        "closure_error_abs",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"details.csv missing required column: {col}")

    df["xiaoban_id"] = df["xiaoban_id"].astype(str)

    cond = (
        (df["tree_count_error_abs"].fillna(0) >= tree_count_err_thr)
        | (df["mean_crown_width_error_abs"].fillna(0) >= crown_err_thr)
        | (df["closure_error_abs"].fillna(0) >= closure_err_thr)
    )

    bad = df[cond].copy()

    if bad.empty:
        tmp = df.copy()
        tmp["error_score"] = _build_error_score(tmp)
        bad = tmp.sort_values("error_score", ascending=False).head(top_k)
    else:
        bad["error_score"] = _build_error_score(bad)
        bad = bad.sort_values("error_score", ascending=False).head(top_k)

    return bad.reset_index(drop=True)


# =========================
# ROI
# =========================

def make_bad_roi_gdf(
    xiaoban_shp: str,
    xiaoban_id_field: str,
    bad_ids: List[str],
    buffer_m: float = 5.0,
) -> gpd.GeoDataFrame:
    xgdf = gpd.read_file(xiaoban_shp)
    if xgdf.crs is None:
        raise ValueError("xiaoban shapefile has no CRS.")

    xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
    bad = xgdf[xgdf[xiaoban_id_field].isin([str(x) for x in bad_ids])].copy()

    if bad.empty:
        raise ValueError(f"No bad xiaoban found in shp. bad_ids={bad_ids}")

    # buffer 必须在投影坐标系下
    if not bad.crs.is_projected:
        bad = bad.to_crs(bad.estimate_utm_crs())

    roi_union = unary_union(bad.geometry.tolist())
    roi_buffered = gpd.GeoDataFrame(
        {"roi_id": ["bad_roi"]},
        geometry=[roi_union.buffer(buffer_m)],
        crs=bad.crs,
    )
    return roi_buffered


# =========================
# config 构建
# =========================

def build_local_refine_config(
    base_config_path: str,
    out_config_path: str,
    local_input_image: str,
    local_output_dir: str,
    local_xiaoban_shp: str,
    params: Dict[str, Any],
    run_name: str,
) -> Dict[str, Any]:
    cfg = load_yaml(base_config_path)

    cfg["run_name"] = run_name
    cfg["input_image"] = local_input_image
    cfg["output_dir"] = local_output_dir
    cfg["xiaoban_shp"] = local_xiaoban_shp

    cfg["metrics_json"] = f"/home/xth/forest_agent_project/outputs/{run_name}/metrics.json"
    cfg["details_csv"] = f"/home/xth/forest_agent_project/outputs/{run_name}/details.csv"

    params = sanitize_params(params)
    for k, v in params.items():
        cfg[k] = v

    save_yaml(cfg, out_config_path)
    return cfg


# =========================
# merge
# =========================

def merge_global_and_local_instances(
    global_inst_shp: str,
    local_inst_shp: str,
    xiaoban_shp: str,
    xiaoban_id_field: str,
    bad_ids: List[str],
    out_merged_shp: str,
):
    global_gdf = gpd.read_file(global_inst_shp)
    local_gdf = gpd.read_file(local_inst_shp)
    xgdf = gpd.read_file(xiaoban_shp)

    if global_gdf.crs is None or local_gdf.crs is None or xgdf.crs is None:
        raise ValueError("One of shapefiles has no CRS.")

    xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
    bad = xgdf[xgdf[xiaoban_id_field].isin([str(x) for x in bad_ids])].copy()
    if bad.empty:
        raise ValueError("No bad xiaoban found for merge.")

    bad = bad.to_crs(global_gdf.crs)
    local_gdf = local_gdf.to_crs(global_gdf.crs)

    bad_union = unary_union(bad.geometry.tolist())

    # 删掉全图中与 bad 小班相交的旧实例，用局部新实例替换
    global_keep = global_gdf[~global_gdf.geometry.intersects(bad_union)].copy()

    merged = pd.concat([global_keep, local_gdf], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=global_gdf.crs)

    out_path = Path(out_merged_shp)
    ensure_parent(out_path)
    merged.to_file(out_path)


# =========================
# merged 评测 + 前后对比
# =========================

def evaluate_merged_result(
    base_cfg: Dict[str, Any],
    local_root: Path,
    merged_shp: str,
    bad_ids: List[str],
    group_plan: List[Dict[str, Any]],
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
) -> Dict[str, Any]:
    merged_metrics_json = str(local_root / "merged_metrics.json")
    merged_details_csv = str(local_root / "merged_details.csv")

    cmd = [
        "python",
        "/home/xth/forest_agent_project/scripts/evaluate_xiaoban_consistency.py",
        "--inst_shp", merged_shp,
        "--patch_raster", base_cfg["input_image"],
        "--xiaoban_shp", base_cfg["xiaoban_shp"],
        "--out_json", merged_metrics_json,
        "--out_csv", merged_details_csv,
        "--id_field", base_cfg["xiaoban_id_field"],
        "--tree_count_field", base_cfg["tree_count_field"],
        "--crown_field", base_cfg["crown_field"],
        "--closure_field", base_cfg["closure_field"],
        "--area_ha_field", base_cfg["area_ha_field"],
    ]

    if base_cfg.get("density_field"):
        cmd.extend(["--density_field", base_cfg["density_field"]])

    if dem_tif:
        cmd.extend(["--dem_tif", dem_tif])
    if slope_tif:
        cmd.extend(["--slope_tif", slope_tif])
    if aspect_tif:
        cmd.extend(["--aspect_tif", aspect_tif])

    res = subprocess.run(cmd, capture_output=True, text=True)

    print("\n===== MERGED EVAL STDOUT =====\n", res.stdout)
    print("\n===== MERGED EVAL STDERR =====\n", res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"Merged evaluation failed:\n{res.stderr}")

    if not Path(merged_metrics_json).exists():
        raise FileNotFoundError(f"merged_metrics.json not found: {merged_metrics_json}")
    if not Path(merged_details_csv).exists():
        raise FileNotFoundError(f"merged_details.csv not found: {merged_details_csv}")

    before_metrics = load_json(base_cfg["metrics_json"])
    after_metrics = load_json(merged_metrics_json)

    before_details = pd.read_csv(base_cfg["details_csv"])
    after_details = pd.read_csv(merged_details_csv)

    if "xiaoban_id" in before_details.columns:
        before_details["xiaoban_id"] = before_details["xiaoban_id"].astype(str)
    if "xiaoban_id" in after_details.columns:
        after_details["xiaoban_id"] = after_details["xiaoban_id"].astype(str)

    metric_keys = [
        "tree_count_error_ratio",
        "mean_crown_width_error_ratio",
        "closure_error_abs",
        "density_error_abs",
    ]

    compare = {
        "base_metrics_json": base_cfg["metrics_json"],
        "merged_metrics_json": merged_metrics_json,
        "base_details_csv": base_cfg["details_csv"],
        "merged_details_csv": merged_details_csv,
        "bad_xiaoban_ids": [str(x) for x in bad_ids],
        "group_plan": group_plan,
        "global_before_after": {},
        "bad_xiaoban_before_after": [],
    }

    for k in metric_keys:
        bv = before_metrics.get(k)
        av = after_metrics.get(k)
        compare["global_before_after"][k] = {
            "before": bv,
            "after": av,
            "delta": (av - bv) if (bv is not None and av is not None) else None,
        }

    # terrain 关键字段一并记录
    terrain_metric_keys = [
        "patch_mean_elev",
        "patch_relief_elev",
        "patch_mean_slope",
        "patch_mean_aspect_deg",
        "patch_dominant_aspect_class",
    ]
    for k in terrain_metric_keys:
        if k in before_metrics or k in after_metrics:
            compare["global_before_after"][k] = {
                "before": before_metrics.get(k),
                "after": after_metrics.get(k),
                "delta": (
                    after_metrics.get(k) - before_metrics.get(k)
                    if isinstance(before_metrics.get(k), (int, float))
                    and isinstance(after_metrics.get(k), (int, float))
                    else None
                ),
            }

    if "xiaoban_id" in before_details.columns and "xiaoban_id" in after_details.columns:
        merged_bad = before_details.merge(
            after_details,
            on="xiaoban_id",
            suffixes=("_before", "_after"),
        )
        merged_bad = merged_bad[merged_bad["xiaoban_id"].isin([str(x) for x in bad_ids])].copy()

        for _, row in merged_bad.iterrows():
            rec = {
                "xiaoban_id": row["xiaoban_id"],
                "tree_count_error_abs_before": row.get("tree_count_error_abs_before"),
                "tree_count_error_abs_after": row.get("tree_count_error_abs_after"),
                "mean_crown_width_error_abs_before": row.get("mean_crown_width_error_abs_before"),
                "mean_crown_width_error_abs_after": row.get("mean_crown_width_error_abs_after"),
                "closure_error_abs_before": row.get("closure_error_abs_before"),
                "closure_error_abs_after": row.get("closure_error_abs_after"),
                "density_error_abs_before": row.get("density_error_abs_before"),
                "density_error_abs_after": row.get("density_error_abs_after"),
            }

            # terrain 字段保留，便于后续分析
            for col in [
                "mean_elev_before", "mean_elev_after",
                "relief_elev_before", "relief_elev_after",
                "mean_slope_before", "mean_slope_after",
                "mean_aspect_deg_before", "mean_aspect_deg_after",
                "dominant_aspect_class_before", "dominant_aspect_class_after",
            ]:
                if col in row.index:
                    rec[col] = row.get(col)

            compare["bad_xiaoban_before_after"].append(rec)

    compare_json = str(local_root / "refine_compare_summary.json")
    save_json(compare, compare_json)

    return {
        "merged_metrics_json": merged_metrics_json,
        "merged_details_csv": merged_details_csv,
        "compare_json": compare_json,
    }


# =========================
# 局部参数选择策略增强版
# =========================

def detect_error_profile(row: pd.Series) -> Dict[str, Any]:
    tree_err = float(row.get("tree_count_error_abs", 0) or 0)
    crown_err = float(row.get("mean_crown_width_error_abs", 0) or 0)
    closure_err = float(row.get("closure_error_abs", 0) or 0)
    density_err = float(row.get("density_error_abs", 0) or 0)

    score_count = tree_err * 1.0
    score_crown = crown_err * 50.0
    score_closure = closure_err * 100.0
    score_density = density_err / 100.0

    dominant = max(
        [
            ("count", score_count),
            ("crown", score_crown),
            ("closure", score_closure),
            ("density", score_density),
        ],
        key=lambda x: x[1],
    )[0]

    pred_tree_count = row.get("pred_tree_count", None)
    expected_tree_count = row.get("expected_tree_count", None)
    pred_cover_ratio = row.get("pred_cover_ratio", None)
    expected_closure = row.get("expected_closure", None)
    pred_density = row.get("pred_density_trees_per_ha", None)
    expected_density = row.get("expected_density", None)

    count_direction = "unknown"
    if pd.notna(pred_tree_count) and pd.notna(expected_tree_count):
        count_direction = "under" if float(pred_tree_count) < float(expected_tree_count) else "over"

    cover_direction = "unknown"
    if pd.notna(pred_cover_ratio) and pd.notna(expected_closure):
        cover_direction = "low" if float(pred_cover_ratio) < float(expected_closure) else "high"

    density_direction = "unknown"
    if pd.notna(pred_density) and pd.notna(expected_density):
        density_direction = "low" if float(pred_density) < float(expected_density) else "high"

    return {
        "dominant_error": dominant,
        "count_direction": count_direction,
        "cover_direction": cover_direction,
        "density_direction": density_direction,
        "score_count": score_count,
        "score_crown": score_crown,
        "score_closure": score_closure,
        "score_density": score_density,
    }


def detect_terrain_profile(row: pd.Series) -> Dict[str, Any]:
    mean_slope = safe_float(row.get("mean_slope"), None)
    relief_elev = safe_float(row.get("relief_elev"), None)
    dominant_aspect = safe_str(row.get("dominant_aspect_class"), None)

    if mean_slope is None:
        slope_class = "unknown"
    elif mean_slope >= 25:
        slope_class = "steep"
    elif mean_slope >= 12:
        slope_class = "moderate"
    else:
        slope_class = "gentle"

    if relief_elev is None:
        relief_class = "unknown"
    elif relief_elev >= 20:
        relief_class = "high_relief"
    elif relief_elev >= 8:
        relief_class = "mid_relief"
    else:
        relief_class = "low_relief"

    return {
        "mean_slope": mean_slope,
        "relief_elev": relief_elev,
        "dominant_aspect_class": dominant_aspect,
        "slope_class": slope_class,
        "relief_class": relief_class,
    }


def choose_local_params_for_one_xiaoban(
    row: pd.Series,
    base_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    规则驱动增强版：
    - 先看误差主导类型
    - 再用 terrain 做微调
    - 坡度优先用于决策；坡向当前主要用于解释和分组元数据保留
    """
    base = sanitize_params(base_params)
    err_profile = detect_error_profile(row)
    terrain_profile = detect_terrain_profile(row)

    strategy = "balanced"
    params = dict(base)

    dominant = err_profile["dominant_error"]
    count_direction = err_profile["count_direction"]
    cover_direction = err_profile["cover_direction"]
    density_direction = err_profile["density_direction"]

    slope_class = terrain_profile["slope_class"]

    # 1) 数量不足：偏补漏检
    if dominant in ("count", "density") and (count_direction == "under" or density_direction == "low"):
        if slope_class == "steep":
            strategy = "steep_count_under"
            params.update({
                "diam_list": "96,192,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.45,
                "augment": True,
                "iou_merge_thr": 0.24,
            })
        elif slope_class == "moderate":
            strategy = "moderate_count_under"
            params.update({
                "diam_list": "96,192,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.35,
                "augment": True,
                "iou_merge_thr": 0.28,
            })
        else:
            strategy = "gentle_count_under"
            params.update({
                "diam_list": "96,160,256",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.35,
                "augment": True,
                "iou_merge_thr": 0.28,
            })

    # 2) 数量过多：偏抑制过分裂/过检
    elif dominant in ("count", "density") and (count_direction == "over" or density_direction == "high"):
        if slope_class == "steep":
            strategy = "steep_count_over"
            params.update({
                "diam_list": "128,256,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.35,
                "augment": True,
                "iou_merge_thr": 0.28,
            })
        else:
            strategy = "count_over"
            params.update({
                "diam_list": "128,256,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.35,
                "augment": True,
                "iou_merge_thr": 0.28,
            })

    # 3) 冠幅主导：更偏边界/冠层恢复
    elif dominant == "crown":
        if slope_class == "steep":
            strategy = "steep_crown_focus"
            params.update({
                "diam_list": "128,256,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.45,
                "augment": True,
                "iou_merge_thr": 0.24,
            })
        else:
            strategy = "crown_focus"
            params.update({
                "diam_list": "128,256,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.45,
                "augment": True,
                "iou_merge_thr": 0.24,
            })

    # 4) 覆盖/郁闭主导：优先覆盖恢复
    elif dominant == "closure":
        if cover_direction == "low":
            if slope_class == "steep":
                strategy = "steep_closure_low"
                params.update({
                    "diam_list": "96,192,320",
                    "tile": 1536,
                    "overlap": 512,
                    "tile_overlap": 0.45,
                    "augment": True,
                    "iou_merge_thr": 0.24,
                })
            else:
                strategy = "closure_low"
                params.update({
                    "diam_list": "96,192,320",
                    "tile": 1536,
                    "overlap": 512,
                    "tile_overlap": 0.45,
                    "augment": True,
                    "iou_merge_thr": 0.24,
                })
        else:
            strategy = "closure_high"
            params.update({
                "diam_list": "128,256,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.35,
                "augment": True,
                "iou_merge_thr": 0.28,
            })

    # 5) 默认折中
    else:
        if slope_class == "steep":
            strategy = "steep_balanced"
            params.update({
                "diam_list": "96,192,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.45,
                "augment": True,
                "iou_merge_thr": 0.24,
            })
        else:
            strategy = "balanced"
            params.update({
                "diam_list": "96,192,320",
                "tile": 1536,
                "overlap": 512,
                "tile_overlap": 0.35,
                "augment": True,
                "iou_merge_thr": 0.28,
            })

    params = sanitize_params(params)

    profile = {
        "error_profile": err_profile,
        "terrain_profile": terrain_profile,
    }
    return strategy, params, profile


def build_group_plan(
    bad_df: pd.DataFrame,
    base_params: Dict[str, Any],
    strategy_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """
    strategy_mode:
    - single_params: 所有 bad xiaoban 共用一组 params
    - auto: 每个 xiaoban 先判型，再按 strategy+params 分组
    """
    base_params = sanitize_params(base_params)

    if bad_df.empty:
        return []

    if strategy_mode == "single_params":
        return [{
            "strategy": "single_params",
            "params": base_params,
            "xiaoban_ids": bad_df["xiaoban_id"].astype(str).tolist(),
            "members": bad_df[["xiaoban_id"]].to_dict(orient="records"),
        }]

    plan_map = {}
    for _, row in bad_df.iterrows():
        strategy, params, profile = choose_local_params_for_one_xiaoban(row, base_params)
        key = (
            strategy,
            params["diam_list"],
            params["tile"],
            params["overlap"],
            params["tile_overlap"],
            params["augment"],
            params["iou_merge_thr"],
            params["bsize"],
        )

        if key not in plan_map:
            plan_map[key] = {
                "strategy": strategy,
                "params": params,
                "xiaoban_ids": [],
                "members": [],
            }

        xid = str(row["xiaoban_id"])
        plan_map[key]["xiaoban_ids"].append(xid)

        member = row.to_dict()
        member["xiaoban_id"] = xid
        member["profile"] = profile
        plan_map[key]["members"].append(member)

    groups = list(plan_map.values())
    groups.sort(key=lambda g: len(g["xiaoban_ids"]), reverse=True)
    return groups


# =========================
# 单个 group 执行
# =========================

def run_one_group_refinement(
    base_config_path: str,
    base_cfg: Dict[str, Any],
    current_global_shp: str,
    group_idx: int,
    group: Dict[str, Any],
    xiaoban_id_field: str,
    buffer_m: float,
    local_root: Path,
    terrain_info: Dict[str, Any],
) -> Dict[str, Any]:
    xiaoban_ids = [str(x) for x in group["xiaoban_ids"]]
    strategy = group["strategy"]
    params = sanitize_params(group["params"])

    group_name = f"group_{group_idx:02d}_{strategy}_{'_'.join(xiaoban_ids)}"
    group_root = local_root / group_name
    ensure_dir(group_root)

    roi_gdf = make_bad_roi_gdf(
        xiaoban_shp=base_cfg["xiaoban_shp"],
        xiaoban_id_field=xiaoban_id_field,
        bad_ids=xiaoban_ids,
        buffer_m=buffer_m,
    )

    local_image = str(group_root / "roi_image.tif")
    local_xiaoban = str(group_root / "roi_xiaoban.shp")
    local_config = str(Path("/home/xth/forest_agent_project/configs/generated") / f"{group_name}.yaml")
    local_output_dir = str(group_root / "seg_output")

    crop_raster_to_geometry(base_cfg["input_image"], roi_gdf, local_image)
    clip_vector_to_geometry(base_cfg["xiaoban_shp"], roi_gdf, local_xiaoban)

    terrain_roi_outputs = crop_roi_terrain_bundle(
        roi_geom_gdf=roi_gdf,
        roi_dir=str(group_root),
        dem_tif=terrain_info.get("dem_tif"),
        slope_tif=terrain_info.get("slope_tif"),
        aspect_tif=terrain_info.get("aspect_tif"),
    )

    cfg = build_local_refine_config(
        base_config_path=base_config_path,
        out_config_path=local_config,
        local_input_image=local_image,
        local_output_dir=local_output_dir,
        local_xiaoban_shp=local_xiaoban,
        params=params,
        run_name=group_name,
    )

    cmd = [
        "python",
        "/home/xth/forest_agent_project/scripts/run_zstreeseg_experiment.py",
        "--config",
        local_config,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    print(f"\n===== LOCAL REFINE STDOUT ({group_name}) =====\n", res.stdout)
    print(f"\n===== LOCAL REFINE STDERR ({group_name}) =====\n", res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"Local refine failed [{group_name}]:\n{res.stderr}")

    local_inst_shp = str(Path(local_output_dir) / "Y_inst.shp")
    if not Path(local_inst_shp).exists():
        raise FileNotFoundError(f"Local Y_inst.shp not found: {local_inst_shp}")

    merged_shp = str(group_root / "merged_after_group.shp")
    merge_global_and_local_instances(
        global_inst_shp=current_global_shp,
        local_inst_shp=local_inst_shp,
        xiaoban_shp=base_cfg["xiaoban_shp"],
        xiaoban_id_field=xiaoban_id_field,
        bad_ids=xiaoban_ids,
        out_merged_shp=merged_shp,
    )

    group_summary = {
        "group_name": group_name,
        "group_root": str(group_root),
        "strategy": strategy,
        "params": params,
        "xiaoban_ids": xiaoban_ids,
        "local_config": local_config,
        "local_output_dir": local_output_dir,
        "local_metrics_json": cfg["metrics_json"],
        "local_details_csv": cfg["details_csv"],
        "local_inst_shp": local_inst_shp,
        "merged_after_group_shp": merged_shp,
        "roi_image_tif": local_image,
        "roi_xiaoban_shp": local_xiaoban,
        "roi_dem_tif": terrain_roi_outputs["roi_dem_tif"],
        "roi_slope_tif": terrain_roi_outputs["roi_slope_tif"],
        "roi_aspect_tif": terrain_roi_outputs["roi_aspect_tif"],
        "members": group.get("members", []),
    }

    save_json(group_summary, str(group_root / "group_summary.json"))
    return group_summary


# =========================
# 主流程
# =========================

def run_local_refinement(
    base_config_path: str,
    global_details_csv: str,
    global_inst_shp: str,
    best_params: Dict[str, Any],
    xiaoban_id_field: str = "XBH",
    top_k: int = 2,
    buffer_m: float = 5.0,
    strategy_mode: str = "auto",
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
):
    base_cfg = load_yaml(base_config_path)
    base_params = sanitize_params(best_params)

    bad_df = select_bad_xiaoban_rows(global_details_csv, top_k=top_k)
    if bad_df.empty:
        raise ValueError("No xiaoban rows selected from details.csv")

    bad_ids = bad_df["xiaoban_id"].astype(str).tolist()
    print(f"[local_refine] bad_xiaoban_ids = {bad_ids}")

    group_plan = build_group_plan(
        bad_df=bad_df,
        base_params=base_params,
        strategy_mode=strategy_mode,
    )

    print("[local_refine] group_plan:")
    for i, g in enumerate(group_plan, 1):
        print(f"  - group {i}: strategy={g['strategy']}, xiaoban_ids={g['xiaoban_ids']}, params={g['params']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refine_name = "local_refine_" + "_".join([str(x) for x in bad_ids]) + f"_{stamp}"
    local_root = Path(f"/home/xth/forest_agent_project/outputs/local_refine/{refine_name}")
    ensure_dir(local_root)

    terrain_info = prepare_terrain_rasters(
        dem_tif=dem_tif,
        slope_tif=slope_tif,
        aspect_tif=aspect_tif,
        work_dir=str(local_root),
    )

    save_json(
        {
            "refine_name": refine_name,
            "base_config_path": base_config_path,
            "global_details_csv": global_details_csv,
            "global_inst_shp": global_inst_shp,
            "strategy_mode": strategy_mode,
            "base_params": base_params,
            "bad_xiaoban_ids": bad_ids,
            "group_plan": group_plan,
            "terrain_inputs": terrain_info,
        },
        str(local_root / "refine_plan.json"),
    )

    current_global_shp = global_inst_shp
    group_summaries = []

    for idx, group in enumerate(group_plan, 1):
        group_summary = run_one_group_refinement(
            base_config_path=base_config_path,
            base_cfg=base_cfg,
            current_global_shp=current_global_shp,
            group_idx=idx,
            group=group,
            xiaoban_id_field=xiaoban_id_field,
            buffer_m=buffer_m,
            local_root=local_root,
            terrain_info=terrain_info,
        )
        group_summaries.append(group_summary)
        current_global_shp = group_summary["merged_after_group_shp"]

    final_merged_shp = str(local_root / "merged_global_local_Y_inst.shp")
    copy_vector_dataset(current_global_shp, final_merged_shp)

    merged_eval = evaluate_merged_result(
        base_cfg=base_cfg,
        local_root=local_root,
        merged_shp=final_merged_shp,
        bad_ids=bad_ids,
        group_plan=[
            {
                "strategy": gs["strategy"],
                "xiaoban_ids": gs["xiaoban_ids"],
                "params": gs["params"],
                "group_name": gs["group_name"],
                "roi_dem_tif": gs.get("roi_dem_tif"),
                "roi_slope_tif": gs.get("roi_slope_tif"),
                "roi_aspect_tif": gs.get("roi_aspect_tif"),
            }
            for gs in group_summaries
        ],
        dem_tif=terrain_info.get("dem_tif"),
        slope_tif=terrain_info.get("slope_tif"),
        aspect_tif=terrain_info.get("aspect_tif"),
    )

    summary = {
        "refine_name": refine_name,
        "strategy_mode": strategy_mode,
        "bad_xiaoban_ids": bad_ids,
        "base_params": base_params,
        "terrain_inputs": terrain_info,
        "group_summaries": group_summaries,
        "merged_shp": final_merged_shp,
        "merged_metrics_json": merged_eval["merged_metrics_json"],
        "merged_details_csv": merged_eval["merged_details_csv"],
        "compare_json": merged_eval["compare_json"],
    }

    summary_path = local_root / "local_refine_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[local_refine] summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--global_details_csv", required=True)
    parser.add_argument("--global_inst_shp", required=True)
    parser.add_argument(
        "--best_params_json",
        required=False,
        default='{"diam_list":"96,192,320","tile":1536,"overlap":512,"tile_overlap":0.35,"augment":true,"iou_merge_thr":0.28,"bsize":256}',
        help='base params json, e.g. \'{"diam_list":"96,192,320","tile":1536,...}\'',
    )
    parser.add_argument("--xiaoban_id_field", default="XBH")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--buffer_m", type=float, default=5.0)
    parser.add_argument(
        "--strategy_mode",
        default="auto",
        choices=["auto", "single_params"],
        help="auto: 按局部误差类型 + terrain 自动分组选参; single_params: 所有 bad xiaoban 共用一组参数",
    )
    parser.add_argument("--dem_tif", default=None, help="Global DEM tif for terrain-aware ROI crop.")
    parser.add_argument("--slope_tif", default=None, help="Optional precomputed slope tif.")
    parser.add_argument("--aspect_tif", default=None, help="Optional precomputed aspect tif.")
    args = parser.parse_args()

    best_params = json.loads(args.best_params_json)

    run_local_refinement(
        base_config_path=args.base_config,
        global_details_csv=args.global_details_csv,
        global_inst_shp=args.global_inst_shp,
        best_params=best_params,
        xiaoban_id_field=args.xiaoban_id_field,
        top_k=args.top_k,
        buffer_m=args.buffer_m,
        strategy_mode=args.strategy_mode,
        dem_tif=args.dem_tif,
        slope_tif=args.slope_tif,
        aspect_tif=args.aspect_tif,
    )


if __name__ == "__main__":
    main()