from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd

from agent.config_builder import load_yaml, save_yaml
from agent.xiaoban_prompt_builder import build_group_param_prompt
from geo_layer.crown_metrics import standardize_inventory_crown_width

SAFE_PARAM_SPACE = {
    "diam_list": ["96,160,256", "96,192,320", "128,192,320", "128,256,320"],
    "tile": [1536, 2048],
    "overlap": [384, 512],
    "tile_overlap": [0.25, 0.35, 0.45],
    "augment": [True, False],
    "iou_merge_thr": [0.18, 0.22, 0.24, 0.28],
    "bsize": [256],
}


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_float(v: Any, default: float | None = None) -> float | None:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    if v is None:
        return default
    return bool(v)


def _pick_allowed(value: Any, allowed: List[Any], default: Any) -> Any:
    return value if value in allowed else default


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "diam_list": "96,160,256",
        "tile": 1536,
        "overlap": 384,
        "tile_overlap": 0.35,
        "augment": True,
        "iou_merge_thr": 0.18,
        "bsize": 256,
    }
    out = dict(defaults)
    out.update(params or {})
    out["diam_list"] = _pick_allowed(out.get("diam_list"), SAFE_PARAM_SPACE["diam_list"], defaults["diam_list"])
    out["tile"] = _pick_allowed(_safe_int(out.get("tile"), defaults["tile"]), SAFE_PARAM_SPACE["tile"], defaults["tile"])
    out["overlap"] = _pick_allowed(_safe_int(out.get("overlap"), defaults["overlap"]), SAFE_PARAM_SPACE["overlap"], defaults["overlap"])
    out["tile_overlap"] = _pick_allowed(
        _safe_float(out.get("tile_overlap"), defaults["tile_overlap"]),
        SAFE_PARAM_SPACE["tile_overlap"],
        defaults["tile_overlap"],
    )
    out["augment"] = _pick_allowed(_safe_bool(out.get("augment"), defaults["augment"]), SAFE_PARAM_SPACE["augment"], defaults["augment"])
    out["iou_merge_thr"] = _pick_allowed(
        _safe_float(out.get("iou_merge_thr"), defaults["iou_merge_thr"]),
        SAFE_PARAM_SPACE["iou_merge_thr"],
        defaults["iou_merge_thr"],
    )
    out["bsize"] = 256
    return out


def get_default_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    default_block = cfg.get("default_segmentation_params") or {}
    merged = {
        "diam_list": default_block.get("diam_list", cfg.get("diam_list")),
        "tile": default_block.get("tile", cfg.get("tile")),
        "overlap": default_block.get("overlap", cfg.get("overlap")),
        "tile_overlap": default_block.get("tile_overlap", cfg.get("tile_overlap")),
        "augment": default_block.get("augment", cfg.get("augment")),
        "iou_merge_thr": default_block.get("iou_merge_thr", cfg.get("iou_merge_thr")),
        "bsize": default_block.get("bsize", cfg.get("bsize", 256)),
    }
    return sanitize_params(merged)


def _dominant(series: pd.Series) -> str | None:
    vals = [str(v) for v in series.dropna().tolist() if str(v).strip()]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def _classify_strategy(row: pd.Series) -> str:
    slope_class = str(row.get("slope_class") or "unknown")
    landform_type = str(row.get("landform_type") or "unknown")
    weighted_density = _safe_float(row.get("weighted_expected_density"), 0.0) or 0.0
    weighted_crown = _safe_float(row.get("weighted_expected_mean_crown_width"), 0.0) or 0.0
    weighted_closure = _safe_float(row.get("weighted_expected_closure"), 0.0) or 0.0

    density_tag = "dense" if weighted_density >= 1600 else "normal" if weighted_density >= 900 else "sparse"
    crown_tag = "large" if weighted_crown >= 6.5 else "small" if weighted_crown <= 4.5 else "medium"
    closure_tag = "closed" if weighted_closure >= 0.70 else "open" if weighted_closure <= 0.45 else "mid"

    if slope_class in {"IV", "V", "VI"}:
        slope_bucket = "steep"
    elif slope_class in {"III"} or "moderate" in slope_class:
        slope_bucket = "moderate"
    elif slope_class in {"I", "II"} or "gentle" in slope_class:
        slope_bucket = "gentle"
    else:
        slope_bucket = "steep" if "mountain" in landform_type else "gentle"

    return f"{slope_bucket}_{density_tag}_{crown_tag}_{closure_tag}"


def _derive_weighted_features(xgdf: gpd.GeoDataFrame, cfg: Dict[str, Any]) -> gpd.GeoDataFrame:
    xgdf = xgdf.copy()
    tree_count_field = cfg.get("tree_count_field")
    crown_field = cfg.get("crown_field")
    closure_field = cfg.get("closure_field")
    area_ha_field = cfg.get("area_ha_field")
    density_field = cfg.get("density_field")

    if tree_count_field and tree_count_field in xgdf.columns:
        xgdf[tree_count_field] = pd.to_numeric(xgdf[tree_count_field], errors="coerce")
    if crown_field and crown_field in xgdf.columns:
        xgdf[crown_field] = xgdf[crown_field].apply(standardize_inventory_crown_width)
        xgdf["inventory_crown_width_m"] = xgdf[crown_field]
    if closure_field and closure_field in xgdf.columns:
        xgdf[closure_field] = pd.to_numeric(xgdf[closure_field], errors="coerce")
    if area_ha_field and area_ha_field in xgdf.columns:
        xgdf[area_ha_field] = pd.to_numeric(xgdf[area_ha_field], errors="coerce")
    if density_field and density_field in xgdf.columns:
        xgdf[density_field] = pd.to_numeric(xgdf[density_field], errors="coerce")

    if "overlap_ratio_in_xiaoban" in xgdf.columns:
        ratio = pd.to_numeric(xgdf["overlap_ratio_in_xiaoban"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        ratio = pd.Series(1.0, index=xgdf.index, dtype=float)

    if "clip_area_ha" in xgdf.columns:
        clip_area_ha = pd.to_numeric(xgdf["clip_area_ha"], errors="coerce").fillna(0.0)
    else:
        metric_gdf = xgdf.to_crs(xgdf.estimate_utm_crs() if not xgdf.crs.is_projected else xgdf.crs)
        clip_area_ha = pd.Series(metric_gdf.geometry.area / 10000.0, index=xgdf.index, dtype=float)

    if tree_count_field and tree_count_field in xgdf.columns:
        xgdf["weighted_expected_tree_count"] = xgdf[tree_count_field].fillna(0.0) * ratio
    else:
        xgdf["weighted_expected_tree_count"] = 0.0

    if "inventory_crown_width_m" in xgdf.columns:
        xgdf["weighted_expected_mean_crown_width"] = xgdf["inventory_crown_width_m"]
    elif crown_field and crown_field in xgdf.columns:
        xgdf["weighted_expected_mean_crown_width"] = xgdf[crown_field]
    else:
        xgdf["weighted_expected_mean_crown_width"] = pd.NA

    if closure_field and closure_field in xgdf.columns:
        xgdf["weighted_expected_closure"] = xgdf[closure_field]
    else:
        xgdf["weighted_expected_closure"] = pd.NA

    if density_field and density_field in xgdf.columns:
        xgdf["weighted_expected_density"] = xgdf[density_field]
    elif tree_count_field and area_ha_field and tree_count_field in xgdf.columns and area_ha_field in xgdf.columns:
        xgdf["weighted_expected_density"] = xgdf[tree_count_field] / xgdf[area_ha_field].replace(0, pd.NA)
    else:
        xgdf["weighted_expected_density"] = pd.NA

    xgdf["clip_area_ha"] = clip_area_ha
    return xgdf


def heuristic_params_for_group(group_df: gpd.GeoDataFrame, default_params: Dict[str, Any]) -> Dict[str, Any]:
    params = sanitize_params(default_params)
    weighted_density = _safe_float((group_df["weighted_expected_density"] * group_df["clip_area_ha"]).sum() / max(group_df["clip_area_ha"].sum(), 1e-6), 0.0) or 0.0
    weighted_crown = _safe_float((pd.to_numeric(group_df["weighted_expected_mean_crown_width"], errors="coerce").fillna(0.0) * group_df["clip_area_ha"]).sum() / max(group_df["clip_area_ha"].sum(), 1e-6), 0.0) or 0.0
    weighted_closure = _safe_float((pd.to_numeric(group_df["weighted_expected_closure"], errors="coerce").fillna(0.0) * group_df["clip_area_ha"]).sum() / max(group_df["clip_area_ha"].sum(), 1e-6), 0.0) or 0.0
    dominant_slope = _dominant(group_df["slope_class"]) if "slope_class" in group_df.columns else None
    dominant_landform = _dominant(group_df["landform_type"]) if "landform_type" in group_df.columns else None
    dominant_slope = dominant_slope or "unknown"
    dominant_landform = dominant_landform or "unknown"

    if dominant_slope in {"IV", "V", "VI"} or "mountain" in dominant_landform:
        params["overlap"] = 512
        params["tile_overlap"] = 0.35
        params["augment"] = False
        params["iou_merge_thr"] = 0.28
    elif dominant_slope in {"III"}:
        params["overlap"] = 512
        params["tile_overlap"] = max(float(params["tile_overlap"]), 0.35)
        params["iou_merge_thr"] = max(float(params["iou_merge_thr"]), 0.28)

    if weighted_density >= 1600 or weighted_closure >= 0.70:
        params["diam_list"] = "128,192,320"
        params["tile"] = 2048
        params["overlap"] = 512
        params["tile_overlap"] = 0.35
        params["augment"] = False
        params["iou_merge_thr"] = 0.28
    elif weighted_density <= 700 and weighted_crown >= 6.5:
        params["diam_list"] = "128,256,320"
        params["tile"] = 2048
        params["tile_overlap"] = 0.35
        params["iou_merge_thr"] = 0.28
    elif weighted_crown <= 4.5:
        params["diam_list"] = "96,192,320"
        params["tile"] = 1536
        params["tile_overlap"] = 0.35
        params["augment"] = False
        params["iou_merge_thr"] = 0.24
    elif weighted_crown >= 5.5:
        params["diam_list"] = "128,192,320"
        params["tile"] = 2048
        params["tile_overlap"] = 0.35
        params["augment"] = False
        params["iou_merge_thr"] = 0.28

    return sanitize_params(params)


def _maybe_apply_llm(
    groups: List[Dict[str, Any]],
    default_params: Dict[str, Any],
    run_meta: Dict[str, Any],
    spatial_context: Dict[str, Any],
    enabled: bool,
) -> List[Dict[str, Any]]:
    if not enabled:
        return groups
    if not os.environ.get("ARK_API_KEY") or not os.environ.get("ARK_MODEL"):
        return groups

    prompt = build_group_param_prompt(
        run_meta=run_meta,
        groups=groups,
        default_params=default_params,
        spatial_context=spatial_context,
    )
    try:
        from agent.doubao_client import call_doubao_json

        llm_output = call_doubao_json(prompt)
    except Exception as e:
        print(f"[xiaoban_planner] LLM planner fallback to heuristic params: {e}")
        return groups

    group_param_map = {}
    for item in llm_output.get("groups", []):
        group_id = item.get("group_id")
        if group_id:
            group_param_map[group_id] = sanitize_params(item.get("params", {}))

    for group in groups:
        if group["group_id"] in group_param_map:
            group["params"] = group_param_map[group["group_id"]]
            group["planner_source"] = "llm"
    return groups


def build_group_plan_for_config(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    xiaoban_path = cfg.get("xiaoban_shp")
    xiaoban_id_field = cfg.get("xiaoban_id_field")
    if not xiaoban_path or not Path(xiaoban_path).exists():
        raise FileNotFoundError(f"xiaoban_shp not found: {xiaoban_path}")
    if not xiaoban_id_field:
        raise ValueError("xiaoban_id_field is required for grouped inference")

    xgdf = gpd.read_file(xiaoban_path)
    if xgdf.empty:
        raise ValueError(f"xiaoban_shp is empty: {xiaoban_path}")
    xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
    xgdf = _derive_weighted_features(xgdf, cfg)
    xgdf["strategy_label"] = xgdf.apply(_classify_strategy, axis=1)

    default_params = get_default_params(cfg)
    groups: List[Dict[str, Any]] = []

    for idx, (strategy_label, group_df) in enumerate(xgdf.groupby("strategy_label"), 1):
        clip_area_ha = pd.to_numeric(group_df["clip_area_ha"], errors="coerce").fillna(0.0)
        total_area = float(clip_area_ha.sum())
        weighted_inventory = {
            "expected_tree_count": float(pd.to_numeric(group_df["weighted_expected_tree_count"], errors="coerce").fillna(0.0).sum()),
            "expected_mean_crown_width": _safe_float(
                (pd.to_numeric(group_df["weighted_expected_mean_crown_width"], errors="coerce").fillna(0.0) * clip_area_ha).sum()
                / max(total_area, 1e-6),
                None,
            ),
            "expected_closure": _safe_float(
                (pd.to_numeric(group_df["weighted_expected_closure"], errors="coerce").fillna(0.0) * clip_area_ha).sum()
                / max(total_area, 1e-6),
                None,
            ),
            "expected_density": _safe_float(
                (pd.to_numeric(group_df["weighted_expected_density"], errors="coerce").fillna(0.0) * clip_area_ha).sum()
                / max(total_area, 1e-6),
                None,
            ),
        }
        groups.append(
            {
                "group_id": f"group_{idx:03d}",
                "strategy_label": strategy_label,
                "planner_source": "heuristic",
                "num_xiaoban": int(len(group_df)),
                "xiaoban_ids": group_df[xiaoban_id_field].astype(str).tolist(),
                "dominant_terrain": {
                    "landform_type": _dominant(group_df["landform_type"]) if "landform_type" in group_df.columns else None,
                    "slope_class": _dominant(group_df["slope_class"]) if "slope_class" in group_df.columns else None,
                    "aspect_class": _dominant(group_df["aspect_class"]) if "aspect_class" in group_df.columns else None,
                    "slope_position_class": _dominant(group_df["slope_position_class"]) if "slope_position_class" in group_df.columns else None,
                },
                "weighted_inventory": weighted_inventory,
                "params": heuristic_params_for_group(group_df, default_params),
            }
        )

    groups = _maybe_apply_llm(
        groups=groups,
        default_params=default_params,
        run_meta={
            "run_name": cfg.get("run_name"),
            "input_image": cfg.get("input_image"),
            "xiaoban_shp": cfg.get("xiaoban_shp"),
        },
        spatial_context=cfg.get("spatial_context_object") or {},
        enabled=bool(cfg.get("grouped_inference_use_llm", True)),
    )

    return {
        "config_path": config_path,
        "run_name": cfg.get("run_name"),
        "planner_mode": "grouped_inference",
        "default_params": default_params,
        "groups": groups,
    }


def materialize_group_configs(plan: Dict[str, Any], base_config_path: str, out_dir: str | Path) -> List[str]:
    cfg = load_yaml(base_config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for group in plan["groups"]:
        group_cfg = dict(cfg)
        for key, value in group["params"].items():
            group_cfg[key] = value
        group_cfg["run_name"] = f"{cfg.get('run_name', 'grouped_run')}_{group['group_id']}"
        group_cfg["_grouped_dispatch_active"] = True
        out_path = out_dir / f"{group['group_id']}.yaml"
        save_yaml(group_cfg, str(out_path))
        created.append(str(out_path))
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    plan = build_group_plan_for_config(args.config)
    save_json(plan, args.out_json)
    print(f"[xiaoban_planner] saved plan to: {args.out_json}")


if __name__ == "__main__":
    main()
