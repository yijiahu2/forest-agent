from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config_builder import load_yaml, save_yaml
from agent.local_refine import (  # noqa: E402
    build_local_refine_config,
    clip_xiaoban_to_geometry_with_fields,
    crop_raster_to_geometry,
    crop_roi_terrain_bundle,
    ensure_dir,
    ensure_parent,
    make_bad_roi_gdf,
)
from agent.xiaoban_planner import build_group_plan_for_config, save_json  # noqa: E402
from reporting.experiment_report import build_experiment_report  # noqa: E402
from geo_layer.instance_ops import (  # noqa: E402
    dedupe_instances_by_overlap,
    filter_instances_to_ids_by_overlap,
    suppress_small_boundary_fragments,
)
from geo_layer.spatial_context import prepare_spatial_context, summarize_xiaoban_terrain_classes  # noqa: E402
from tools.stage_cache_client import run_stage1_via_worker, run_stage2_via_worker  # noqa: E402
from tools.process_runner import run_streaming  # noqa: E402


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_path(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return True


def remove_vector_dataset(path: str | Path) -> list[str]:
    p = Path(path)
    removed: list[str] = []
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        cand = p.with_suffix(ext)
        if cand.exists():
            cand.unlink()
            removed.append(str(cand))
    return removed


def slim_group_summary(group_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "group_id": group_summary.get("group_id"),
        "strategy_label": group_summary.get("strategy_label"),
        "planner_source": group_summary.get("planner_source"),
        "group_name": group_summary.get("group_name"),
        "xiaoban_ids": group_summary.get("xiaoban_ids"),
        "params": group_summary.get("params"),
        "terrain_summary": group_summary.get("terrain_summary"),
    }


def cleanup_group_artifacts(group_summaries: List[Dict[str, Any]], keep_debug_outputs: bool) -> Dict[str, Any]:
    removed: Dict[str, Any] = {"removed_files": [], "removed_vector_datasets": []}
    if keep_debug_outputs:
        return removed

    for group_summary in group_summaries:
        for key in [
            "roi_image",
            "roi_dem_tif",
            "roi_slope_tif",
            "roi_aspect_tif",
            "roi_landform_tif",
            "roi_slope_position_tif",
            "m_sem_tif",
            "y_inst_tif",
            "y_inst_color_png",
        ]:
            path = group_summary.get(key)
            if path and remove_path(path):
                removed["removed_files"].append(path)

        for key in ["roi_xiaoban_shp", "group_inst_shp", "filtered_group_inst_shp"]:
            path = group_summary.get(key)
            if path:
                removed_vec = remove_vector_dataset(path)
                if removed_vec:
                    removed["removed_vector_datasets"].append({"label": key, "paths": removed_vec})

        group_output_dir = group_summary.get("group_output_dir")
        if group_output_dir and remove_path(group_output_dir):
            removed["removed_files"].append(group_output_dir)

        if group_output_dir:
            group_dir = Path(group_output_dir).parent
            if group_dir.exists():
                shutil.rmtree(group_dir, ignore_errors=True)
                removed["removed_files"].append(str(group_dir))

    return removed


def _group_root_from_cfg(cfg: Dict[str, Any]) -> Path:
    metrics_parent = Path(cfg["metrics_json"]).resolve().parent
    return metrics_parent / "grouped_inference"


def _prepare_grouped_runtime_config(config_path: str) -> str:
    cfg = load_yaml(config_path)
    context_dir = Path(cfg["metrics_json"]).resolve().parent / "grouped_spatial_context"
    ensure_dir(context_dir)

    existing_xiaoban = cfg.get("xiaoban_shp")
    if (
        cfg.get("spatial_context_summary_json")
        and Path(str(cfg["spatial_context_summary_json"])).exists()
        and existing_xiaoban
        and Path(str(existing_xiaoban)).exists()
        and Path(str(existing_xiaoban)).suffix.lower() == ".gpkg"
        and "context_xiaoban" in Path(str(existing_xiaoban)).name
        and cfg.get("dem_tif")
        and Path(str(cfg["dem_tif"])).exists()
    ):
        return config_path

    context_result = prepare_spatial_context(
        dom_tif=cfg["input_image"],
        dem_tif=cfg.get("dem_tif"),
        xiaoban_shp=cfg.get("xiaoban_shp"),
        out_dir=context_dir,
        xiaoban_id_field=cfg.get("xiaoban_id_field"),
        tree_count_field=cfg.get("tree_count_field"),
        crown_field=cfg.get("crown_field"),
        closure_field=cfg.get("closure_field"),
        area_ha_field=cfg.get("area_ha_field"),
        density_field=cfg.get("density_field"),
        flat_slope_threshold_deg=float(cfg.get("flat_slope_threshold_deg", 5.0)),
        plain_relief_threshold_m=float(cfg.get("plain_relief_threshold_m", 30.0)),
    )

    cfg["xiaoban_shp"] = context_result.get("xiaoban_shp", cfg.get("xiaoban_shp"))
    for key in ["dem_tif", "slope_tif", "aspect_tif", "landform_tif", "slope_position_tif"]:
        if context_result.get(key):
            cfg[key] = context_result[key]
    cfg["spatial_context_summary_json"] = context_result.get("summary_json")
    cfg["terrain_landform_field"] = "landform_type"
    cfg["terrain_slope_class_field"] = "slope_class"
    cfg["terrain_aspect_class_field"] = "aspect_class"
    cfg["terrain_slope_position_field"] = "slope_position_class"

    runtime_config = context_dir / "runtime_grouped_config.yaml"
    save_yaml(cfg, str(runtime_config))
    return str(runtime_config)


def _filter_instances_to_group(
    inst_shp: str,
    group_xiaoban_shp: str,
    xiaoban_id_field: str,
    group_ids: List[str],
    out_shp: str,
) -> str | None:
    inst = gpd.read_file(inst_shp)
    if inst.empty:
        return None

    xgdf = gpd.read_file(group_xiaoban_shp)
    xgdf[xiaoban_id_field] = xgdf[xiaoban_id_field].astype(str)
    target = xgdf[xgdf[xiaoban_id_field].isin([str(x) for x in group_ids])].copy()
    if target.empty:
        return None
    filtered = filter_instances_to_ids_by_overlap(
        inst_gdf=inst,
        polygon_gdf=xgdf,
        id_field=xiaoban_id_field,
        allowed_ids=group_ids,
    )
    if filtered.empty:
        return None

    filtered = suppress_small_boundary_fragments(filtered, target, boundary_band_m=1.5, min_area_m2=6.0)
    filtered = dedupe_instances_by_overlap(filtered, overlap_ratio_thr=0.5)

    out_path = Path(out_shp)
    ensure_parent(out_path)
    filtered.to_file(out_path)
    return str(out_path)


def _merge_group_outputs(group_shps: List[str], out_shp: str) -> str:
    gdfs = [gpd.read_file(path) for path in group_shps if path and Path(path).exists()]
    gdfs = [gdf for gdf in gdfs if not gdf.empty]
    if not gdfs:
        raise ValueError("No non-empty grouped outputs to merge.")

    base_crs = gdfs[0].crs
    normalized = [gdf.to_crs(base_crs) for gdf in gdfs]
    merged = pd.concat(normalized, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=base_crs)
    merged = dedupe_instances_by_overlap(merged, overlap_ratio_thr=0.5)

    out_path = Path(out_shp)
    ensure_parent(out_path)
    merged.to_file(out_path)
    return str(out_path)


def _run_group(
    cfg: Dict[str, Any],
    base_config_path: str,
    group: Dict[str, Any],
    group_root: Path,
    terrain_info: Dict[str, Any],
    xiaoban_id_field: str,
) -> Dict[str, Any]:
    group_dir = group_root / group["group_id"]
    ensure_dir(group_dir)

    roi_gdf = make_bad_roi_gdf(
        xiaoban_shp=cfg["xiaoban_shp"],
        xiaoban_id_field=xiaoban_id_field,
        bad_ids=group["xiaoban_ids"],
        buffer_m=float(cfg.get("grouped_inference_buffer_m", 5.0)),
    )

    roi_image = str(group_dir / "roi_image.tif")
    roi_xiaoban = str(group_dir / "roi_xiaoban.gpkg")
    roi_output_dir = str(group_dir / "seg_output")
    roi_config = str(group_dir / "group_config.yaml")

    crop_raster_to_geometry(cfg["input_image"], roi_gdf, roi_image)
    clip_xiaoban_to_geometry_with_fields(
        src_vector=cfg["xiaoban_shp"],
        geom_gdf=roi_gdf,
        out_vector=roi_xiaoban,
        xiaoban_id_field=cfg["xiaoban_id_field"],
        tree_count_field=cfg.get("tree_count_field"),
        crown_field=cfg.get("crown_field"),
        closure_field=cfg.get("closure_field"),
        area_ha_field=cfg.get("area_ha_field"),
        density_field=cfg.get("density_field"),
    )

    terrain_roi = crop_roi_terrain_bundle(
        roi_geom_gdf=roi_gdf,
        roi_dir=str(group_dir),
        dem_tif=terrain_info.get("dem_tif"),
        slope_tif=terrain_info.get("slope_tif"),
        aspect_tif=terrain_info.get("aspect_tif"),
        landform_tif=terrain_info.get("landform_tif"),
        slope_position_tif=terrain_info.get("slope_position_tif"),
    )

    local_cfg = build_local_refine_config(
        base_config_path=base_config_path,
        out_config_path=roi_config,
        local_input_image=roi_image,
        local_output_dir=roi_output_dir,
        local_xiaoban_shp=roi_xiaoban,
        params=group["params"],
        run_name=f"{cfg.get('run_name', 'grouped_run')}_{group['group_id']}",
        local_dem_tif=terrain_roi.get("roi_dem_tif"),
        local_slope_tif=terrain_roi.get("roi_slope_tif"),
        local_aspect_tif=terrain_roi.get("roi_aspect_tif"),
        local_landform_tif=terrain_roi.get("roi_landform_tif"),
        local_slope_position_tif=terrain_roi.get("roi_slope_position_tif"),
    )
    local_cfg["_grouped_dispatch_active"] = True
    local_cfg["disable_mlflow"] = True
    with open(roi_config, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(local_cfg, f, allow_unicode=True, sort_keys=False)
    stage1_info = run_stage1_via_worker(local_cfg)
    stage2_info = run_stage2_via_worker(local_cfg, stage1_info["m_sem_tif"])
    group_inst_shp = stage2_info["y_inst_shp"]
    filtered_shp = _filter_instances_to_group(
        inst_shp=group_inst_shp,
        group_xiaoban_shp=roi_xiaoban,
        xiaoban_id_field=xiaoban_id_field,
        group_ids=group["xiaoban_ids"],
        out_shp=str(group_dir / "Y_inst_group_filtered.shp"),
    )

    terrain_summary = {}
    try:
        terrain_summary = summarize_xiaoban_terrain_classes(gpd.read_file(roi_xiaoban))
    except Exception:
        terrain_summary = {}

    return {
        "group_id": group["group_id"],
        "strategy_label": group.get("strategy_label"),
        "planner_source": group.get("planner_source"),
        "group_name": local_cfg["run_name"],
        "xiaoban_ids": group["xiaoban_ids"],
        "params": group["params"],
        "roi_image": roi_image,
        "roi_xiaoban_shp": roi_xiaoban,
        "roi_dem_tif": terrain_roi.get("roi_dem_tif"),
        "roi_slope_tif": terrain_roi.get("roi_slope_tif"),
        "roi_aspect_tif": terrain_roi.get("roi_aspect_tif"),
        "roi_landform_tif": terrain_roi.get("roi_landform_tif"),
        "roi_slope_position_tif": terrain_roi.get("roi_slope_position_tif"),
        "group_output_dir": roi_output_dir,
        "group_inst_shp": group_inst_shp,
        "filtered_group_inst_shp": filtered_shp,
        "m_sem_tif": stage1_info["m_sem_tif"],
        "y_inst_tif": stage2_info["y_inst_tif"],
        "y_inst_color_png": stage2_info["y_inst_color_png"],
        "terrain_info": {
            "dem_tif": local_cfg.get("dem_tif"),
            "slope_tif": local_cfg.get("slope_tif"),
            "aspect_tif": local_cfg.get("aspect_tif"),
            "landform_tif": local_cfg.get("landform_tif"),
            "slope_position_tif": local_cfg.get("slope_position_tif"),
        },
        "terrain_summary": terrain_summary,
    }


def _run_final_evaluation(cfg: Dict[str, Any], merged_shp: str, terrain_info: Dict[str, Any]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_xiaoban_consistency",
        "--inst_shp",
        merged_shp,
        "--patch_raster",
        cfg["input_image"],
        "--xiaoban_shp",
        cfg["xiaoban_shp"],
        "--out_json",
        cfg["metrics_json"],
        "--out_csv",
        cfg["details_csv"],
        "--id_field",
        cfg["xiaoban_id_field"],
        "--tree_count_field",
        cfg["tree_count_field"],
        "--crown_field",
        cfg["crown_field"],
        "--closure_field",
        cfg["closure_field"],
        "--area_ha_field",
        cfg["area_ha_field"],
        "--flat_slope_threshold_deg",
        str(cfg.get("flat_slope_threshold_deg", 5.0)),
        "--plain_relief_threshold_m",
        str(cfg.get("plain_relief_threshold_m", 30.0)),
    ]
    if cfg.get("density_field"):
        cmd.extend(["--density_field", str(cfg["density_field"])])
    if terrain_info.get("dem_tif"):
        cmd.extend(["--dem_tif", str(terrain_info["dem_tif"])])
    if terrain_info.get("slope_tif"):
        cmd.extend(["--slope_tif", str(terrain_info["slope_tif"])])
    if terrain_info.get("aspect_tif"):
        cmd.extend(["--aspect_tif", str(terrain_info["aspect_tif"])])

    res = run_streaming(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"grouped evaluation failed:\n{res.stdout}")

    return {
        "metrics_json": cfg["metrics_json"],
        "details_csv": cfg["details_csv"],
        "metrics": load_json(cfg["metrics_json"]),
    }


def run_grouped_experiment(config_path: str) -> Dict[str, Any]:
    runtime_config_path = _prepare_grouped_runtime_config(config_path)
    cfg = load_yaml(runtime_config_path)
    group_plan = build_group_plan_for_config(runtime_config_path)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(Path(cfg["metrics_json"]))
    ensure_parent(Path(cfg["details_csv"]))

    terrain_info = {
        "dem_tif": cfg.get("dem_tif"),
        "slope_tif": cfg.get("slope_tif"),
        "aspect_tif": cfg.get("aspect_tif"),
        "landform_tif": cfg.get("landform_tif"),
        "slope_position_tif": cfg.get("slope_position_tif"),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_root = _group_root_from_cfg(cfg) / f"{cfg.get('run_name', 'grouped_run')}_{stamp}"
    ensure_dir(group_root)

    plan_json = group_root / "xiaoban_plan.json"
    save_json(group_plan, plan_json)

    group_summaries = []
    merged_inputs: List[str] = []
    for group in group_plan["groups"]:
        group_summary = _run_group(
            cfg=cfg,
            base_config_path=runtime_config_path,
            group=group,
            group_root=group_root,
            terrain_info=terrain_info,
            xiaoban_id_field=cfg["xiaoban_id_field"],
        )
        group_summaries.append(group_summary)
        if group_summary.get("filtered_group_inst_shp"):
            merged_inputs.append(group_summary["filtered_group_inst_shp"])

    merged_shp = str(output_dir / "Y_inst.shp")
    _merge_group_outputs(merged_inputs, merged_shp)
    eval_info = _run_final_evaluation(cfg, merged_shp, terrain_info)
    cleanup_info = cleanup_group_artifacts(
        group_summaries=group_summaries,
        keep_debug_outputs=bool(cfg.get("keep_debug_outputs", False)),
    )
    if not cfg.get("keep_debug_outputs", False):
        remove_path(plan_json)
    slim_groups = [slim_group_summary(gs) for gs in group_summaries]

    summary = {
        "mode": "grouped_inference",
        "run_name": cfg.get("run_name"),
        "config_path": config_path,
        "group_root": str(group_root) if cfg.get("keep_debug_outputs", False) else None,
        "group_plan_json": str(plan_json) if cfg.get("keep_debug_outputs", False) else None,
        "group_count": len(slim_groups),
        "group_summaries": slim_groups,
        "merged_inst_shp": merged_shp,
        "metrics_json": eval_info["metrics_json"],
        "details_csv": eval_info["details_csv"],
        "metrics": eval_info["metrics"],
        "terrain_info": terrain_info,
        "cleanup": cleanup_info,
    }
    summary_json = str(Path(cfg["metrics_json"]).resolve().parent / "run_experiment_summary.json")
    summary["summary_json"] = summary_json
    save_json(summary, summary_json)
    report_md = str(Path(cfg["metrics_json"]).resolve().parent / "run_experiment_report.md")
    report_path = build_experiment_report(summary, report_md)
    if cfg.get("keep_debug_outputs", False):
        summary["report_md"] = report_path
    else:
        remove_path(report_path)
        summary["report_md"] = None
    save_json(summary, summary_json)
    print(f"[grouped_runner] summary saved to: {summary_json}")
    if cfg.get("keep_debug_outputs", False):
        print(f"[grouped_runner] report saved to: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_grouped_experiment(args.config)


if __name__ == "__main__":
    main()
