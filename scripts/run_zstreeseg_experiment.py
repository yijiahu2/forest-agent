from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow

from agent.config_builder import load_yaml
from geo_layer.terrain_features import generate_terrain_products


def ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path):
    path = Path(path)
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    print("\n===== CMD =====")
    print(" ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    print("\n===== STDOUT =====")
    print(res.stdout)
    print("\n===== STDERR =====")
    print(res.stderr)
    return res


def run_bash_in_conda_env(
    command: str,
    conda_sh: str,
    conda_env: str,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    bash_cmd = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)} && {command}"
    print("\n===== BASH CMD =====")
    print(bash_cmd)
    res = subprocess.run(
        ["bash", "-lc", bash_cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    print("\n===== STDOUT =====")
    print(res.stdout)
    print("\n===== STDERR =====")
    print(res.stderr)
    return res


def require_file(path: str | Path, desc: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def prepare_terrain_inputs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dem_tif = cfg.get("dem_tif")
    slope_tif = cfg.get("slope_tif")
    aspect_tif = cfg.get("aspect_tif")

    result = {
        "dem_tif": dem_tif,
        "slope_tif": slope_tif,
        "aspect_tif": aspect_tif,
        "landform_tif": cfg.get("landform_tif"),
        "slope_position_tif": cfg.get("slope_position_tif"),
        "terrain_generated": False,
    }

    if not dem_tif:
        return result

    if slope_tif and aspect_tif:
        return result

    metrics_json = cfg.get("metrics_json")
    if metrics_json:
        terrain_dir = Path(metrics_json).resolve().parent / "terrain_cache"
    else:
        terrain_dir = Path(cfg["output_dir"]).resolve() / "terrain_cache"
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


def get_stage_output_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    output_dir = Path(cfg["output_dir"])
    return {
        "m_sem_tif": str(output_dir / "M_sem.tif"),
        "m_sem_png": str(output_dir / "M_sem.png"),
        "y_inst_tif": str(output_dir / "Y_inst.tif"),
        "y_inst_shp": str(output_dir / "Y_inst.shp"),
        "y_inst_color_png": str(output_dir / "Y_inst_color.png"),
    }


def normalize_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def get_eval_output_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    return {
        "metrics_json": cfg["metrics_json"],
        "details_csv": cfg["details_csv"],
    }


def collect_run_metadata(cfg: Dict[str, Any], terrain_info: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "experiment_name",
        "run_name",
        "compartment_id",
        "patch_id",
        "forest_type",
        "agent_version",
        "input_image",
        "output_dir",
        "xiaoban_shp",
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "density_field",
        "area_ha_field",
        "stage1_script",
        "stage2_script",
        "diam_list",
        "tile",
        "overlap",
        "tile_overlap",
        "bsize",
        "augment",
        "iou_merge_thr",
        "stage1_ckpt",
        "stage1_extra_args",
    ]
    meta = {k: cfg.get(k) for k in keys if k in cfg}
    meta["terrain_info"] = terrain_info
    meta["spatial_context_object_json"] = cfg.get("spatial_context_object_json")
    meta["terrain_constraint_fields"] = {
        "terrain_landform_field": cfg.get("terrain_landform_field", "landform_type"),
        "terrain_slope_class_field": cfg.get("terrain_slope_class_field", "slope_class"),
        "terrain_aspect_class_field": cfg.get("terrain_aspect_class_field", "aspect_class"),
        "terrain_slope_position_field": cfg.get("terrain_slope_position_field", "slope_position_class"),
    }
    return meta


def _normalize_extra_args(extra_args: Any) -> list[str]:
    if extra_args is None:
        return []
    if isinstance(extra_args, list):
        return [str(x) for x in extra_args]
    if isinstance(extra_args, str) and extra_args.strip():
        return shlex.split(extra_args)
    return []


def _normalize_stage1_extra_args(extra_args: Any) -> list[str]:
    """
    规范化 stage1_extra_args，并移除其中潜在重复的 ckpt 参数，
    避免与 stage1_ckpt 字段重复注入。
    """
    args = _normalize_extra_args(extra_args)

    cleaned: list[str] = []
    skip_next = False

    for item in args:
        s = str(item).strip()

        if skip_next:
            skip_next = False
            continue

        if s in {"--ckpt", "--checkpoint"}:
            skip_next = True
            continue

        if s.startswith("--ckpt=") or s.startswith("--checkpoint="):
            continue

        cleaned.append(s)

    return cleaned


def run_stage1(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = get_stage_output_paths(cfg)
    ensure_parent(paths["m_sem_tif"])

    script = cfg["stage1_script"]
    work_dir = cfg["work_dir"]
    input_image = cfg["input_image"]

    cmd = [
        "python",
        script,
        "--in_tif",
        input_image,
        "--out_dir",
        cfg["output_dir"],
    ]

    stage1_ckpt = cfg.get("stage1_ckpt")
    if stage1_ckpt:
        require_file(stage1_ckpt, "stage1_ckpt")
        cmd.extend(["--ckpt", str(stage1_ckpt)])

    stage1_extra_args = _normalize_stage1_extra_args(cfg.get("stage1_extra_args"))
    if stage1_extra_args:
        cmd.extend(stage1_extra_args)

    res = run_bash_in_conda_env(
        command=" ".join(shlex.quote(x) for x in cmd),
        conda_sh=cfg["conda_sh"],
        conda_env=cfg["conda_env"],
        cwd=work_dir,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Stage1 failed:\n{res.stderr}")

    require_file(paths["m_sem_tif"], "Stage1 M_sem.tif")
    return {
        "cmd": cmd,
        "m_sem_tif": paths["m_sem_tif"],
        "m_sem_png": paths["m_sem_png"],
    }


def run_stage2(cfg: Dict[str, Any], m_sem_tif: str) -> Dict[str, Any]:
    paths = get_stage_output_paths(cfg)

    cmd = [
        "python",
        cfg["stage2_script"],
        "--in_tif",
        cfg["input_image"],
        "--msem_tif",
        m_sem_tif,
        "--out_dir",
        cfg["output_dir"],
        "--diam_list",
        str(cfg["diam_list"]),
        "--tile",
        str(cfg["tile"]),
        "--overlap",
        str(cfg["overlap"]),
        "--tile_overlap",
        str(cfg["tile_overlap"]),
        "--bsize",
        str(cfg["bsize"]),
        "--iou_merge_thr",
        str(cfg["iou_merge_thr"]),
    ]

    if normalize_bool(cfg.get("augment", True)):
        cmd.append("--augment")

    res = run_bash_in_conda_env(
        command=" ".join(shlex.quote(x) for x in cmd),
        conda_sh=cfg["conda_sh"],
        conda_env=cfg["conda_env"],
        cwd=cfg["work_dir"],
    )
    if res.returncode != 0:
        raise RuntimeError(f"Stage2 failed:\n{res.stderr}")

    require_file(paths["y_inst_shp"], "Stage2 Y_inst.shp")
    return {
        "cmd": cmd,
        "y_inst_tif": paths["y_inst_tif"],
        "y_inst_shp": paths["y_inst_shp"],
        "y_inst_color_png": paths["y_inst_color_png"],
    }

def run_evaluation(cfg: Dict[str, Any], inst_shp: str, terrain_info: Dict[str, Any]) -> Dict[str, Any]:
    eval_paths = get_eval_output_paths(cfg)
    ensure_parent(eval_paths["metrics_json"])
    ensure_parent(eval_paths["details_csv"])

    cmd = [
        "python",
        "-m",
        "scripts.evaluate_xiaoban_consistency",
        "--inst_shp", inst_shp,
        "--patch_raster", cfg["input_image"],
        "--xiaoban_shp", cfg["xiaoban_shp"],
        "--out_json", eval_paths["metrics_json"],
        "--out_csv", eval_paths["details_csv"],
        "--id_field", cfg["xiaoban_id_field"],
        "--tree_count_field", cfg["tree_count_field"],
        "--crown_field", cfg["crown_field"],
        "--closure_field", cfg["closure_field"],
        "--area_ha_field", cfg["area_ha_field"],
    ]

    if cfg.get("density_field"):
        cmd.extend(["--density_field", str(cfg["density_field"])])

    if terrain_info.get("dem_tif"):
        cmd.extend(["--dem_tif", str(terrain_info["dem_tif"])])
    if terrain_info.get("slope_tif"):
        cmd.extend(["--slope_tif", str(terrain_info["slope_tif"])])
    if terrain_info.get("aspect_tif"):
        cmd.extend(["--aspect_tif", str(terrain_info["aspect_tif"])])

    # 新增 terrain 规则参数
    cmd.extend([
        "--flat_slope_threshold_deg", str(cfg.get("flat_slope_threshold_deg", 5.0)),
        "--plain_relief_threshold_m", str(cfg.get("plain_relief_threshold_m", 30.0)),
    ])

    res = run_cmd(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Evaluation failed:\n{res.stderr}")

    require_file(eval_paths["metrics_json"], "Evaluation metrics_json")
    require_file(eval_paths["details_csv"], "Evaluation details_csv")

    metrics = load_json(eval_paths["metrics_json"])
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Evaluation metrics json is empty or invalid: {eval_paths['metrics_json']}")

    return {
        "cmd": cmd,
        "metrics_json": eval_paths["metrics_json"],
        "details_csv": eval_paths["details_csv"],
        "metrics": metrics,
        "terrain_info": terrain_info,
        "terrain_rule_config": {
            "flat_slope_threshold_deg": cfg.get("flat_slope_threshold_deg", 5.0),
            "plain_relief_threshold_m": cfg.get("plain_relief_threshold_m", 30.0),
        },
    }


def log_to_mlflow(
    cfg: Dict[str, Any],
    run_meta: Dict[str, Any],
    stage1_info: Dict[str, Any],
    stage2_info: Dict[str, Any],
    eval_info: Dict[str, Any],
):
    experiment_name = cfg.get("experiment_name", "forest_agent_dev")
    run_name = cfg.get("run_name", Path(cfg["output_dir"]).name)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        for k, v in run_meta.items():
            if k == "terrain_info" and isinstance(v, dict):
                for tk, tv in v.items():
                    mlflow.log_param(f"terrain.{tk}", tv if tv is not None else "")
            else:
                mlflow.log_param(k, v if v is not None else "")

        metrics = eval_info["metrics"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        artifact_candidates = [
            stage1_info.get("m_sem_tif"),
            stage1_info.get("m_sem_png"),
            stage2_info.get("y_inst_tif"),
            stage2_info.get("y_inst_shp"),
            stage2_info.get("y_inst_color_png"),
            eval_info.get("metrics_json"),
            eval_info.get("details_csv"),
        ]

        if stage2_info.get("y_inst_shp"):
            shp = Path(stage2_info["y_inst_shp"])
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
                p = shp.with_suffix(ext)
                if p.exists():
                    artifact_candidates.append(str(p))

        for p in artifact_candidates:
            if p and Path(p).exists():
                try:
                    mlflow.log_artifact(str(p))
                except Exception as e:
                    print(f"[warn] mlflow log_artifact failed for {p}: {e}")


def run_experiment(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg["flat_slope_threshold_deg"] = cfg.get("flat_slope_threshold_deg", 5.0)
    cfg["plain_relief_threshold_m"] = cfg.get("plain_relief_threshold_m", 30.0)
    cfg["terrain_landform_field"] = cfg.get("terrain_landform_field", "landform_type")
    cfg["terrain_slope_class_field"] = cfg.get("terrain_slope_class_field", "slope_class")
    cfg["terrain_aspect_class_field"] = cfg.get("terrain_aspect_class_field", "aspect_class")
    cfg["terrain_slope_position_field"] = cfg.get("terrain_slope_position_field", "slope_position_class")

    required_keys = [
        "input_image",
        "output_dir",
        "metrics_json",
        "details_csv",
        "xiaoban_shp",
        "xiaoban_id_field",
        "tree_count_field",
        "crown_field",
        "closure_field",
        "area_ha_field",
        "stage1_script",
        "stage2_script",
        "conda_sh",
        "conda_env",
        "work_dir",
        "diam_list",
        "tile",
        "overlap",
        "tile_overlap",
        "bsize",
        "augment",
        "iou_merge_thr",
    ]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")

    if int(cfg["bsize"]) != 256:
        raise ValueError(f"Unsafe bsize={cfg['bsize']}. Must be fixed to 256.")

    terrain_info = prepare_terrain_inputs_from_cfg(cfg)

    stage1_info = run_stage1(cfg)
    stage2_info = run_stage2(cfg, stage1_info["m_sem_tif"])
    eval_info = run_evaluation(cfg, stage2_info["y_inst_shp"], terrain_info)

    run_meta = collect_run_metadata(cfg, terrain_info)

    try:
        log_to_mlflow(
            cfg=cfg,
            run_meta=run_meta,
            stage1_info=stage1_info,
            stage2_info=stage2_info,
            eval_info=eval_info,
        )
    except Exception as e:
        print(f"[warn] MLflow logging failed: {e}")

    summary = {
        "config_path": config_path,
        "run_meta": run_meta,
        "stage1": stage1_info,
        "stage2": stage2_info,
        "evaluation": {
            "metrics_json": eval_info["metrics_json"],
            "details_csv": eval_info["details_csv"],
        },
        "metrics": eval_info["metrics"],
    }

    summary_json = str(Path(cfg["metrics_json"]).resolve().parent / "run_experiment_summary.json")
    save_json(summary, summary_json)

    print(f"[runner] summary saved to: {summary_json}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()