from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config_builder import load_yaml
from agent.local_refine import run_local_refinement
from geo_layer.context_object import build_spatial_context_object_from_config
from geo_layer.spatial_context import prepare_spatial_context
from tools.process_runner import run_streaming


# =========================
# basic io utils
# =========================
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str | Path):
    path = Path(path)
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Dict[str, Any], path: str | Path):
    import yaml

    path = Path(path)
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def run_subprocess(cmd, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    res = run_streaming(cmd, cwd=cwd, env=env)

    return {
        "cmd": cmd,
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


def build_pipeline_layout(pipeline_root: Path) -> Dict[str, Path]:
    intermediate = pipeline_root / "intermediate"
    layout = {
        "intermediate": intermediate,
        "baseline": intermediate / "baseline",
        "agent": intermediate / "agent",
        "optuna": intermediate / "optuna",
        "local_refine": intermediate / "local_refine",
        "finetune": intermediate / "finetune",
    }
    for path in layout.values():
        ensure_dir(path)
    return layout


def build_stage_env(
    *,
    output_root: Path,
    generated_config_dir: Optional[Path] = None,
    agent_out: Optional[Path] = None,
    optuna_trial_summary_dir: Optional[Path] = None,
    local_refine_root: Optional[Path] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["FOREST_AGENT_OUTPUT_ROOT"] = str(output_root)
    if generated_config_dir is not None:
        env["FOREST_AGENT_GENERATED_CONFIG_DIR"] = str(generated_config_dir)
    if agent_out is not None:
        env["FOREST_AGENT_AGENT_OUT"] = str(agent_out)
    if optuna_trial_summary_dir is not None:
        env["FOREST_AGENT_OPTUNA_TRIAL_SUMMARY_DIR"] = str(optuna_trial_summary_dir)
    if local_refine_root is not None:
        env["FOREST_AGENT_LOCAL_REFINE_ROOT"] = str(local_refine_root)
    return env


def copy_path(src: str | Path, dst: str | Path) -> Optional[str]:
    src_path = Path(src)
    if not src_path.exists():
        return None

    dst_path = Path(dst)
    ensure_parent(dst_path)
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def remove_path(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return True


def copy_vector_dataset(src: str | Path, dst: str | Path) -> list[str]:
    src_path = Path(src)
    if not src_path.exists():
        return []

    copied: list[str] = []
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        cand = src_path.with_suffix(ext)
        if cand.exists():
            out = Path(dst).with_suffix(ext)
            ensure_parent(out)
            shutil.copy2(cand, out)
            copied.append(str(out))
    return copied


def remove_vector_dataset(path: str | Path) -> None:
    base = Path(path)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"]:
        remove_path(base.with_suffix(ext))


def get_outputs_root_from_pipeline_root(pipeline_root: Path) -> Path:
    parent = pipeline_root.parent
    if parent.name == "pipeline":
        return parent.parent
    return parent


def get_final_root_from_pipeline_root(pipeline_root: Path) -> Path:
    outputs_root = get_outputs_root_from_pipeline_root(pipeline_root)
    pipeline_name = pipeline_root.name
    if pipeline_name.startswith("pipeline_"):
        final_name = f"final_{pipeline_name[len('pipeline_'):]}"
    else:
        final_name = f"final_{pipeline_name}"
    return outputs_root / final_name


def resolve_pipeline_artifact_path(path_str: Optional[str], pipeline_root: Path) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    legacy_prefix = f"/home/xth/forest_agent_project/outputs/pipeline/{pipeline_root.name}/"
    new_prefix = f"{str(pipeline_root)}/"
    if path_str.startswith(legacy_prefix):
        remapped = Path(path_str.replace(legacy_prefix, new_prefix, 1))
        if remapped.exists():
            return remapped

    return None


def render_vector_preview(src: str | Path, dst: str | Path) -> Optional[str]:
    src_path = Path(src)
    if not src_path.exists():
        return None

    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except Exception:
        return None

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        return None

    dst_path = Path(dst)
    ensure_parent(dst_path)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    gdf.boundary.plot(ax=ax, linewidth=0.3, color="#0b5d1e")
    gdf.plot(ax=ax, linewidth=0, color="#7ccf7a", alpha=0.75)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0)
    fig.savefig(dst_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(dst_path)


def build_semantic_union(src: str | Path, dst: str | Path) -> list[str]:
    src_path = Path(src)
    if not src_path.exists():
        return []

    try:
        import geopandas as gpd
    except Exception:
        return []

    gdf = gpd.read_file(src_path)
    if gdf.empty:
        return []

    union_geom = gdf.geometry.union_all()
    union_gdf = gpd.GeoDataFrame({"class": [1]}, geometry=[union_geom], crs=gdf.crs)
    union_gdf = union_gdf.explode(index_parts=False).reset_index(drop=True)
    union_gdf = union_gdf[~union_gdf.geometry.is_empty].copy()
    if union_gdf.empty:
        return []

    out_path = Path(dst)
    ensure_parent(out_path)
    union_gdf.to_file(out_path)
    return [str(p) for p in out_path.parent.glob(f"{out_path.stem}.*")]


def _safe_name(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "run"


def resolve_final_round_summary_json(summary: Dict[str, Any]) -> Optional[str]:
    final_artifacts = summary.get("final_artifacts", {})

    finetune_summary_json = final_artifacts.get("finetune_summary_json")
    if finetune_summary_json:
        rerun_summary = Path(finetune_summary_json).parent / "rerun_after_finetune" / "run_experiment_summary.json"
        if rerun_summary.exists():
            return str(rerun_summary)

    baseline_stage = summary.get("stages", {}).get("baseline", {})
    run_summary_json = baseline_stage.get("run_summary_json")
    if run_summary_json and Path(run_summary_json).exists():
        return str(run_summary_json)

    return None


def score_run_summary(run_summary: Dict[str, Any]) -> Optional[float]:
    metrics = run_summary.get("metrics") or {}
    tree_ratio = metrics.get("tree_count_error_ratio")
    crown_ratio = metrics.get("mean_crown_width_error_ratio")
    closure_abs = metrics.get("closure_error_abs")
    if tree_ratio is None or crown_ratio is None or closure_abs is None:
        return None
    try:
        return float(tree_ratio) + float(crown_ratio) + float(closure_abs)
    except Exception:
        return None


def resolve_best_round_summary_json(pipeline_root: Path) -> Optional[str]:
    best_path: Optional[Path] = None
    best_score: Optional[float] = None
    for summary_path in sorted((pipeline_root / "intermediate").rglob("run_experiment_summary.json")):
        try:
            run_summary = load_json(summary_path)
        except Exception:
            continue
        score = score_run_summary(run_summary)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_path = summary_path
    return str(best_path) if best_path else None


def build_round_selection(summary: Dict[str, Any], pipeline_root: Path) -> list[Dict[str, Any]]:
    selected: list[Dict[str, Any]] = []
    seen: set[str] = set()

    for label, summary_json in [
        ("final", resolve_final_round_summary_json(summary)),
        ("best", resolve_best_round_summary_json(pipeline_root)),
    ]:
        if not summary_json or summary_json in seen or not Path(summary_json).exists():
            continue
        try:
            run_summary = load_json(summary_json)
        except Exception:
            continue
        run_name = str(run_summary.get("run_name") or Path(summary_json).parent.name)
        selected.append(
            {
                "label": label,
                "run_name": run_name,
                "summary_json": summary_json,
                "score": score_run_summary(run_summary),
            }
        )
        seen.add(summary_json)

    return selected


def enrich_summary_with_round_selection(summary: Dict[str, Any], pipeline_root: Path) -> None:
    selected = build_round_selection(summary, pipeline_root)
    best_summary_json = resolve_best_round_summary_json(pipeline_root)
    summary["selected_rounds"] = selected
    summary["final_round"] = next((item for item in selected if item["label"] == "final"), None)
    best_round = next((item for item in selected if item["label"] == "best"), None)
    if best_round is None and best_summary_json and summary.get("final_round"):
        final_round = summary["final_round"]
        if final_round.get("summary_json") == best_summary_json:
            best_round = {
                **final_round,
                "label": "best",
            }
    summary["best_round"] = best_round


VECTOR_DATASET_EXTS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"}


def prune_dir_children(
    root: Path,
    *,
    keep_files: Optional[set[str]] = None,
    keep_dirs: Optional[set[str]] = None,
    keep_vector_stems: Optional[set[str]] = None,
) -> None:
    if not root.exists():
        return
    keep_files = keep_files or set()
    keep_dirs = keep_dirs or set()
    keep_vector_stems = keep_vector_stems or set()

    for child in root.iterdir():
        if child.is_dir():
            if child.name in keep_dirs:
                continue
            remove_path(child)
            continue

        if child.name in keep_files:
            continue
        if child.suffix.lower() in VECTOR_DATASET_EXTS and child.stem in keep_vector_stems:
            continue
        remove_path(child)


def apply_minimal_retention(pipeline_root: Path, summary: Dict[str, Any]) -> None:
    intermediate = pipeline_root / "intermediate"
    if not intermediate.exists():
        return

    selected = build_round_selection(summary, pipeline_root)
    selected_summary_paths = {
        str(Path(item["summary_json"]).resolve())
        for item in selected
        if item.get("summary_json")
    }

    # baseline: keep only the evaluation anchors and the baseline instance shp.
    baseline_root = intermediate / "baseline"
    prune_dir_children(baseline_root, keep_dirs={"evaluation", "seg_output"})
    prune_dir_children(
        baseline_root / "evaluation",
        keep_files={"metrics.json", "details.csv", "run_experiment_summary.json"},
    )
    prune_dir_children(
        baseline_root / "seg_output",
        keep_vector_stems={"Y_inst"},
    )

    # agent: only keep the final parameter summary.
    agent_root = intermediate / "agent"
    prune_dir_children(agent_root, keep_files={"final_summary.json"})

    # optuna: only keep the best jsons needed for later reruns.
    prune_dir_children(intermediate / "optuna" / "single", keep_files={"optuna_single_best.json"})
    prune_dir_children(intermediate / "optuna" / "multi", keep_files={"optuna_multi_best.json"})

    # local_refine: keep only the merged outputs from the latest refine run.
    local_root = intermediate / "local_refine"
    local_stage = summary.get("stages", {}).get("local_refine", {}) or {}
    merged_shp = local_stage.get("merged_shp")
    local_run_root = Path(merged_shp).parent if merged_shp else None
    if local_run_root and local_run_root.exists():
        prune_dir_children(local_root, keep_dirs={local_run_root.name})
        prune_dir_children(
            local_run_root,
            keep_files={
                "merged_metrics.json",
                "merged_details.csv",
                "local_refine_summary.json",
                "refine_compare_summary.json",
            },
            keep_vector_stems={"merged_global_local_Y_inst"},
        )

    # finetune: keep the summary, best ckpt, lightweight finetuned config, and rerun outputs.
    finetune_root = intermediate / "finetune"
    prune_dir_children(
        finetune_root,
        keep_files={"finetune_pipeline_summary.json"},
        keep_dirs={"training", "finetuned_infer", "rerun_after_finetune", "compare"},
    )

    train_summary_json = finetune_root / "training" / "train_summary.json"
    best_ckpt_name: Optional[str] = None
    finetune_summary_json = finetune_root / "finetune_pipeline_summary.json"
    if finetune_summary_json.exists():
        try:
            finetune_summary = load_json(finetune_summary_json)
            for step in finetune_summary.get("steps", []):
                if step.get("step") == "train_stage1_light" and step.get("ckpt"):
                    best_ckpt_name = Path(step["ckpt"]).name
                    break
        except Exception:
            pass

    training_root = finetune_root / "training"
    if training_root.exists():
        for child in training_root.iterdir():
            if child.name == "train_summary.json":
                continue
            if child.name != "external_trainer":
                remove_path(child)
        external_root = training_root / "external_trainer"
        if external_root.exists():
            logs_root = external_root / "logs"
            if logs_root.exists():
                for version_dir in logs_root.iterdir():
                    if not version_dir.is_dir():
                        remove_path(version_dir)
                        continue
                    prune_dir_children(
                        version_dir,
                        keep_files={"hparams.yaml", "metrics.csv", "pipeline_config.yaml"},
                        keep_dirs={"checkpoints"},
                    )
                    ckpt_dir = version_dir / "checkpoints"
                    if ckpt_dir.exists():
                        keep_files = {best_ckpt_name} if best_ckpt_name else set()
                        prune_dir_children(ckpt_dir, keep_files=keep_files)

    prune_dir_children(
        finetune_root / "finetuned_infer",
        keep_files={"exp_finetuned.yaml", "integration_summary.json"},
    )
    prune_dir_children(
        finetune_root / "rerun_after_finetune",
        keep_files={"metrics.json", "details.csv", "run_experiment_summary.json"},
        keep_vector_stems={"Y_inst"},
    )
    prune_dir_children(
        finetune_root / "compare",
        keep_files={"finetune_gain_summary.json"},
    )

    # Preserve any selected non-baseline/non-finetune run summaries for future republish.
    for summary_path in sorted(intermediate.rglob("run_experiment_summary.json")):
        if str(summary_path.resolve()) in selected_summary_paths:
            continue
        baseline_summary = baseline_root / "evaluation" / "run_experiment_summary.json"
        finetune_summary = finetune_root / "rerun_after_finetune" / "run_experiment_summary.json"
        if summary_path == baseline_summary or summary_path == finetune_summary:
            continue
        remove_path(summary_path)


def publish_pipeline_user_view(pipeline_root: Path, summary: Dict[str, Any]) -> None:
    final_root = get_final_root_from_pipeline_root(pipeline_root)
    if final_root.exists():
        shutil.rmtree(final_root)
    ensure_dir(final_root)
    selected = build_round_selection(summary, pipeline_root)
    for item in selected:
        summary_path = Path(item["summary_json"])
        try:
            run_summary = load_json(summary_path)
        except Exception:
            continue

        inst_shp = resolve_pipeline_artifact_path(
            run_summary.get("merged_inst_shp") or run_summary.get("stage2", {}).get("y_inst_shp"),
            pipeline_root,
        )
        if not inst_shp:
            continue

        run_name = _safe_name(run_summary.get("run_name") or summary_path.parent.name)
        round_root = final_root / run_name
        ensure_dir(round_root)

        copy_vector_dataset(inst_shp, round_root / "merged_Y_inst.shp")
        render_vector_preview(inst_shp, round_root / "merged_Y_inst_color.png")
        build_semantic_union(inst_shp, round_root / "M_sem.shp")
        render_vector_preview(round_root / "M_sem.shp", round_root / "M_sem.png")

        try:
            from reporting.experiment_report import build_experiment_report

            build_experiment_report(run_summary, round_root / "report.md")
        except Exception:
            pass


# =========================
# parse helpers
# =========================
def parse_best_params_from_agent_summary(agent_summary_json: str) -> Dict[str, Any]:
    data = load_json(agent_summary_json)

    if "best_params" in data and isinstance(data["best_params"], dict):
        return data["best_params"]

    if "final_params" in data and isinstance(data["final_params"], dict):
        return data["final_params"]

    raise ValueError(f"Cannot parse params from agent summary: {agent_summary_json}")


def parse_best_params_from_optuna_best(optuna_best_json: str) -> Dict[str, Any]:
    data = load_json(optuna_best_json)

    if "best_params" in data and isinstance(data["best_params"], dict):
        return data["best_params"]

    if "representative_params" in data and isinstance(data["representative_params"], dict):
        return data["representative_params"]

    if "params" in data and isinstance(data["params"], dict):
        return data["params"]

    raise ValueError(f"Cannot parse params from optuna best json: {optuna_best_json}")


def get_params_from_sources(
    best_params_json: Optional[str],
    agent_summary_json: Optional[str],
    optuna_best_json: Optional[str],
) -> Optional[Dict[str, Any]]:
    if best_params_json:
        return json.loads(best_params_json)

    if optuna_best_json:
        return parse_best_params_from_optuna_best(optuna_best_json)

    if agent_summary_json:
        return parse_best_params_from_agent_summary(agent_summary_json)

    return None


# =========================
# validation / precheck
# =========================
def assert_exists(path_str: Optional[str], name: str):
    if not path_str:
        raise ValueError(f"{name} is empty")
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")


def assert_file(path_str: Optional[str], name: str):
    assert_exists(path_str, name)
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"{name} is not a file: {p}")


def assert_dir(path_str: Optional[str], name: str):
    assert_exists(path_str, name)
    p = Path(path_str)
    if not p.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {p}")


def assert_vector_input_complete(vector_path: Optional[str], name: str = "xiaoban_shp"):
    """
    兼容多种矢量格式：
    - .shp: 要求 .shp/.dbf/.shx sidecar 完整
    - .gpkg/.geojson/.json/.fgb: 只要求文件存在
    """
    assert_file(vector_path, name)
    vec = Path(vector_path)
    suffix = vec.suffix.lower()

    if suffix == ".shp":
        stem = vec.with_suffix("")
        required = [
            stem.with_suffix(".shp"),
            stem.with_suffix(".dbf"),
            stem.with_suffix(".shx"),
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{name} incomplete, missing sidecar files: {missing}")
        return

    if suffix in {".gpkg", ".geojson", ".json", ".fgb"}:
        return

    raise ValueError(
        f"Unsupported vector format for {name}: {vec}. "
        "Please provide .shp/.gpkg/.geojson/.json/.fgb"
    )


def assert_env_var(name: str):
    val = os.getenv(name, "").strip()
    if not val:
        raise EnvironmentError(f"Required environment variable is not set: {name}")


def run_precheck(
    *,
    args: argparse.Namespace,
    runtime_base_config: str,
) -> Dict[str, Any]:
    cfg = load_yaml(runtime_base_config)

    assert_file(args.base_config, "base_config")
    assert_file(runtime_base_config, "runtime_base_config")

    assert_file(cfg.get("input_image"), "input_image")
    assert_vector_input_complete(cfg.get("xiaoban_shp"), "xiaoban_shp")

    if cfg.get("dem_tif"):
        assert_file(cfg.get("dem_tif"), "dem_tif")
    if cfg.get("slope_tif"):
        assert_file(cfg.get("slope_tif"), "slope_tif")
    if cfg.get("aspect_tif"):
        assert_file(cfg.get("aspect_tif"), "aspect_tif")

    output_dir = cfg.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir missing in config")
    ensure_dir(Path(output_dir))

    if args.run_agent:
        assert_env_var("ARK_API_KEY")
        assert_env_var("ARK_BASE_URL")
        assert_env_var("ARK_MODEL")

    if cfg.get("work_dir"):
        assert_dir(cfg.get("work_dir"), "work_dir")
    if cfg.get("conda_sh"):
        assert_file(cfg.get("conda_sh"), "conda_sh")

    if args.run_optuna_single or args.run_optuna_multi:
        if args.optuna_storage and args.optuna_storage.startswith("sqlite:///"):
            db_path = args.optuna_storage.replace("sqlite:///", "/", 1)
            db_parent = Path(db_path).parent
            ensure_dir(db_parent)

    return {
        "checked": True,
        "runtime_base_config": runtime_base_config,
        "input_image": cfg.get("input_image"),
        "xiaoban_shp": cfg.get("xiaoban_shp"),
        "output_dir": cfg.get("output_dir"),
        "terrain": {
            "dem_tif": cfg.get("dem_tif"),
            "slope_tif": cfg.get("slope_tif"),
            "aspect_tif": cfg.get("aspect_tif"),
        },
        "spatial_context": {
            "enabled": bool(getattr(args, "auto_spatial_context", False)),
            "summary_json": cfg.get("spatial_context_summary_json"),
        },
        "agent_env": {
            "ARK_API_KEY": bool(os.getenv("ARK_API_KEY", "").strip()) if args.run_agent else None,
            "ARK_BASE_URL": bool(os.getenv("ARK_BASE_URL", "").strip()) if args.run_agent else None,
            "ARK_MODEL": bool(os.getenv("ARK_MODEL", "").strip()) if args.run_agent else None,
        },
        "optuna": {
            "study_name": args.optuna_study_name,
            "storage": args.optuna_storage,
            "resume": args.optuna_resume,
        },
    }


# =========================
# runtime config builder
# =========================
def apply_cli_terrain_overrides_to_config(
    base_config_path: str,
    dem_tif: Optional[str],
    slope_tif: Optional[str],
    aspect_tif: Optional[str],
    pipeline_root: Path,
) -> str:
    if not dem_tif and not slope_tif and not aspect_tif:
        return base_config_path

    cfg = load_yaml(base_config_path)

    if dem_tif is not None:
        cfg["dem_tif"] = dem_tif
    if slope_tif is not None:
        cfg["slope_tif"] = slope_tif
    if aspect_tif is not None:
        cfg["aspect_tif"] = aspect_tif

    temp_cfg = pipeline_root / "runtime_base_config.yaml"
    save_yaml(cfg, temp_cfg)
    return str(temp_cfg)


def maybe_prepare_spatial_context_runtime_config(
    runtime_base_config: str,
    pipeline_root: Path,
    enable_auto_spatial_context: bool,
) -> str:
    """
    根据 input_image(DOM) 的范围，自动裁出同范围的：
    - dem_tif
    - slope_tif
    - aspect_tif
    - xiaoban_shp

    然后把这些路径写回 runtime config，供后续五层流程继续使用。
    """
    if not enable_auto_spatial_context:
        return runtime_base_config

    cfg = load_yaml(runtime_base_config)

    input_image = cfg.get("input_image")
    dem_tif = cfg.get("dem_tif")
    xiaoban_shp = cfg.get("xiaoban_shp")
    xiaoban_id_field = cfg.get("xiaoban_id_field")

    if not input_image:
        raise ValueError("auto_spatial_context=True, but input_image is missing in config.")

    if not dem_tif and not xiaoban_shp:
        # 没有 DEM / xiaoban 可裁，直接返回
        return runtime_base_config

    context_dir = pipeline_root / "spatial_context"
    ensure_dir(context_dir)

    result = prepare_spatial_context(
        dom_tif=input_image,
        dem_tif=dem_tif,
        xiaoban_shp=xiaoban_shp,
        out_dir=context_dir,
        xiaoban_id_field=xiaoban_id_field,
        tree_count_field=cfg.get("tree_count_field"),
        crown_field=cfg.get("crown_field"),
        closure_field=cfg.get("closure_field"),
        area_ha_field=cfg.get("area_ha_field"),
        density_field=cfg.get("density_field"),
    )

    if result.get("dem_tif"):
        cfg["dem_tif"] = result["dem_tif"]
    if result.get("slope_tif"):
        cfg["slope_tif"] = result["slope_tif"]
    if result.get("aspect_tif"):
        cfg["aspect_tif"] = result["aspect_tif"]
    if result.get("landform_tif"):
        cfg["landform_tif"] = result["landform_tif"]
    if result.get("slope_position_tif"):
        cfg["slope_position_tif"] = result["slope_position_tif"]
    if result.get("xiaoban_shp"):
        cfg["xiaoban_shp"] = result["xiaoban_shp"]

    cfg["terrain_landform_field"] = "landform_type"
    cfg["terrain_slope_class_field"] = "slope_class"
    cfg["terrain_aspect_class_field"] = "aspect_class"
    cfg["terrain_slope_position_field"] = "slope_position_class"
    cfg["flat_slope_threshold_deg"] = cfg.get("flat_slope_threshold_deg", 5.0)
    cfg["plain_relief_threshold_m"] = cfg.get("plain_relief_threshold_m", 30.0)
    cfg["spatial_context_summary_json"] = result.get("summary_json")

    temp_cfg = pipeline_root / "runtime_base_config.yaml"
    save_yaml(cfg, temp_cfg)
    return str(temp_cfg)


def sync_spatial_context_to_config(
    runtime_base_config: str,
    pipeline_root: Path,
) -> tuple[str, Dict[str, Any]]:
    cfg = load_yaml(runtime_base_config)
    context_obj = build_spatial_context_object_from_config(cfg)

    cfg["spatial_context_object"] = context_obj

    context_json = pipeline_root / "spatial_context_object.json"
    save_json(context_obj, context_json)
    cfg["spatial_context_object_json"] = str(context_json)

    temp_cfg = pipeline_root / "runtime_base_config.yaml"
    save_yaml(cfg, temp_cfg)
    return str(temp_cfg), context_obj


def normalize_runtime_outputs_to_pipeline(
    runtime_base_config: str,
    pipeline_root: Path,
) -> str:
    cfg = load_yaml(runtime_base_config)
    run_name = str(cfg.get("run_name", Path(pipeline_root).name))

    baseline_dir = pipeline_root / "intermediate" / "baseline"
    seg_output_dir = baseline_dir / "seg_output"
    eval_dir = baseline_dir / "evaluation"

    cfg["output_dir"] = str(seg_output_dir)
    cfg["metrics_json"] = str(eval_dir / "metrics.json")
    cfg["details_csv"] = str(eval_dir / "details.csv")
    cfg["run_name"] = run_name
    cfg["keep_stage1_artifacts"] = True

    temp_cfg = pipeline_root / "runtime_base_config.yaml"
    save_yaml(cfg, temp_cfg)
    return str(temp_cfg)


def resolve_runtime_base_config(
    *,
    args: argparse.Namespace,
    pipeline_root: Path,
) -> str:
    existing_runtime = pipeline_root / "runtime_base_config.yaml"

    if args.resume and existing_runtime.exists():
        runtime_base_config = str(existing_runtime)
    else:
        runtime_base_config = apply_cli_terrain_overrides_to_config(
            base_config_path=args.base_config,
            dem_tif=args.dem_tif,
            slope_tif=args.slope_tif,
            aspect_tif=args.aspect_tif,
            pipeline_root=pipeline_root,
        )

        runtime_base_config = maybe_prepare_spatial_context_runtime_config(
            runtime_base_config=runtime_base_config,
            pipeline_root=pipeline_root,
            enable_auto_spatial_context=args.auto_spatial_context,
        )

    runtime_base_config = normalize_runtime_outputs_to_pipeline(
        runtime_base_config=runtime_base_config,
        pipeline_root=pipeline_root,
    )

    return runtime_base_config


# =========================
# stage runners
# =========================
def run_baseline_stage(
    base_config_path: str,
) -> Dict[str, Any]:
    base_cfg = load_yaml(base_config_path)

    cmd = [
        "python",
        "-m",
        "scripts.run_zstreeseg_experiment",
        "--config",
        base_config_path,
    ]
    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project")
    if res["returncode"] != 0:
        raise RuntimeError(f"baseline run failed:\n{res['stderr']}")

    metrics_json = base_cfg["metrics_json"]
    details_csv = base_cfg["details_csv"]
    inst_shp = str(Path(base_cfg["output_dir"]) / "Y_inst.shp")
    run_summary_json = str(Path(metrics_json).resolve().parent / "run_experiment_summary.json")

    if not Path(metrics_json).exists():
        raise FileNotFoundError(f"baseline metrics.json not found: {metrics_json}")
    if not Path(details_csv).exists():
        raise FileNotFoundError(f"baseline details.csv not found: {details_csv}")
    if not Path(inst_shp).exists():
        raise FileNotFoundError(f"baseline Y_inst.shp not found: {inst_shp}")

    run_summary = None
    if Path(run_summary_json).exists():
        try:
            run_summary = load_json(run_summary_json)
        except Exception:
            run_summary = None

    return {
        "baseline_run": res,
        "metrics_json": metrics_json,
        "details_csv": details_csv,
        "inst_shp": inst_shp,
        "run_summary_json": run_summary_json if Path(run_summary_json).exists() else None,
        "run_summary": run_summary,
    }


def run_agent_stage(
    base_config_path: str,
    stage_root: Path,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    generated_dir = stage_root / "generated_configs"
    agent_out = stage_root / "final_summary.json"
    env = build_stage_env(
        output_root=stage_root / "runs",
        generated_config_dir=generated_dir,
        agent_out=agent_out,
    )
    cmd = [
        "python",
        "-m",
        "agent.graph",
        "--base_config",
        base_config_path,
        "--max_rounds",
        str(max_rounds),
        "--out_json",
        str(agent_out),
    ]
    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project", env=env)
    if res["returncode"] != 0:
        raise RuntimeError(f"agent stage failed:\n{res['stderr']}")

    summary_json = str(agent_out)
    if not Path(summary_json).exists():
        raise FileNotFoundError(f"agent final_summary.json not found: {summary_json}")

    return {
        "agent_run": res,
        "agent_summary_json": summary_json,
    }


def run_optuna_single_stage(
    base_config_path: str,
    stage_root: Path,
    n_trials: int = 2,
    agent_hint_json: Optional[str] = None,
    spatial_context_json: Optional[str] = None,
    out_best_json: Optional[str] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    generated_dir = stage_root / "generated_configs"
    trial_summary_dir = stage_root / "trials"
    out_best_json = out_best_json or str(stage_root / "optuna_single_best.json")
    storage = storage or f"sqlite:///{stage_root / 'optuna_single.db'}"
    env = build_stage_env(
        output_root=stage_root / "runs",
        generated_config_dir=generated_dir,
        optuna_trial_summary_dir=trial_summary_dir,
    )
    cmd = [
        "python",
        "-m",
        "optuna_layer.search",
        "--base_config",
        base_config_path,
        "--n_trials",
        str(n_trials),
        "--out_best_json",
        out_best_json,
    ]

    if agent_hint_json:
        cmd.extend(["--agent_hint_json", agent_hint_json])
    if spatial_context_json:
        cmd.extend(["--spatial_context_json", spatial_context_json])
    if study_name:
        cmd.extend(["--study_name", study_name])
    if storage:
        cmd.extend(["--storage", storage])
    if resume:
        cmd.append("--resume")

    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project", env=env)
    if res["returncode"] != 0:
        raise RuntimeError(f"optuna single failed:\n{res['stderr']}")

    if not Path(out_best_json).exists():
        raise FileNotFoundError(f"optuna single best json not found: {out_best_json}")

    best_data = load_json(out_best_json)

    return {
        "optuna_single_run": res,
        "optuna_single_best_json": out_best_json,
        "study_name": best_data.get("study_name"),
        "storage": best_data.get("storage"),
    }


def run_optuna_multi_stage(
    base_config_path: str,
    stage_root: Path,
    n_trials: int = 2,
    agent_hint_json: Optional[str] = None,
    spatial_context_json: Optional[str] = None,
    out_best_json: Optional[str] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    generated_dir = stage_root / "generated_configs"
    trial_summary_dir = stage_root / "trials"
    out_best_json = out_best_json or str(stage_root / "optuna_multi_best.json")
    storage = storage or f"sqlite:///{stage_root / 'optuna_multi.db'}"
    env = build_stage_env(
        output_root=stage_root / "runs",
        generated_config_dir=generated_dir,
        optuna_trial_summary_dir=trial_summary_dir,
    )
    cmd = [
        "python",
        "-m",
        "optuna_layer.search_multi",
        "--base_config",
        base_config_path,
        "--n_trials",
        str(n_trials),
        "--out_best_json",
        out_best_json,
    ]

    if agent_hint_json:
        cmd.extend(["--agent_hint_json", agent_hint_json])
    if spatial_context_json:
        cmd.extend(["--spatial_context_json", spatial_context_json])
    if study_name:
        cmd.extend(["--study_name", study_name])
    if storage:
        cmd.extend(["--storage", storage])
    if resume:
        cmd.append("--resume")

    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project", env=env)
    if res["returncode"] != 0:
        raise RuntimeError(f"optuna multi failed:\n{res['stderr']}")

    if not Path(out_best_json).exists():
        raise FileNotFoundError(f"optuna multi best json not found: {out_best_json}")

    best_data = load_json(out_best_json)

    return {
        "optuna_multi_run": res,
        "optuna_multi_best_json": out_best_json,
        "study_name": best_data.get("study_name"),
        "storage": best_data.get("storage"),
    }

def run_local_refine_stage(
    base_config_path: str,
    stage_root: Path,
    global_details_csv: str,
    global_inst_shp: str,
    params: Dict[str, Any],
    xiaoban_id_field: str = "XBH",
    top_k: int = 2,
    buffer_m: float = 5.0,
    strategy_mode: str = "auto",
    dem_tif: Optional[str] = None,
    slope_tif: Optional[str] = None,
    aspect_tif: Optional[str] = None,
    landform_tif: Optional[str] = None,
    slope_position_tif: Optional[str] = None,
    flat_slope_threshold_deg: float = 5.0,
    plain_relief_threshold_m: float = 30.0,
) -> Dict[str, Any]:
    """
    执行 local refine 阶段，并返回标准化结果摘要。

    当前职责：
    1. 将 baseline/global 结果与最优参数送入 run_local_refinement
    2. 透传 ROI 级 terrain 输入（DEM / slope / aspect）
    3. 在 stage 输出中补充 terrain 约束规范信息，便于 pipeline summary 统一表达

    说明：
    - 当前主线真正参与 local refine 的 terrain 输入仍主要是 dem_tif / slope_tif / aspect_tif
    - landform_tif / slope_position_tif 先作为保留字段，便于后续扩展，不要求 run_local_refinement 立刻使用
    """

    summary = run_local_refinement(
        base_config_path=base_config_path,
        global_details_csv=global_details_csv,
        global_inst_shp=global_inst_shp,
        best_params=params,
        xiaoban_id_field=xiaoban_id_field,
        top_k=top_k,
        buffer_m=buffer_m,
        strategy_mode=strategy_mode,
        dem_tif=dem_tif,
        slope_tif=slope_tif,
        aspect_tif=aspect_tif,
        local_refine_root=str(stage_root),
    )

    terrain_inputs = {
        "dem_tif": dem_tif,
        "slope_tif": slope_tif,
        "aspect_tif": aspect_tif,
        "landform_tif": landform_tif,
        "slope_position_tif": slope_position_tif,
    }

    terrain_rule_config = {
        "flat_slope_threshold_deg": flat_slope_threshold_deg,
        "plain_relief_threshold_m": plain_relief_threshold_m,
    }

    terrain_constraint_fields = {
        "terrain_landform_field": "landform_type",
        "terrain_slope_class_field": "slope_class",
        "terrain_aspect_class_field": "aspect_class",
        "terrain_slope_position_field": "slope_position_class",
    }

    return {
        "local_refine_summary": summary,
        "merged_shp": summary.get("merged_shp"),
        "merged_metrics_json": summary.get("merged_metrics_json"),
        "merged_details_csv": summary.get("merged_details_csv"),
        "compare_json": summary.get("compare_json"),
        "terrain_inputs": terrain_inputs,
        "terrain_rule_config": terrain_rule_config,
        "terrain_constraint_fields": terrain_constraint_fields,
    }

# =========================
# pipeline state management
# =========================
STAGE_ORDER = [
    "precheck",
    "baseline",
    "agent",
    "optuna_single",
    "optuna_multi",
    "local_refine",
    "finetune",
]


def init_pipeline_state(
    *,
    pipeline_root: Path,
    args: argparse.Namespace,
    runtime_base_config: str,
) -> Dict[str, Any]:
    state_path = pipeline_root / "pipeline_state.json"

    if args.resume and state_path.exists():
        state = load_json(state_path)
        return state

    state = {
        "pipeline_name": pipeline_root.name,
        "pipeline_root": str(pipeline_root),
        "base_config": args.base_config,
        "runtime_base_config": runtime_base_config,
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "updated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": "running",
        "run_flags": {
            "run_baseline": args.run_baseline,
            "run_agent": args.run_agent,
            "run_optuna_single": args.run_optuna_single,
            "run_optuna_multi": args.run_optuna_multi,
            "run_local_refine": args.run_local_refine,
            "run_finetune": args.run_finetune,
        },
        "stages": {
            name: {
                "status": "pending",
                "outputs": {},
                "error": None,
            }
            for name in STAGE_ORDER
        },
        "last_error": None,
    }
    save_pipeline_state(pipeline_root, state)
    return state


def save_pipeline_state(pipeline_root: Path, state: Dict[str, Any]):
    state["updated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(state, pipeline_root / "pipeline_state.json")


def mark_stage_running(state: Dict[str, Any], stage: str):
    state["stages"][stage]["status"] = "running"
    state["stages"][stage]["error"] = None


def mark_stage_success(state: Dict[str, Any], stage: str, outputs: Optional[Dict[str, Any]] = None):
    state["stages"][stage]["status"] = "success"
    state["stages"][stage]["error"] = None
    state["stages"][stage]["outputs"] = outputs or {}


def mark_stage_failed(state: Dict[str, Any], stage: str, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    state["stages"][stage]["status"] = "failed"
    state["stages"][stage]["error"] = tb
    state["last_error"] = {
        "stage": stage,
        "message": str(exc),
        "traceback": tb,
    }
    state["status"] = "failed"


def mark_stage_skipped(state: Dict[str, Any], stage: str, reason: str):
    state["stages"][stage]["status"] = "skipped"
    state["stages"][stage]["error"] = reason


def stage_success(state: Dict[str, Any], stage: str) -> bool:
    return state["stages"][stage]["status"] == "success"


def should_force_rerun(stage: str, force_rerun: Optional[str]) -> bool:
    if not force_rerun:
        return False
    if force_rerun not in STAGE_ORDER:
        return False
    return STAGE_ORDER.index(stage) >= STAGE_ORDER.index(force_rerun)


def execute_stage(
    *,
    state: Dict[str, Any],
    pipeline_root: Path,
    stage_name: str,
    enabled: bool,
    resume: bool,
    force_rerun: Optional[str],
    fn,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        mark_stage_skipped(state, stage_name, "stage disabled by args")
        save_pipeline_state(pipeline_root, state)
        return None

    need_force = should_force_rerun(stage_name, force_rerun)

    if resume and stage_success(state, stage_name) and not need_force:
        print(f"[resume] skip stage={stage_name}, already success")
        return state["stages"][stage_name].get("outputs", {})

    try:
        mark_stage_running(state, stage_name)
        save_pipeline_state(pipeline_root, state)

        outputs = fn()
        outputs = outputs or {}

        mark_stage_success(state, stage_name, outputs)
        state["status"] = "running"
        save_pipeline_state(pipeline_root, state)
        return outputs
    except Exception as e:
        mark_stage_failed(state, stage_name, e)
        save_pipeline_state(pipeline_root, state)
        raise


# =========================
# helpers for summary/state sync
# =========================
def stage_outputs_or_none(state: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
    data = state["stages"].get(stage, {})
    if data.get("status") == "success":
        return data.get("outputs", {})
    return None


def add_baseline_to_summary(summary: Dict[str, Any], baseline_info: Dict[str, Any]):
    summary["stages"]["baseline"] = {
        "metrics_json": baseline_info.get("metrics_json"),
        "details_csv": baseline_info.get("details_csv"),
        "inst_shp": baseline_info.get("inst_shp"),
        "run_summary_json": baseline_info.get("run_summary_json"),
    }
    if baseline_info.get("run_summary"):
        summary["stages"]["baseline"]["run_summary_excerpt"] = {
            "terrain_info": baseline_info["run_summary"].get("run_meta", {}).get("terrain_info"),
            "evaluation": baseline_info["run_summary"].get("evaluation"),
        }


def add_agent_to_summary(summary: Dict[str, Any], agent_info: Dict[str, Any]):
    summary["stages"]["agent"] = {
        "agent_summary_json": agent_info.get("agent_summary_json"),
    }


def add_optuna_single_to_summary(summary: Dict[str, Any], info: Dict[str, Any]):
    summary["stages"]["optuna_single"] = {
        "optuna_single_best_json": info.get("optuna_single_best_json"),
        "study_name": info.get("study_name"),
        "storage": info.get("storage"),
    }


def add_optuna_multi_to_summary(summary: Dict[str, Any], info: Dict[str, Any]):
    summary["stages"]["optuna_multi"] = {
        "optuna_multi_best_json": info.get("optuna_multi_best_json"),
        "study_name": info.get("study_name"),
        "storage": info.get("storage"),
    }


def add_local_refine_to_summary(summary: Dict[str, Any], info: Dict[str, Any]):
    summary["stages"]["local_refine"] = {
        "merged_shp": info.get("merged_shp"),
        "merged_metrics_json": info.get("merged_metrics_json"),
        "merged_details_csv": info.get("merged_details_csv"),
        "compare_json": info.get("compare_json"),
    }


def sync_summary_from_state(summary: Dict[str, Any], state: Dict[str, Any]):
    baseline = stage_outputs_or_none(state, "baseline")
    if baseline:
        add_baseline_to_summary(summary, baseline)

    agent = stage_outputs_or_none(state, "agent")
    if agent:
        add_agent_to_summary(summary, agent)

    optuna_single = stage_outputs_or_none(state, "optuna_single")
    if optuna_single:
        add_optuna_single_to_summary(summary, optuna_single)

    optuna_multi = stage_outputs_or_none(state, "optuna_multi")
    if optuna_multi:
        add_optuna_multi_to_summary(summary, optuna_multi)

    local_refine = stage_outputs_or_none(state, "local_refine")
    if local_refine:
        add_local_refine_to_summary(summary, local_refine)

    finetune = stage_outputs_or_none(state, "finetune")
    if finetune:
        summary["stages"]["finetune"] = {
            "finetune_summary_json": finetune.get("finetune_summary_json"),
            "status": finetune.get("finetune_summary", {}).get("status"),
        }


def save_pipeline_summary(
    *,
    pipeline_root: Path,
    summary: Dict[str, Any],
):
    enrich_summary_with_round_selection(summary, pipeline_root)
    summary_path = pipeline_root / "pipeline_summary.json"
    save_json(summary, summary_path)
    publish_pipeline_user_view(pipeline_root, summary)
    state_path = pipeline_root / "pipeline_state.json"
    if state_path.exists():
        try:
            state = load_json(state_path)
            if state.get("status") == "success":
                apply_minimal_retention(pipeline_root, summary)
                save_json(summary, summary_path)
                publish_pipeline_user_view(pipeline_root, summary)
        except Exception:
            pass
    print(f"[pipeline] summary saved to: {summary_path}")


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_config", required=True)
    parser.add_argument("--global_inst_shp", default=None)

    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--run_agent", action="store_true")
    parser.add_argument("--run_optuna_single", action="store_true")
    parser.add_argument("--run_optuna_multi", action="store_true")
    parser.add_argument("--run_local_refine", action="store_true")
    parser.add_argument("--run_finetune", action="store_true")
    parser.add_argument("--finetune_config", default=None)

    parser.add_argument("--best_params_json", default=None)
    parser.add_argument("--agent_summary_json", default=None)
    parser.add_argument("--optuna_best_json", default=None)

    parser.add_argument("--strategy_mode", default="auto", choices=["auto", "single_params"])
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--buffer_m", type=float, default=5.0)

    parser.add_argument("--agent_max_rounds", type=int, default=3)
    parser.add_argument("--n_trials_single", type=int, default=2)
    parser.add_argument("--n_trials_multi", type=int, default=2)

    # terrain
    parser.add_argument("--dem_tif", default=None)
    parser.add_argument("--slope_tif", default=None)
    parser.add_argument("--aspect_tif", default=None)

    # spatial context
    parser.add_argument(
        "--auto_spatial_context",
        action="store_true",
        help="根据 input_image 的范围自动裁 DEM / 小班 shp，并生成局部 slope/aspect。",
    )

    # pipeline control
    parser.add_argument("--resume", action="store_true", help="resume from existing pipeline_state.json")
    parser.add_argument("--run_dir", default=None, help="existing pipeline output dir for resume/reuse")
    parser.add_argument(
        "--force_rerun",
        default=None,
        choices=STAGE_ORDER,
        help="force rerun this stage and all downstream stages",
    )
    parser.add_argument(
        "--stop_after",
        default=None,
        choices=STAGE_ORDER,
        help="stop after this stage succeeds",
    )

    # optuna controls
    parser.add_argument("--optuna_study_name", default=None)
    parser.add_argument("--optuna_storage", default=None)
    parser.add_argument("--optuna_resume", action="store_true")

    args = parser.parse_args()

    if args.run_dir:
        pipeline_root = Path(args.run_dir)
        ensure_dir(pipeline_root)
        pipeline_name = pipeline_root.name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_name = f"pipeline_{Path(args.base_config).stem}_{stamp}"
        pipeline_root = Path(f"/home/xth/forest_agent_project/outputs/{pipeline_name}")
        ensure_dir(pipeline_root)

    layout = build_pipeline_layout(pipeline_root)

    runtime_base_config = resolve_runtime_base_config(
        args=args,
        pipeline_root=pipeline_root,
    )
    runtime_base_config, spatial_context_object = sync_spatial_context_to_config(
        runtime_base_config=runtime_base_config,
        pipeline_root=pipeline_root,
    )

    base_cfg = load_yaml(runtime_base_config)

    state = init_pipeline_state(
        pipeline_root=pipeline_root,
        args=args,
        runtime_base_config=runtime_base_config,
    )

    summary: Dict[str, Any] = {
        "pipeline_name": pipeline_name,
        "pipeline_root": str(pipeline_root),
        "final_root": str(get_final_root_from_pipeline_root(pipeline_root)),
        "base_config": args.base_config,
        "runtime_base_config": runtime_base_config,
        "run_flags": {
            "run_baseline": args.run_baseline,
            "run_agent": args.run_agent,
            "run_optuna_single": args.run_optuna_single,
            "run_optuna_multi": args.run_optuna_multi,
            "run_local_refine": args.run_local_refine,
            "run_finetune": args.run_finetune,
        },
        "terrain_inputs": spatial_context_object.get("terrain_inputs", {}),
        "spatial_context": {
            "enabled": spatial_context_object.get("spatial_context_enabled", False),
            "summary_json": spatial_context_object.get("spatial_context_summary_json"),
        },
        "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
        "spatial_context_object": spatial_context_object,
        "control": {
            "resume": args.resume,
            "run_dir": args.run_dir,
            "force_rerun": args.force_rerun,
            "stop_after": args.stop_after,
            "optuna_study_name": args.optuna_study_name,
            "optuna_storage": args.optuna_storage,
            "optuna_resume": args.optuna_resume,
        },
        "stages": {},
    }

    current_agent_summary_json = args.agent_summary_json
    current_optuna_best_json = args.optuna_best_json

    current_global_inst_shp = args.global_inst_shp
    current_global_metrics_json = base_cfg.get("metrics_json")
    current_global_details_csv = base_cfg.get("details_csv")

    prev_baseline = stage_outputs_or_none(state, "baseline")
    if prev_baseline:
        current_global_inst_shp = prev_baseline.get("inst_shp", current_global_inst_shp)
        current_global_metrics_json = prev_baseline.get("metrics_json", current_global_metrics_json)
        current_global_details_csv = prev_baseline.get("details_csv", current_global_details_csv)

    prev_agent = stage_outputs_or_none(state, "agent")
    if prev_agent and not current_agent_summary_json:
        current_agent_summary_json = prev_agent.get("agent_summary_json")

    prev_optuna_single = stage_outputs_or_none(state, "optuna_single")
    if prev_optuna_single and not current_optuna_best_json:
        current_optuna_best_json = prev_optuna_single.get("optuna_single_best_json")

    prev_optuna_multi = stage_outputs_or_none(state, "optuna_multi")
    if prev_optuna_multi:
        current_optuna_best_json = prev_optuna_multi.get("optuna_multi_best_json", current_optuna_best_json)

    try:
        # 0) precheck
        precheck_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="precheck",
            enabled=True,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=lambda: run_precheck(
                args=args,
                runtime_base_config=runtime_base_config,
            ),
        )
        summary["stages"]["precheck"] = precheck_info or state["stages"]["precheck"].get("outputs", {})

        if args.stop_after == "precheck":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        # 1) baseline
        baseline_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="baseline",
            enabled=args.run_baseline,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=lambda: run_baseline_stage(
                base_config_path=runtime_base_config,
            ),
        )
        if baseline_info:
            add_baseline_to_summary(summary, baseline_info)
            current_global_inst_shp = baseline_info["inst_shp"]
            current_global_metrics_json = baseline_info["metrics_json"]
            current_global_details_csv = baseline_info["details_csv"]
        else:
            prev = stage_outputs_or_none(state, "baseline")
            if prev:
                add_baseline_to_summary(summary, prev)
                current_global_inst_shp = prev.get("inst_shp", current_global_inst_shp)
                current_global_metrics_json = prev.get("metrics_json", current_global_metrics_json)
                current_global_details_csv = prev.get("details_csv", current_global_details_csv)

        if args.stop_after == "baseline":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        # 2) agent
        agent_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="agent",
            enabled=args.run_agent,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=lambda: run_agent_stage(
                base_config_path=runtime_base_config,
                stage_root=layout["agent"],
                max_rounds=args.agent_max_rounds,
            ),
        )
        if agent_info:
            current_agent_summary_json = agent_info["agent_summary_json"]
            add_agent_to_summary(summary, agent_info)
        else:
            prev = stage_outputs_or_none(state, "agent")
            if prev:
                current_agent_summary_json = prev.get("agent_summary_json", current_agent_summary_json)
                add_agent_to_summary(summary, prev)

        if args.stop_after == "agent":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        # 3) optuna single
        optuna_single_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="optuna_single",
            enabled=args.run_optuna_single,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=lambda: run_optuna_single_stage(
                base_config_path=runtime_base_config,
                stage_root=layout["optuna"] / "single",
                n_trials=args.n_trials_single,
                agent_hint_json=current_agent_summary_json,
                spatial_context_json=base_cfg.get("spatial_context_object_json"),
                study_name=args.optuna_study_name,
                storage=args.optuna_storage,
                resume=args.optuna_resume,
            ),
        )
        if optuna_single_info:
            current_optuna_best_json = optuna_single_info["optuna_single_best_json"]
            add_optuna_single_to_summary(summary, optuna_single_info)
        else:
            prev = stage_outputs_or_none(state, "optuna_single")
            if prev:
                current_optuna_best_json = prev.get("optuna_single_best_json", current_optuna_best_json)
                add_optuna_single_to_summary(summary, prev)

        if args.stop_after == "optuna_single":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        # 4) optuna multi
        optuna_multi_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="optuna_multi",
            enabled=args.run_optuna_multi,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=lambda: run_optuna_multi_stage(
                base_config_path=runtime_base_config,
                stage_root=layout["optuna"] / "multi",
                n_trials=args.n_trials_multi,
                agent_hint_json=current_agent_summary_json,
                spatial_context_json=base_cfg.get("spatial_context_object_json"),
                study_name=args.optuna_study_name,
                storage=args.optuna_storage,
                resume=args.optuna_resume,
            ),
        )
        if optuna_multi_info:
            current_optuna_best_json = optuna_multi_info["optuna_multi_best_json"]
            add_optuna_multi_to_summary(summary, optuna_multi_info)
        else:
            prev = stage_outputs_or_none(state, "optuna_multi")
            if prev:
                current_optuna_best_json = prev.get("optuna_multi_best_json", current_optuna_best_json)
                add_optuna_multi_to_summary(summary, prev)

        if args.stop_after == "optuna_multi":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        # 5) local refine
        def _run_local_refine():
            params = get_params_from_sources(
                best_params_json=args.best_params_json,
                agent_summary_json=current_agent_summary_json,
                optuna_best_json=current_optuna_best_json,
            )
            if params is None:
                raise ValueError(
                    "run_local_refine requires params source. Provide one of: "
                    "--best_params_json / --agent_summary_json / --optuna_best_json "
                    "or run agent/optuna in the same pipeline."
                )

            if current_global_details_csv is None or not Path(current_global_details_csv).exists():
                raise FileNotFoundError(f"global details csv not found: {current_global_details_csv}")

            if current_global_inst_shp is None or not Path(current_global_inst_shp).exists():
                raise FileNotFoundError(
                    f"global inst shp not found: {current_global_inst_shp}. "
                    "Provide --global_inst_shp or run --run_baseline first."
                )

            return run_local_refine_stage(
                base_config_path=runtime_base_config,
                stage_root=layout["local_refine"],
                global_details_csv=current_global_details_csv,
                global_inst_shp=current_global_inst_shp,
                params=params,
                xiaoban_id_field=base_cfg["xiaoban_id_field"],
                top_k=args.top_k,
                buffer_m=args.buffer_m,
                strategy_mode=args.strategy_mode,
                dem_tif=base_cfg.get("dem_tif"),
                slope_tif=base_cfg.get("slope_tif"),
                aspect_tif=base_cfg.get("aspect_tif"),
            )

        local_refine_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="local_refine",
            enabled=args.run_local_refine,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=_run_local_refine,
        )
        if local_refine_info:
            add_local_refine_to_summary(summary, local_refine_info)
        else:
            prev = stage_outputs_or_none(state, "local_refine")
            if prev:
                add_local_refine_to_summary(summary, prev)

        if args.stop_after == "local_refine":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        # 6) finetune
        def _run_finetune():
            if not args.finetune_config:
                raise ValueError("run_finetune requires --finetune_config")

            finetune_cfg = load_yaml(args.finetune_config)
            latest_local_refine = local_refine_info or stage_outputs_or_none(state, "local_refine") or {}

            # 将 finetune 输入显式对齐到当前 pipeline 里最新可用的上游产物，
            # 避免继续读取 finetune 模板里写死的历史 local_refine 路径。
            finetune_cfg["metrics_json"] = (
                latest_local_refine.get("merged_metrics_json")
                or current_global_metrics_json
                or finetune_cfg.get("metrics_json")
            )
            finetune_cfg["details_csv"] = (
                latest_local_refine.get("merged_details_csv")
                or current_global_details_csv
                or finetune_cfg.get("details_csv")
            )
            finetune_cfg["pseudo_inst_shp"] = (
                latest_local_refine.get("merged_shp")
                or current_global_inst_shp
                or finetune_cfg.get("pseudo_inst_shp")
            )

            # 几何/地形输入也对齐 runtime base config，保证与本次 pipeline 的 spatial context 一致。
            for key in [
                "input_image",
                "xiaoban_shp",
                "xiaoban_id_field",
                "tree_count_field",
                "crown_field",
                "closure_field",
                "area_ha_field",
                "density_field",
                "dem_tif",
                "slope_tif",
                "aspect_tif",
                "landform_tif",
                "slope_position_tif",
                "flat_slope_threshold_deg",
                "plain_relief_threshold_m",
            ]:
                if key in base_cfg:
                    finetune_cfg[key] = base_cfg.get(key)

            finetune_cfg["output_dir"] = str(layout["finetune"])

            finetune_cfg["pipeline_run_dir"] = str(pipeline_root)
            finetune_cfg["spatial_context_object_json"] = base_cfg.get("spatial_context_object_json")

            runtime_finetune_config = pipeline_root / "runtime_finetune_config.yaml"
            save_yaml(finetune_cfg, runtime_finetune_config)

            cmd = [
                sys.executable,
                "-m",
                "pipeline.run_finetune_pipeline",
                "--config",
                str(runtime_finetune_config),
            ]
            res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project")
            if res["returncode"] != 0:
                raise RuntimeError(f"finetune stage failed:\n{res['stderr']}")

            finetune_out_dir = Path(finetune_cfg["output_dir"])
            finetune_summary_json = finetune_out_dir / "finetune_pipeline_summary.json"

            if not finetune_summary_json.exists():
                raise FileNotFoundError(f"finetune summary not found: {finetune_summary_json}")

            finetune_summary = load_json(finetune_summary_json)

            return {
                "runtime_finetune_config": str(runtime_finetune_config),
                "finetune_summary_json": str(finetune_summary_json),
                "finetune_summary": finetune_summary,
            }

        finetune_info = execute_stage(
            state=state,
            pipeline_root=pipeline_root,
            stage_name="finetune",
            enabled=args.run_finetune,
            resume=args.resume,
            force_rerun=args.force_rerun,
            fn=_run_finetune,
        )
        if finetune_info:
            summary["stages"]["finetune"] = {
                "finetune_summary_json": finetune_info.get("finetune_summary_json"),
                "status": finetune_info.get("finetune_summary", {}).get("status"),
            }
        else:
            prev = stage_outputs_or_none(state, "finetune")
            if prev:
                summary["stages"]["finetune"] = {
                    "finetune_summary_json": prev.get("finetune_summary_json"),
                    "status": prev.get("finetune_summary", {}).get("status"),
                }

        if args.stop_after == "finetune":
            state["status"] = "stopped"
            save_pipeline_state(pipeline_root, state)
            sync_summary_from_state(summary, state)
            summary["final_artifacts"] = {
                "global_metrics_json": current_global_metrics_json,
                "global_details_csv": current_global_details_csv,
                "global_inst_shp": current_global_inst_shp,
                "agent_summary_json": current_agent_summary_json,
                "optuna_best_json": current_optuna_best_json,
                "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
            }
            finetune_prev = stage_outputs_or_none(state, "finetune")
            if finetune_prev:
                summary["final_artifacts"]["finetune_summary_json"] = finetune_prev.get("finetune_summary_json")
            save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
            return

        state["status"] = "success"
        save_pipeline_state(pipeline_root, state)

    except Exception:
        sync_summary_from_state(summary, state)
        summary["final_artifacts"] = {
            "global_metrics_json": current_global_metrics_json,
            "global_details_csv": current_global_details_csv,
            "global_inst_shp": current_global_inst_shp,
            "agent_summary_json": current_agent_summary_json,
            "optuna_best_json": current_optuna_best_json,
            "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
        }
        finetune_prev = stage_outputs_or_none(state, "finetune")
        if finetune_prev:
            summary["final_artifacts"]["finetune_summary_json"] = finetune_prev.get("finetune_summary_json")
        summary["pipeline_state_json"] = str(pipeline_root / "pipeline_state.json")
        save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)
        raise

    sync_summary_from_state(summary, state)
    summary["final_artifacts"] = {
        "global_metrics_json": current_global_metrics_json,
        "global_details_csv": current_global_details_csv,
        "global_inst_shp": current_global_inst_shp,
        "agent_summary_json": current_agent_summary_json,
        "optuna_best_json": current_optuna_best_json,
        "spatial_context_summary_json": base_cfg.get("spatial_context_summary_json"),
                "spatial_context_object_json": base_cfg.get("spatial_context_object_json"),
    }
    finetune_prev = stage_outputs_or_none(state, "finetune")
    if finetune_prev:
        summary["final_artifacts"]["finetune_summary_json"] = finetune_prev.get("finetune_summary_json")
    summary["pipeline_state_json"] = str(pipeline_root / "pipeline_state.json")
    save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)


if __name__ == "__main__":
    main()
