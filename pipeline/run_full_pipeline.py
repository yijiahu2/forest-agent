from __future__ import annotations

import argparse
import json
import os
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


def run_subprocess(cmd, cwd: Optional[str] = None) -> Dict[str, Any]:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    print("\n===== CMD =====")
    print(" ".join(cmd))
    print("\n===== STDOUT =====")
    print(res.stdout)
    print("\n===== STDERR =====")
    print(res.stderr)

    return {
        "cmd": cmd,
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


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


def assert_shapefile_complete(shp_path: Optional[str], name: str = "xiaoban_shp"):
    assert_file(shp_path, name)
    shp = Path(shp_path)
    stem = shp.with_suffix("")
    required = [
        stem.with_suffix(".shp"),
        stem.with_suffix(".dbf"),
        stem.with_suffix(".shx"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{name} incomplete, missing sidecar files: {missing}")


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
    assert_shapefile_complete(cfg.get("xiaoban_shp"), "xiaoban_shp")

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

    # optuna storage 仅做路径级预检查
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
    cfg["dem_tif"] = dem_tif
    cfg["slope_tif"] = slope_tif
    cfg["aspect_tif"] = aspect_tif

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
        return str(existing_runtime)

    return apply_cli_terrain_overrides_to_config(
        base_config_path=args.base_config,
        dem_tif=args.dem_tif,
        slope_tif=args.slope_tif,
        aspect_tif=args.aspect_tif,
        pipeline_root=pipeline_root,
    )


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
    max_rounds: int = 3,
) -> Dict[str, Any]:
    cmd = [
        "python",
        "-m",
        "agent.graph",
        "--base_config",
        base_config_path,
        "--max_rounds",
        str(max_rounds),
    ]
    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project")
    if res["returncode"] != 0:
        raise RuntimeError(f"agent stage failed:\n{res['stderr']}")

    summary_json = "/home/xth/forest_agent_project/outputs/agent/final_summary.json"
    if not Path(summary_json).exists():
        raise FileNotFoundError(f"agent final_summary.json not found: {summary_json}")

    return {
        "agent_run": res,
        "agent_summary_json": summary_json,
    }


def run_optuna_single_stage(
    base_config_path: str,
    n_trials: int = 2,
    agent_hint_json: Optional[str] = None,
    out_best_json: str = "/home/xth/forest_agent_project/outputs/optuna/optuna_single_best.json",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
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
    if study_name:
        cmd.extend(["--study_name", study_name])
    if storage:
        cmd.extend(["--storage", storage])
    if resume:
        cmd.append("--resume")

    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project")
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
    n_trials: int = 2,
    agent_hint_json: Optional[str] = None,
    out_best_json: str = "/home/xth/forest_agent_project/outputs/optuna/optuna_multi_best.json",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
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
    if study_name:
        cmd.extend(["--study_name", study_name])
    if storage:
        cmd.extend(["--storage", storage])
    if resume:
        cmd.append("--resume")

    res = run_subprocess(cmd, cwd="/home/xth/forest_agent_project")
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
) -> Dict[str, Any]:
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
    )

    return {
        "local_refine_summary": summary,
        "merged_shp": summary.get("merged_shp"),
        "merged_metrics_json": summary.get("merged_metrics_json"),
        "merged_details_csv": summary.get("merged_details_csv"),
        "compare_json": summary.get("compare_json"),
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


def save_pipeline_summary(
    *,
    pipeline_root: Path,
    summary: Dict[str, Any],
):
    summary_path = pipeline_root / "pipeline_summary.json"
    save_json(summary, summary_path)
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

    # resolve pipeline root
    if args.run_dir:
        pipeline_root = Path(args.run_dir)
        ensure_dir(pipeline_root)
        pipeline_name = pipeline_root.name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_name = f"pipeline_{Path(args.base_config).stem}_{stamp}"
        pipeline_root = Path(f"/home/xth/forest_agent_project/outputs/pipeline/{pipeline_name}")
        ensure_dir(pipeline_root)

    runtime_base_config = resolve_runtime_base_config(
        args=args,
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
        "base_config": args.base_config,
        "runtime_base_config": runtime_base_config,
        "run_flags": {
            "run_baseline": args.run_baseline,
            "run_agent": args.run_agent,
            "run_optuna_single": args.run_optuna_single,
            "run_optuna_multi": args.run_optuna_multi,
            "run_local_refine": args.run_local_refine,
        },
        "terrain_inputs": {
            "dem_tif": base_cfg.get("dem_tif"),
            "slope_tif": base_cfg.get("slope_tif"),
            "aspect_tif": base_cfg.get("aspect_tif"),
        },
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
                n_trials=args.n_trials_single,
                agent_hint_json=current_agent_summary_json,
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
                n_trials=args.n_trials_multi,
                agent_hint_json=current_agent_summary_json,
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
        }
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
    }
    summary["pipeline_state_json"] = str(pipeline_root / "pipeline_state.json")
    save_pipeline_summary(pipeline_root=pipeline_root, summary=summary)


if __name__ == "__main__":
    main()