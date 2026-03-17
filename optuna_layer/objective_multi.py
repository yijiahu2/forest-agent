import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config_builder import load_yaml, save_yaml  # noqa: E402
from tools.process_runner import run_streaming  # noqa: E402


DEFAULT_RUNNER = "/home/xth/forest_agent_project/scripts/run_zstreeseg_experiment.py"
DEFAULT_BASE_CONFIG = "/home/xth/forest_agent_project/configs/exp_dom194.yaml"
DEFAULT_TRIAL_CFG_DIR = "/home/xth/forest_agent_project/configs/generated"
DEFAULT_TRIAL_SUMMARY_DIR = "/home/xth/forest_agent_project/outputs/optuna/trials"


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    ensure_parent(Path(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(params)
    out["bsize"] = 256
    return out


def suggest_params(trial, search_space):
    params = {
        "diam_list": trial.suggest_categorical("diam_list", search_space["diam_list"]),
        "tile": trial.suggest_categorical("tile", search_space["tile"]),
        "overlap": trial.suggest_categorical("overlap", search_space["overlap"]),
        "tile_overlap": trial.suggest_categorical("tile_overlap", search_space["tile_overlap"]),
        "augment": trial.suggest_categorical("augment", search_space["augment"]),
        "iou_merge_thr": trial.suggest_categorical("iou_merge_thr", search_space["iou_merge_thr"]),
        "bsize": 256,
    }
    return sanitize_params(params)


def build_trial_config(
    base_config_path: str,
    params: Dict[str, Any],
    trial_number: int,
) -> Dict[str, Any]:
    cfg = load_yaml(base_config_path)
    params = sanitize_params(params)

    base_run_name = cfg.get("run_name", "optuna_multi_run")
    run_name = f"{base_run_name}_optuna_multi_trial_{trial_number:04d}"

    cfg["run_name"] = run_name
    for k, v in params.items():
        cfg[k] = v

    cfg["metrics_json"] = f"/home/xth/forest_agent_project/outputs/{run_name}/metrics.json"
    cfg["details_csv"] = f"/home/xth/forest_agent_project/outputs/{run_name}/details.csv"

    cfg_path = Path(DEFAULT_TRIAL_CFG_DIR) / f"{run_name}.yaml"
    ensure_parent(cfg_path)
    save_yaml(cfg, str(cfg_path))

    return {
        "config": cfg,
        "config_path": str(cfg_path),
        "run_name": run_name,
        "params": params,
    }


def run_trial_experiment(config_path: str) -> Dict[str, Any]:
    cmd = [
        "python",
        DEFAULT_RUNNER,
        "--config",
        config_path,
    ]
    res = run_streaming(cmd)

    if res.returncode != 0:
        raise RuntimeError(f"Trial failed:\n{res.stdout}")

    cfg = load_yaml(config_path)
    metrics_json = cfg["metrics_json"]
    details_csv = cfg["details_csv"]

    if not Path(metrics_json).exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_json}")

    metrics = load_json(metrics_json)
    if not metrics:
        raise ValueError(f"metrics.json is empty: {metrics_json}")

    return {
        "metrics": metrics,
        "metrics_json": metrics_json,
        "details_csv": details_csv,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


def compute_multi_objectives(metrics: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    4 个目标，全部越小越好
    """
    tree_count_error_ratio = float(metrics.get("tree_count_error_ratio", 999.0))
    mean_crown_width_error_ratio = float(metrics.get("mean_crown_width_error_ratio", 999.0))
    closure_error_abs = float(metrics.get("closure_error_abs", 999.0))
    density_error_abs = float(metrics.get("density_error_abs", 999999.0)) / 1000.0
    return (
        tree_count_error_ratio,
        mean_crown_width_error_ratio,
        closure_error_abs,
        density_error_abs,
    )


def objective_multi_impl(trial, params: Dict[str, Any], base_config_path: str = DEFAULT_BASE_CONFIG):
    trial_cfg = build_trial_config(
        base_config_path=base_config_path,
        params=params,
        trial_number=trial.number,
    )

    run_info = run_trial_experiment(trial_cfg["config_path"])
    metrics = run_info["metrics"]
    values = compute_multi_objectives(metrics)

    trial.set_user_attr("params", params)
    trial.set_user_attr("run_name", trial_cfg["run_name"])
    trial.set_user_attr("config_path", trial_cfg["config_path"])
    trial.set_user_attr("metrics_json", run_info["metrics_json"])
    trial.set_user_attr("details_csv", run_info["details_csv"])
    trial.set_user_attr("values", values)

    summary = {
        "trial_number": trial.number,
        "run_name": trial_cfg["run_name"],
        "params": params,
        "metrics": metrics,
        "values": values,
        "metrics_json": run_info["metrics_json"],
        "details_csv": run_info["details_csv"],
        "config_path": trial_cfg["config_path"],
    }
    out_json = Path(DEFAULT_TRIAL_SUMMARY_DIR) / f"multi_trial_{trial.number:04d}.json"
    save_json(summary, str(out_json))

    return values


def make_objective_multi(
    search_space,
    base_config_path: str = DEFAULT_BASE_CONFIG,
):
    def _objective(trial):
        params = suggest_params(trial, search_space)
        return objective_multi_impl(
            trial=trial,
            params=params,
            base_config_path=base_config_path,
        )

    return _objective
