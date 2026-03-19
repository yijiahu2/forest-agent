import copy
import os
import yaml
from pathlib import Path
from typing import Dict, Any


SAFE_PARAM_CONSTRAINTS = {
    "bsize": 256,
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _default_output_root() -> Path:
    return Path(os.getenv("FOREST_AGENT_OUTPUT_ROOT", "/home/xth/forest_agent_project/outputs"))


def sanitize_next_params(next_params: Dict[str, Any]) -> Dict[str, Any]:
    params = copy.deepcopy(next_params)

    # 强制固定运行安全参数
    params["bsize"] = SAFE_PARAM_CONSTRAINTS["bsize"]

    # augment 统一转 bool
    if "augment" in params:
        if isinstance(params["augment"], str):
            params["augment"] = params["augment"].lower() == "true"

    # 数值字段转类型
    for key in ["tile", "overlap", "bsize"]:
        if key in params:
            params[key] = int(params[key])

    for key in ["tile_overlap", "iou_merge_thr"]:
        if key in params:
            params[key] = float(params[key])

    return params


def build_next_config(
    base_config_path: str,
    out_config_path: str,
    run_name: str,
    next_params: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = load_yaml(base_config_path)
    cfg = copy.deepcopy(cfg)

    cfg["run_name"] = run_name

    next_params = sanitize_next_params(next_params)

    for k, v in next_params.items():
        cfg[k] = v

    base_outputs = _default_output_root()
    cfg["output_dir"] = str(base_outputs / run_name / "seg_output")
    cfg["metrics_json"] = str(base_outputs / run_name / "metrics.json")
    cfg["details_csv"] = str(base_outputs / run_name / "details.csv")

    save_yaml(cfg, out_config_path)
    return cfg
