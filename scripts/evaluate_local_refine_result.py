import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
from tools.process_runner import run_streaming


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_merged_evaluation(
    base_config_path: str,
    local_refine_summary_path: str,
):
    base_cfg = load_yaml(base_config_path)
    summary = load_json(local_refine_summary_path)

    merged_shp = summary["merged_shp"]
    local_root = str(Path(local_refine_summary_path).parent)

    merged_metrics_json = str(Path(local_root) / "merged_metrics.json")
    merged_details_csv = str(Path(local_root) / "merged_details.csv")

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

    res = run_streaming(cmd)

    if res.returncode != 0:
        raise RuntimeError(f"Merged evaluation failed:\n{res.stdout}")

    if not Path(merged_metrics_json).exists():
        raise FileNotFoundError(f"merged_metrics.json not found: {merged_metrics_json}")
    if not Path(merged_details_csv).exists():
        raise FileNotFoundError(f"merged_details.csv not found: {merged_details_csv}")

    before_metrics = load_json(base_cfg["metrics_json"])
    after_metrics = load_json(merged_metrics_json)

    bad_ids = [str(x) for x in summary.get("bad_xiaoban_ids", [])]

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
        "bad_xiaoban_ids": bad_ids,
        "global_before_after": {},
        "bad_xiaoban_before_after": [],
    }

    for k in metric_keys:
        bv = before_metrics.get(k)
        av = after_metrics.get(k)
        compare["global_before_after"][k] = {
            "before": bv,
            "after": av,
            "delta": (av - bv) if (bv is not None and av is not None) else None
        }

    if bad_ids and "xiaoban_id" in before_details.columns and "xiaoban_id" in after_details.columns:
        merged_bad = before_details.merge(
            after_details,
            on="xiaoban_id",
            suffixes=("_before", "_after")
        )
        merged_bad = merged_bad[merged_bad["xiaoban_id"].isin(bad_ids)].copy()

        for _, row in merged_bad.iterrows():
            compare["bad_xiaoban_before_after"].append({
                "xiaoban_id": row["xiaoban_id"],
                "tree_count_error_abs_before": row.get("tree_count_error_abs_before"),
                "tree_count_error_abs_after": row.get("tree_count_error_abs_after"),
                "mean_crown_width_error_abs_before": row.get("mean_crown_width_error_abs_before"),
                "mean_crown_width_error_abs_after": row.get("mean_crown_width_error_abs_after"),
                "closure_error_abs_before": row.get("closure_error_abs_before"),
                "closure_error_abs_after": row.get("closure_error_abs_after"),
            })

    compare_json = str(Path(local_root) / "refine_compare_summary.json")
    save_json(compare, compare_json)

    print(f"[evaluate_local_refine_result] compare summary saved to: {compare_json}")
    return {
        "merged_metrics_json": merged_metrics_json,
        "merged_details_csv": merged_details_csv,
        "compare_json": compare_json,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--local_refine_summary", required=True)
    args = parser.parse_args()

    run_merged_evaluation(
        base_config_path=args.base_config,
        local_refine_summary_path=args.local_refine_summary,
    )


if __name__ == "__main__":
    main()
