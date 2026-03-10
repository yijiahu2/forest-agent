from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from finetune_layer.io_utils import (
    dump_csv,
    dump_json,
    load_csv,
    load_yaml,
    normalize_details_df,
)


def _get_metric_weights(df: pd.DataFrame) -> dict[str, float]:
    """
    根据当前实际可用误差列动态分配权重。
    """
    candidates = {
        "tree_count_error_ratio": 0.40,
        "mean_crown_width_error_ratio": 0.30,
        "closure_error_abs": 0.20,
        "density_error_abs": 0.10,
    }
    usable = {k: v for k, v in candidates.items() if k in df.columns and df[k].notna().any()}
    if not usable:
        return {}
    s = sum(usable.values())
    return {k: v / s for k, v in usable.items()}


def compute_candidate_score(row: pd.Series, weights: dict[str, float]) -> float:
    score = 1.0

    if "tree_count_error_ratio" in weights:
        tree_err = float(row.get("tree_count_error_ratio", 1.0) or 1.0)
        score -= min(tree_err, 1.0) * weights["tree_count_error_ratio"]

    if "mean_crown_width_error_ratio" in weights:
        crown_err = float(row.get("mean_crown_width_error_ratio", 1.0) or 1.0)
        score -= min(crown_err, 1.0) * weights["mean_crown_width_error_ratio"]

    if "closure_error_abs" in weights:
        closure_err = float(row.get("closure_error_abs", 1.0) or 1.0)
        score -= min(closure_err, 1.0) * weights["closure_error_abs"]

    if "density_error_abs" in weights:
        density_err = float(row.get("density_error_abs", 1.0) or 1.0)
        score -= min(density_err, 1.0) * weights["density_error_abs"]

    mean_slope = float(row.get("mean_slope", 0.0) or 0.0)
    relief = float(row.get("relief_elev", 0.0) or 0.0)
    score -= min(mean_slope / 45.0, 1.0) * 0.05
    score -= min(relief / 80.0, 1.0) * 0.05
    return float(score)


def build_masks(work: pd.DataFrame, cfg: dict, usable_metrics: list[str]) -> tuple[pd.Series, pd.Series]:
    pseudo_mask = pd.Series(True, index=work.index)
    hard_mask = pd.Series(False, index=work.index)

    if "tree_count_error_ratio" in usable_metrics:
        max_tree = float(cfg["max_tree_count_error_ratio"])
        pseudo_mask &= work["tree_count_error_ratio"].fillna(np.inf) <= max_tree
        hard_mask |= work["tree_count_error_ratio"].fillna(0.0) > max_tree

    if "mean_crown_width_error_ratio" in usable_metrics:
        max_crown = float(cfg["max_crown_error_ratio"])
        pseudo_mask &= work["mean_crown_width_error_ratio"].fillna(np.inf) <= max_crown
        hard_mask |= work["mean_crown_width_error_ratio"].fillna(0.0) > max_crown

    if "closure_error_abs" in usable_metrics:
        max_closure = float(cfg["max_closure_error_abs"])
        pseudo_mask &= work["closure_error_abs"].fillna(np.inf) <= max_closure
        hard_mask |= work["closure_error_abs"].fillna(0.0) > max_closure

    max_slope_easy = float(cfg.get("max_mean_slope_for_easy", 28.0))
    max_relief_easy = float(cfg.get("max_relief_for_easy", 35.0))
    pseudo_mask &= work["mean_slope"].fillna(0.0) <= max_slope_easy
    pseudo_mask &= work["relief_elev"].fillna(0.0) <= max_relief_easy

    return pseudo_mask, hard_mask


def split_candidates(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    work = df.copy()

    usable_metrics = [
        c for c in [
            "tree_count_error_ratio",
            "mean_crown_width_error_ratio",
            "closure_error_abs",
            "density_error_abs",
        ]
        if c in work.columns and work[c].notna().any()
    ]
    metric_weights = _get_metric_weights(work)

    work["score"] = work.apply(lambda r: compute_candidate_score(r, metric_weights), axis=1)

    pseudo_mask, hard_mask = build_masks(work, cfg, usable_metrics)

    pseudo_df = work.loc[pseudo_mask].sort_values("score", ascending=False).copy()
    hard_df = work.loc[hard_mask].sort_values("score", ascending=True).copy()

    replay_n = max(10, int(len(work) * float(cfg.get("good_replay_ratio", 0.25))))
    good_df = work.sort_values("score", ascending=False).head(replay_n).copy()

    pseudo_df["split"] = "pseudo"
    hard_df["split"] = "hard"
    good_df["split"] = "replay_good"

    meta = {
        "usable_metrics": usable_metrics,
        "metric_weights": metric_weights,
    }
    return pseudo_df, hard_df, good_df, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    details_csv = cfg["details_csv"]
    out_dir = Path(cfg["output_dir"]) / "pseudo_select"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_csv(details_csv)
    norm_df, rename_map, missing, debug_info = normalize_details_df(raw_df, cfg)

    dump_csv(norm_df, out_dir / "normalized_details.csv")

    if missing:
        debug_payload = {
            "details_csv": details_csv,
            "rename_map": rename_map,
            "debug_info": debug_info,
            "raw_columns": raw_df.columns.tolist(),
            "normalized_columns": norm_df.columns.tolist(),
            "hint": "当前最低要求是 XBH + 至少一个可用误差指标。",
        }
        dump_json(debug_payload, out_dir / "pseudo_select_debug.json")
        raise ValueError(f"details.csv 规范化失败: {missing}")

    pseudo_df, hard_df, good_df, meta = split_candidates(norm_df, cfg)

    max_pseudo = int(cfg.get("max_pseudo_rois", 80))
    pseudo_df = pseudo_df.head(max_pseudo)

    dump_csv(pseudo_df, out_dir / "pseudo_candidates.csv")
    dump_csv(hard_df, out_dir / "hard_cases.csv")
    dump_csv(good_df, out_dir / "replay_good_cases.csv")

    summary = {
        "details_csv": details_csv,
        "num_total": int(len(norm_df)),
        "num_pseudo": int(len(pseudo_df)),
        "num_hard": int(len(hard_df)),
        "num_replay_good": int(len(good_df)),
        "pseudo_strategy": cfg.get("pseudo_strategy", "conservative"),
        "rename_map": rename_map,
        "raw_columns": raw_df.columns.tolist(),
        "normalized_columns": norm_df.columns.tolist(),
        "usable_metrics": meta["usable_metrics"],
        "metric_weights": meta["metric_weights"],
        "outputs": {
            "normalized_details_csv": str(out_dir / "normalized_details.csv"),
            "pseudo_candidates_csv": str(out_dir / "pseudo_candidates.csv"),
            "hard_cases_csv": str(out_dir / "hard_cases.csv"),
            "replay_good_cases_csv": str(out_dir / "replay_good_cases.csv"),
        },
    }
    dump_json(summary, out_dir / "pseudo_select_summary.json")
    print(f"[OK] pseudo select done: {out_dir}")


if __name__ == "__main__":
    main()