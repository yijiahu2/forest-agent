# ~/forest_agent_project/optuna_layer/study_tools.py

from typing import Dict, Any


def compute_objective_score(metrics: Dict[str, Any]) -> float:
    """
    分数越小越好
    """
    tree_err = float(metrics.get("tree_count_error_ratio", 999))
    crown_err = float(metrics.get("mean_crown_width_error_ratio", 999))
    closure_err = float(metrics.get("closure_error_abs", 999))
    density_err = float(metrics.get("density_error_abs", 999))

    # density 的量纲远大，因此缩放一下
    density_scaled = density_err / 1000.0

    score = (
        1.0 * tree_err +
        2.0 * crown_err +
        3.0 * closure_err +
        1.0 * density_scaled
    )
    return score


def summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "pred_tree_count",
        "expected_tree_count",
        "tree_count_error_ratio",
        "pred_mean_crown_width",
        "expected_mean_crown_width",
        "mean_crown_width_error_ratio",
        "pred_cover_ratio",
        "expected_closure",
        "closure_error_abs",
        "pred_density_trees_per_ha",
        "expected_density",
        "density_error_abs",
    ]
    return {k: metrics.get(k) for k in keep if k in metrics}
