import os
import json
from pathlib import Path
from typing import List, Dict, Any

import mlflow
import pandas as pd


def init_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)


def get_experiment_id(experiment_name: str):
    init_mlflow()
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    return exp.experiment_id


def get_best_runs(
    experiment_name: str,
    order_by: str = "metrics.closure_error_abs ASC",
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    init_mlflow()
    exp_id = get_experiment_id(experiment_name)
    if exp_id is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=[order_by],
        max_results=max_results,
    )
    if runs.empty:
        return []

    keep_cols = [
        "run_id",
        "tags.mlflow.runName",
        "params.diam_list",
        "params.tile",
        "params.overlap",
        "params.tile_overlap",
        "params.bsize",
        "params.augment",
        "params.iou_merge_thr",
        "metrics.tree_count_error_ratio",
        "metrics.mean_crown_width_error_ratio",
        "metrics.closure_error_abs",
        "metrics.density_error_abs",
        "tags.diagnosis_label",
    ]
    keep_cols = [c for c in keep_cols if c in runs.columns]
    return runs[keep_cols].to_dict(orient="records")


def get_latest_run_for_name(experiment_name: str, run_name: str) -> Dict[str, Any]:
    init_mlflow()
    exp_id = get_experiment_id(experiment_name)
    if exp_id is None:
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return {}

    row = runs.iloc[0].to_dict()
    return row


def read_metrics_json(metrics_json_path: str) -> Dict[str, Any]:
    p = Path(metrics_json_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}