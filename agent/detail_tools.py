import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def summarize_details_csv(details_csv_path: str, top_k: int = 3) -> Dict[str, Any]:
    """
    从 details.csv 提取：
    1. 小班总数
    2. tree_count_error_abs 最大的前 top_k 个小班
    3. mean_crown_width_error_abs 最大的前 top_k 个小班
    4. closure_error_abs 最大的前 top_k 个小班
    """
    p = Path(details_csv_path)
    if not p.exists():
        return {"exists": False, "top_k_xiaoban": []}

    df = pd.read_csv(p)
    if df.empty:
        return {"exists": True, "num_xiaoban": 0, "top_k_xiaoban": []}

    # 补齐缺失列，避免排序时报错
    for col in [
        "tree_count_error_abs",
        "mean_crown_width_error_abs",
        "closure_error_abs",
        "density_error_abs",
        "pred_tree_count",
        "pred_mean_crown_width",
        "pred_cover_ratio",
        "expected_tree_count",
        "expected_mean_crown_width",
        "expected_closure",
        "expected_density",
        "pred_density_trees_per_ha",
    ]:
        if col not in df.columns:
            df[col] = None

    # 构造综合误差分数，便于筛出最差小班
    def row_score(row):
        score = 0.0
        weights = [
            ("tree_count_error_abs", 1.0),
            ("mean_crown_width_error_abs", 5.0),
            ("closure_error_abs", 10.0),
            ("density_error_abs", 0.001),
        ]
        for col, w in weights:
            v = safe_float(row.get(col))
            if v is not None:
                score += w * abs(v)
        return score

    df["error_score"] = df.apply(row_score, axis=1)
    df = df.sort_values("error_score", ascending=False)

    top_rows: List[Dict[str, Any]] = []
    for _, row in df.head(top_k).iterrows():
        top_rows.append({
            "xiaoban_id": str(row.get("xiaoban_id")),
            "pred_tree_count": safe_float(row.get("pred_tree_count")),
            "expected_tree_count": safe_float(row.get("expected_tree_count")),
            "tree_count_error_abs": safe_float(row.get("tree_count_error_abs")),

            "pred_mean_crown_width": safe_float(row.get("pred_mean_crown_width")),
            "expected_mean_crown_width": safe_float(row.get("expected_mean_crown_width")),
            "mean_crown_width_error_abs": safe_float(row.get("mean_crown_width_error_abs")),

            "pred_cover_ratio": safe_float(row.get("pred_cover_ratio")),
            "expected_closure": safe_float(row.get("expected_closure")),
            "closure_error_abs": safe_float(row.get("closure_error_abs")),

            "pred_density_trees_per_ha": safe_float(row.get("pred_density_trees_per_ha")),
            "expected_density": safe_float(row.get("expected_density")),
            "density_error_abs": safe_float(row.get("density_error_abs")),

            "error_score": safe_float(row.get("error_score")),
        })

    return {
        "exists": True,
        "num_xiaoban": int(len(df)),
        "top_k_xiaoban": top_rows,
    }