from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finetune_layer.io_utils import dump_json


JOIN_COL = "xiaoban_id"

BEFORE_KEEP = [
    JOIN_COL,
    "tree_count_error_abs",
    "mean_crown_width_error_abs",
    "closure_error_abs",
    "density_error_abs",
    "mean_slope",
    "relief_elev",
    "dominant_aspect_class",
    "landform_type",
    "slope_class",
    "aspect_class",
    "slope_position_class",
]

AFTER_KEEP = [
    JOIN_COL,
    "tree_count_error_abs",
    "mean_crown_width_error_abs",
    "closure_error_abs",
    "density_error_abs",
]


def _keep_existing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_csv", required=True)
    parser.add_argument("--after_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", required=False)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = pd.read_csv(args.before_csv)
    after = pd.read_csv(args.after_csv)

    if JOIN_COL not in before.columns:
        raise RuntimeError(
            f"before_csv 缺少主键列 {JOIN_COL}，实际列为: {list(before.columns)}"
        )
    if JOIN_COL not in after.columns:
        raise RuntimeError(
            f"after_csv 缺少主键列 {JOIN_COL}，实际列为: {list(after.columns)}"
        )

    before = _keep_existing(before, BEFORE_KEEP)
    after = _keep_existing(after, AFTER_KEEP)

    merged = before.merge(after, on=JOIN_COL, suffixes=("_before", "_after"))

    if (
        "tree_count_error_abs_before" in merged.columns
        and "tree_count_error_abs_after" in merged.columns
    ):
        merged["gain_tree_count"] = (
            merged["tree_count_error_abs_before"] - merged["tree_count_error_abs_after"]
        )

    if (
        "mean_crown_width_error_abs_before" in merged.columns
        and "mean_crown_width_error_abs_after" in merged.columns
    ):
        merged["gain_crown"] = (
            merged["mean_crown_width_error_abs_before"]
            - merged["mean_crown_width_error_abs_after"]
        )

    if (
        "closure_error_abs_before" in merged.columns
        and "closure_error_abs_after" in merged.columns
    ):
        merged["gain_closure"] = (
            merged["closure_error_abs_before"] - merged["closure_error_abs_after"]
        )

    if (
        "density_error_abs_before" in merged.columns
        and "density_error_abs_after" in merged.columns
    ):
        merged["gain_density"] = (
            merged["density_error_abs_before"] - merged["density_error_abs_after"]
        )

    compare_csv = out_dir / "finetune_compare.csv"
    merged.to_csv(compare_csv, index=False)


    terrain_group_cols = [
        c
        for c in ["landform_type", "slope_class", "aspect_class", "slope_position_class"]
        if c in merged.columns
    ]
    stratified_gain = []
    if terrain_group_cols:
        grouped = merged.groupby(terrain_group_cols, dropna=False)
        for key, sub in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_dict = {terrain_group_cols[i]: key_tuple[i] for i in range(len(terrain_group_cols))}
            stratified_gain.append(
                {
                    **key_dict,
                    "num_samples": int(len(sub)),
                    "mean_gain_tree_count": float(sub["gain_tree_count"].mean()) if "gain_tree_count" in sub.columns else None,
                    "mean_gain_crown": float(sub["gain_crown"].mean()) if "gain_crown" in sub.columns else None,
                    "mean_gain_closure": float(sub["gain_closure"].mean()) if "gain_closure" in sub.columns else None,
                    "mean_gain_density": float(sub["gain_density"].mean()) if "gain_density" in sub.columns else None,
                }
            )

    summary = {
        "num_compared": int(len(merged)),
        "join_col": JOIN_COL,
        "mean_gain_tree_count": float(merged["gain_tree_count"].mean())
        if "gain_tree_count" in merged.columns and len(merged) > 0
        else None,
        "mean_gain_crown": float(merged["gain_crown"].mean())
        if "gain_crown" in merged.columns and len(merged) > 0
        else None,
        "mean_gain_closure": float(merged["gain_closure"].mean())
        if "gain_closure" in merged.columns and len(merged) > 0
        else None,
        "mean_gain_density": float(merged["gain_density"].mean())
        if "gain_density" in merged.columns and len(merged) > 0
        else None,
        "num_tree_improved": int((merged["gain_tree_count"] > 0).sum())
        if "gain_tree_count" in merged.columns
        else None,
        "num_crown_improved": int((merged["gain_crown"] > 0).sum())
        if "gain_crown" in merged.columns
        else None,
        "num_closure_improved": int((merged["gain_closure"] > 0).sum())
        if "gain_closure" in merged.columns
        else None,
        "num_density_improved": int((merged["gain_density"] > 0).sum())
        if "gain_density" in merged.columns
        else None,
        "terrain_group_cols": terrain_group_cols,
        "stratified_gain": stratified_gain,
    }

    dump_json(summary, out_dir / "finetune_gain_summary.json")
    print(f"[OK] finetune compare done: {out_dir}")


if __name__ == "__main__":
    main()