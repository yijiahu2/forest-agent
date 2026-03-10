from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finetune_layer.io_utils import dump_csv, dump_json, load_csv


COMPARE_COLS = [
    "XBH",
    "tree_count_error_ratio",
    "mean_crown_width_error_ratio",
    "closure_error_abs",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_csv", required=True)
    parser.add_argument("--after_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = load_csv(args.before_csv)
    after = load_csv(args.after_csv)

    keep_before = [c for c in before.columns if c in COMPARE_COLS or c in ["mean_slope", "relief_elev", "dominant_aspect_class"]]
    keep_after = [c for c in after.columns if c in COMPARE_COLS]

    before = before[keep_before].copy()
    after = after[keep_after].copy()

    merged = before.merge(after, on="XBH", suffixes=("_before", "_after"))

    merged["gain_tree_count"] = merged["tree_count_error_ratio_before"] - merged["tree_count_error_ratio_after"]
    merged["gain_crown"] = merged["mean_crown_width_error_ratio_before"] - merged["mean_crown_width_error_ratio_after"]
    merged["gain_closure"] = merged["closure_error_abs_before"] - merged["closure_error_abs_after"]

    dump_csv(merged, out_dir / "finetune_compare.csv")

    summary = {
        "num_compared": int(len(merged)),
        "mean_gain_tree_count": float(merged["gain_tree_count"].mean()) if len(merged) else 0.0,
        "mean_gain_crown": float(merged["gain_crown"].mean()) if len(merged) else 0.0,
        "mean_gain_closure": float(merged["gain_closure"].mean()) if len(merged) else 0.0,
        "num_tree_improved": int((merged["gain_tree_count"] > 0).sum()) if len(merged) else 0,
        "num_crown_improved": int((merged["gain_crown"] > 0).sum()) if len(merged) else 0,
        "num_closure_improved": int((merged["gain_closure"] > 0).sum()) if len(merged) else 0,
    }
    dump_json(summary, out_dir / "finetune_gain_summary.json")
    print(f"[OK] finetune compare done: {out_dir}")


if __name__ == "__main__":
    main()