from __future__ import annotations

import argparse
from pathlib import Path

from finetune_layer.io_utils import dump_csv, dump_json, load_csv, load_yaml, normalize_details_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before_csv", required=True)
    parser.add_argument("--after_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)

    before_raw = load_csv(args.before_csv)
    after_raw = load_csv(args.after_csv)

    before, _, missing_before = normalize_details_df(before_raw, cfg)
    after, _, missing_after = normalize_details_df(after_raw, cfg)

    if missing_before:
        raise ValueError(f"before_csv 规范化后仍缺列: {missing_before}")
    if missing_after:
        raise ValueError(f"after_csv 规范化后仍缺列: {missing_after}")

    keep_before = ["XBH", "tree_count_error_ratio", "mean_crown_width_error_ratio", "closure_error_abs",
                   "mean_slope", "relief_elev", "dominant_aspect_class"]
    keep_after = ["XBH", "tree_count_error_ratio", "mean_crown_width_error_ratio", "closure_error_abs"]

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