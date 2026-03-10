from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finetune_layer.io_utils import dump_csv, dump_json, load_csv, load_yaml


REQUIRED_COLS = [
    "XBH",
    "tree_count_error_ratio",
    "mean_crown_width_error_ratio",
    "closure_error_abs",
]


def compute_candidate_score(row: pd.Series) -> float:
    tree_err = float(row.get("tree_count_error_ratio", 9.9))
    crown_err = float(row.get("mean_crown_width_error_ratio", 9.9))
    closure_err = float(row.get("closure_error_abs", 9.9))
    mean_slope = float(row.get("mean_slope", 0.0) or 0.0)
    relief = float(row.get("relief_elev", 0.0) or 0.0)

    # 分数越高越适合作为伪标签
    score = 1.0
    score -= min(tree_err, 1.0) * 0.40
    score -= min(crown_err, 1.0) * 0.30
    score -= min(closure_err, 1.0) * 0.20
    score -= min(mean_slope / 45.0, 1.0) * 0.05
    score -= min(relief / 80.0, 1.0) * 0.05
    return float(score)


def split_candidates(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_tree = float(cfg["max_tree_count_error_ratio"])
    max_crown = float(cfg["max_crown_error_ratio"])
    max_closure = float(cfg["max_closure_error_abs"])
    max_slope_easy = float(cfg.get("max_mean_slope_for_easy", 28.0))
    max_relief_easy = float(cfg.get("max_relief_for_easy", 35.0))

    work = df.copy()
    work["score"] = work.apply(compute_candidate_score, axis=1)

    pseudo_mask = (
        (work["tree_count_error_ratio"] <= max_tree)
        & (work["mean_crown_width_error_ratio"] <= max_crown)
        & (work["closure_error_abs"] <= max_closure)
        & (work.get("mean_slope", 0.0).fillna(0.0) <= max_slope_easy)
        & (work.get("relief_elev", 0.0).fillna(0.0) <= max_relief_easy)
    )

    hard_mask = (
        (work["tree_count_error_ratio"] > max_tree)
        | (work["mean_crown_width_error_ratio"] > max_crown)
        | (work["closure_error_abs"] > max_closure)
    )

    pseudo_df = work.loc[pseudo_mask].sort_values("score", ascending=False).copy()
    hard_df = work.loc[hard_mask].sort_values("score", ascending=True).copy()

    # 为了防止训练分布塌缩，保留一部分 good cases 回放
    good_df = work.sort_values("score", ascending=False).head(max(10, int(len(work) * cfg.get("good_replay_ratio", 0.25)))).copy()

    pseudo_df["split"] = "pseudo"
    hard_df["split"] = "hard"
    good_df["split"] = "replay_good"
    return pseudo_df, hard_df, good_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    details_csv = cfg["details_csv"]
    out_dir = Path(cfg["output_dir"]) / "pseudo_select"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(details_csv)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"details.csv 缺少必要列: {missing}")

    pseudo_df, hard_df, good_df = split_candidates(df, cfg)

    max_pseudo = int(cfg.get("max_pseudo_rois", 80))
    pseudo_df = pseudo_df.head(max_pseudo)

    dump_csv(pseudo_df, out_dir / "pseudo_candidates.csv")
    dump_csv(hard_df, out_dir / "hard_cases.csv")
    dump_csv(good_df, out_dir / "replay_good_cases.csv")

    summary = {
        "details_csv": details_csv,
        "num_total": int(len(df)),
        "num_pseudo": int(len(pseudo_df)),
        "num_hard": int(len(hard_df)),
        "num_replay_good": int(len(good_df)),
        "pseudo_strategy": cfg.get("pseudo_strategy", "conservative"),
        "outputs": {
            "pseudo_candidates_csv": str(out_dir / "pseudo_candidates.csv"),
            "hard_cases_csv": str(out_dir / "hard_cases.csv"),
            "replay_good_cases_csv": str(out_dir / "replay_good_cases.csv"),
        },
    }
    dump_json(summary, out_dir / "pseudo_select_summary.json")
    print(f"[OK] pseudo select done: {out_dir}")


if __name__ == "__main__":
    main()