from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Any, digits: int = 4) -> str:
    x = _safe_float(v)
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _metric_table(metrics: Dict[str, Any]) -> List[str]:
    rows = [
        ("预测树木数量", metrics.get("pred_tree_count"), "期望树木数量", metrics.get("expected_tree_count")),
        ("树木数量误差率", metrics.get("tree_count_error_ratio"), "树木数量误差绝对值", metrics.get("tree_count_error_abs")),
        ("预测平均冠幅(m)", metrics.get("pred_mean_crown_width"), "期望平均冠幅(m)", metrics.get("expected_mean_crown_width")),
        ("冠幅误差率", metrics.get("mean_crown_width_error_ratio"), "冠幅误差绝对值", metrics.get("mean_crown_width_error_abs")),
        ("预测郁闭度", metrics.get("pred_cover_ratio"), "期望郁闭度", metrics.get("expected_closure")),
        ("郁闭度误差绝对值", metrics.get("closure_error_abs"), "预测密度(株/ha)", metrics.get("pred_density_trees_per_ha")),
        ("期望密度(株/ha)", metrics.get("expected_density"), "密度误差绝对值", metrics.get("density_error_abs")),
    ]
    lines = [
        "| 指标 | 数值 | 对照指标 | 数值 |",
        "|---|---:|---|---:|",
    ]
    for left_name, left_val, right_name, right_val in rows:
        lines.append(f"| {left_name} | {_fmt(left_val)} | {right_name} | {_fmt(right_val)} |")
    return lines


def _analyze_issues(metrics: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    tree_ratio = _safe_float(metrics.get("tree_count_error_ratio"))
    pred_tree = _safe_float(metrics.get("pred_tree_count"))
    exp_tree = _safe_float(metrics.get("expected_tree_count"))
    pred_crown = _safe_float(metrics.get("pred_mean_crown_width"))
    exp_crown = _safe_float(metrics.get("expected_mean_crown_width"))
    crown_ratio = _safe_float(metrics.get("mean_crown_width_error_ratio"))
    closure_err = _safe_float(metrics.get("closure_error_abs"))
    pred_density = _safe_float(metrics.get("pred_density_trees_per_ha"))
    exp_density = _safe_float(metrics.get("expected_density"))

    if pred_tree is not None and exp_tree is not None and pred_crown is not None and exp_crown is not None:
        if pred_tree > exp_tree and pred_crown < exp_crown:
            findings.append("存在明显过分割倾向：预测树木数量高于期望值，同时预测平均冠幅低于期望值，说明单木实例被切得偏碎。")
        elif pred_tree < exp_tree and pred_crown > exp_crown:
            findings.append("存在漏分或过度合并倾向：预测树木数量偏低，同时平均冠幅偏大，说明多棵相邻树冠可能被合并。")

    if tree_ratio is not None and tree_ratio >= 0.30:
        findings.append(f"树木数量误差率较高（{tree_ratio:.4f}），当前实例级结果与小班调查数量约束偏差较大。")

    if crown_ratio is not None and crown_ratio >= 0.25:
        findings.append(f"平均冠幅误差率较高（{crown_ratio:.4f}），当前树冠边界尺度与小班调查冠幅口径仍存在明显偏差。")

    if closure_err is not None:
        if closure_err <= 0.08:
            findings.append(f"郁闭度误差较小（{closure_err:.4f}），说明树冠覆盖范围整体位置基本合理，主要问题更可能出在实例切分粒度。")
        elif closure_err >= 0.15:
            findings.append(f"郁闭度误差偏大（{closure_err:.4f}），说明树冠总体覆盖范围本身也存在明显偏差。")

    if pred_density is not None and exp_density is not None and pred_density > exp_density * 1.4:
        findings.append("预测密度显著高于期望密度，进一步支持当前结果存在实例过碎、边界裂分过多的问题。")

    if not findings:
        findings.append("当前指标未显示出特别突出的单一失真类型，建议结合叠加图进一步检查局部区域分割质量。")
    return findings


def _top_problem_xiaoban(details_csv: str | Path, limit: int = 10) -> pd.DataFrame:
    details = pd.read_csv(details_csv)
    if details.empty:
        return details

    for col in ["tree_count_error_abs", "mean_crown_width_error_abs", "closure_error_abs", "density_error_abs"]:
        if col not in details.columns:
            details[col] = 0.0
        details[col] = pd.to_numeric(details[col], errors="coerce").fillna(0.0)

    details["problem_score"] = (
        details["tree_count_error_abs"] * 1.0
        + details["mean_crown_width_error_abs"] * 50.0
        + details["closure_error_abs"] * 100.0
        + details["density_error_abs"] / 100.0
    )
    cols = [
        c
        for c in [
            "xiaoban_id",
            "pred_tree_count",
            "expected_tree_count",
            "pred_mean_crown_width",
            "expected_mean_crown_width",
            "pred_cover_ratio",
            "expected_closure",
            "problem_score",
        ]
        if c in details.columns
    ]
    return details.sort_values("problem_score", ascending=False)[cols].head(limit)


def _df_to_markdown_table(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(_fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_experiment_report(
    summary: Dict[str, Any],
    report_path: str | Path,
) -> str:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = summary.get("metrics") or _load_json(summary["metrics_json"])
    details_csv = summary.get("details_csv") or summary.get("evaluation", {}).get("details_csv")

    lines: List[str] = []
    lines.append(f"# 实验结果报告：{summary.get('run_name', summary.get('mode', 'experiment'))}")
    lines.append("")
    lines.append("## 结果概览")
    lines.append("")
    lines.extend(_metric_table(metrics))
    lines.append("")

    lines.append("## 关键产物")
    lines.append("")
    artifact_rows = [
        ("结果指标 JSON", summary.get("metrics_json")),
        ("结果明细 CSV", details_csv),
        ("最终实例结果 SHP", summary.get("merged_inst_shp") or summary.get("stage2", {}).get("y_inst_shp")),
        ("实验摘要 JSON", summary.get("summary_json") or str(report_path.parent / "run_experiment_summary.json")),
        ("分组计划 JSON", summary.get("group_plan_json")),
        ("分组目录", summary.get("group_root")),
    ]
    for label, path in artifact_rows:
        if path:
            lines.append(f"- {label}: `{path}`")
    lines.append("")

    lines.append("## 自动问题分析")
    lines.append("")
    for item in _analyze_issues(metrics):
        lines.append(f"- {item}")
    lines.append("")

    if details_csv and Path(details_csv).exists():
        top_df = _top_problem_xiaoban(details_csv)
        if not top_df.empty:
            lines.append("## 重点问题小班")
            lines.append("")
            lines.extend(_df_to_markdown_table(top_df))
            lines.append("")

    group_summaries = summary.get("group_summaries") or []
    if group_summaries:
        lines.append("## 分组参数")
        lines.append("")
        lines.append("| Group | 小班数 | diam_list | tile | overlap | tile_overlap | augment | iou_merge_thr |")
        lines.append("|---|---:|---|---:|---:|---:|---|---:|")
        for group in group_summaries:
            params = group.get("params", {})
            xiaoban_ids = group.get("xiaoban_ids") or []
            lines.append(
                f"| {group.get('group_id', '-')} | {len(xiaoban_ids)} | {params.get('diam_list', '-')} | "
                f"{params.get('tile', '-')} | {params.get('overlap', '-')} | {params.get('tile_overlap', '-')} | "
                f"{params.get('augment', '-')} | {params.get('iou_merge_thr', '-')} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(report_path)
