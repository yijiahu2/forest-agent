import json


def build_proposal_prompt(
    prior_summary: dict,
    latest_metrics: dict,
    best_history: list,
    details_summary: dict,
    iteration: int,
    max_iterations: int,
) -> str:
    return f"""
你是森林单木分割参数优化智能体。
目标：针对复杂林分 patch，优化 ZS-TreeSeg 参数，使以下误差尽可能小：
1. tree_count_error_ratio
2. mean_crown_width_error_ratio
3. closure_error_abs
4. density_error_abs

当前是第 {iteration} / {max_iterations} 轮。

当前 patch/小班先验摘要：
{json.dumps(prior_summary, ensure_ascii=False, indent=2)}

当前 patch 总体实验结果：
{json.dumps(latest_metrics, ensure_ascii=False, indent=2)}

当前 patch 内误差最大的若干小班摘要：
{json.dumps(details_summary, ensure_ascii=False, indent=2)}

历史较优实验：
{json.dumps(best_history, ensure_ascii=False, indent=2)}

可调参数范围：
- diam_list: ["96,160,256", "96,192,320", "128,192,320", "128,256,320"]
- tile: [1536, 2048]
- overlap: [384, 512]
- tile_overlap: [0.25, 0.35, 0.45]
- augment: [true, false]
- iou_merge_thr: [0.18, 0.22, 0.24, 0.28]

重要约束：
1. bsize 固定为 256，绝对不能修改
2. 只能从给定范围中选值
3. 当 tree_count_error_ratio 已经很小（例如 < 0.05）时，优先优化冠幅误差、郁闭度误差和密度误差
4. 如果主导问题是冠层覆盖不足，应优先考虑提升冠层完整性，而不是一味增加实例数量
5. 不要输出 markdown
6. 不要输出多余文字

请严格输出 JSON，格式如下：
{{
  "diagnosis_label": "一个简短英文标签",
  "next_params": {{
    "diam_list": "96,192,320",
    "tile": 2048,
    "overlap": 512,
    "tile_overlap": 0.35,
    "bsize": 256,
    "augment": true,
    "iou_merge_thr": 0.22
  }},
  "reason": "简要说明"
}}
"""