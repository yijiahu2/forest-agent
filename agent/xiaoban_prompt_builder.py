from __future__ import annotations

import json
from typing import Any, Dict, List


def build_group_param_prompt(
    run_meta: Dict[str, Any],
    groups: List[Dict[str, Any]],
    default_params: Dict[str, Any],
    spatial_context: Dict[str, Any] | None = None,
) -> str:
    compact_groups = []
    for group in groups:
        compact_groups.append(
            {
                "group_id": group["group_id"],
                "strategy_label": group.get("strategy_label"),
                "num_xiaoban": group.get("num_xiaoban"),
                "xiaoban_ids": group.get("xiaoban_ids"),
                "dominant_terrain": group.get("dominant_terrain"),
                "weighted_inventory": group.get("weighted_inventory"),
                "current_params": group.get("params"),
            }
        )

    return f"""
你是森林单木分割分区规划智能体。
目标：基于不同小班组的 inventory + terrain 特征，为每个 group 选择更合适的首轮分割参数。

运行元信息：
{json.dumps(run_meta, ensure_ascii=False, indent=2)}

默认兜底参数：
{json.dumps(default_params, ensure_ascii=False, indent=2)}

空间上下文摘要：
{json.dumps(spatial_context or {}, ensure_ascii=False, indent=2)}

待分析的小班组：
{json.dumps(compact_groups, ensure_ascii=False, indent=2)}

可调参数范围：
- diam_list: ["96,160,256", "96,192,320", "128,192,320", "128,256,320"]
- tile: [1536, 2048]
- overlap: [384, 512]
- tile_overlap: [0.25, 0.35, 0.45]
- augment: [true, false]
- iou_merge_thr: [0.18, 0.22, 0.24, 0.28]
- bsize: [256]

约束：
1. 必须覆盖每个 group_id
2. 只能从给定范围中选值
3. bsize 固定为 256
4. 山地、陡坡、高密度、郁闭度高的小班组，优先考虑更稳健的 overlap / tile_overlap
5. 不要输出 markdown，不要输出解释性文字

请严格输出 JSON，格式如下：
{{
  "groups": [
    {{
      "group_id": "group_001",
      "params": {{
        "diam_list": "96,192,320",
        "tile": 1536,
        "overlap": 512,
        "tile_overlap": 0.45,
        "augment": true,
        "iou_merge_thr": 0.24,
        "bsize": 256
      }},
      "reason": "简短说明"
    }}
  ]
}}
"""
