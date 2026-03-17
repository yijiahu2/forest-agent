from __future__ import annotations

from typing import Any, Dict, Optional


SAFE_SEARCH_SPACE = {
    "diam_list": [
        "96,160,256",
        "96,192,320",
        "128,192,320",
        "128,256,320",
    ],
    "tile": [1536, 2048],
    "overlap": [384, 512],
    "tile_overlap": [0.25, 0.35, 0.45],
    "augment": [True, False],
    "iou_merge_thr": [0.18, 0.22, 0.24, 0.28],
    "bsize": 256,
}


def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _nearest_choices(base_choices, center_value, window: int = 1):
    """
    针对有序离散候选（如 tile、overlap、tile_overlap、iou_merge_thr），
    以 hint 为中心取一个局部窗口。
    """
    if center_value not in base_choices:
        return list(base_choices)

    idx = base_choices.index(center_value)
    lo = max(0, idx - window)
    hi = min(len(base_choices), idx + window + 1)
    return list(base_choices[lo:hi])


def build_search_space(
    hint_params: Optional[Dict[str, Any]] = None,
    spatial_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    根据 agent hint 缩小搜索空间。
    设计原则：
    1. 永远保持安全空间子集
    2. bsize 始终固定为 256
    3. hint 不存在时返回完整 SAFE_SEARCH_SPACE
    4. categorical choices 一旦确定，在单次搜索内保持稳定
    """
    space = {
        "diam_list": list(SAFE_SEARCH_SPACE["diam_list"]),
        "tile": list(SAFE_SEARCH_SPACE["tile"]),
        "overlap": list(SAFE_SEARCH_SPACE["overlap"]),
        "tile_overlap": list(SAFE_SEARCH_SPACE["tile_overlap"]),
        "augment": list(SAFE_SEARCH_SPACE["augment"]),
        "iou_merge_thr": list(SAFE_SEARCH_SPACE["iou_merge_thr"]),
        "bsize": 256,
    }

    if not hint_params:
        return space

    # diam_list: 以 hint 为中心，保留 hint 本身；若合法，再允许一个轻微邻域
    hint_diam = hint_params.get("diam_list")
    if hint_diam in SAFE_SEARCH_SPACE["diam_list"]:
        idx = SAFE_SEARCH_SPACE["diam_list"].index(hint_diam)
        lo = max(0, idx - 1)
        hi = min(len(SAFE_SEARCH_SPACE["diam_list"]), idx + 2)
        space["diam_list"] = _dedup_keep_order(SAFE_SEARCH_SPACE["diam_list"][lo:hi])

    # tile
    hint_tile = hint_params.get("tile")
    if hint_tile in SAFE_SEARCH_SPACE["tile"]:
        space["tile"] = _nearest_choices(SAFE_SEARCH_SPACE["tile"], hint_tile, window=0)

    # overlap
    hint_overlap = hint_params.get("overlap")
    if hint_overlap in SAFE_SEARCH_SPACE["overlap"]:
        space["overlap"] = _nearest_choices(SAFE_SEARCH_SPACE["overlap"], hint_overlap, window=0)

    # tile_overlap
    hint_tile_overlap = hint_params.get("tile_overlap")
    if hint_tile_overlap in SAFE_SEARCH_SPACE["tile_overlap"]:
        space["tile_overlap"] = _nearest_choices(SAFE_SEARCH_SPACE["tile_overlap"], hint_tile_overlap, window=1)

    # augment
    hint_augment = hint_params.get("augment")
    if hint_augment in [True, False]:
        space["augment"] = [bool(hint_augment)]

    # iou_merge_thr
    hint_iou = hint_params.get("iou_merge_thr")
    if hint_iou in SAFE_SEARCH_SPACE["iou_merge_thr"]:
        space["iou_merge_thr"] = _nearest_choices(SAFE_SEARCH_SPACE["iou_merge_thr"], hint_iou, window=1)

    # bsize 强制固定
    space["bsize"] = 256

    return _apply_spatial_context_constraints(space, spatial_context)

def _apply_spatial_context_constraints(space: Dict[str, Any], spatial_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not spatial_context:
        return space

    terrain_profile = spatial_context.get("dominant_terrain_profile", {})
    slope_class = terrain_profile.get("dominant_slope_class")
    landform = terrain_profile.get("dominant_landform")
    aspect_class = terrain_profile.get("dominant_aspect_class")
    slope_position = terrain_profile.get("dominant_slope_position_class")

    if slope_class in {"steep", "very_steep"}:
        space["tile"] = [min(space["tile"])]
        space["overlap"] = [max(space["overlap"])]
        space["tile_overlap"] = [max(space["tile_overlap"])]

    if landform in {"mountain", "hill"}:
        space["iou_merge_thr"] = [x for x in space["iou_merge_thr"] if x <= 0.24] or space["iou_merge_thr"]

    if slope_class in {"flat", "gentle"} and landform in {"plain", "tableland"}:
        space["augment"] = [True]

    if aspect_class in {"north", "northeast", "northwest", "flat_no_aspect"}:
        space["augment"] = [True]

    if slope_position in {"ridge", "valley"}:
        space["tile_overlap"] = [max(space["tile_overlap"])]
        space["overlap"] = [max(space["overlap"])]

    return space
