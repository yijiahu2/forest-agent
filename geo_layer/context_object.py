from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_spatial_context_object_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    spatial_summary_json = cfg.get("spatial_context_summary_json")
    spatial_summary: Dict[str, Any] = {}
    if spatial_summary_json and Path(spatial_summary_json).exists():
        try:
            spatial_summary = _load_json(spatial_summary_json)
        except Exception:
            spatial_summary = {}

    terrain_rule_config = {
        "flat_slope_threshold_deg": float(cfg.get("flat_slope_threshold_deg", 5.0)),
        "plain_relief_threshold_m": float(cfg.get("plain_relief_threshold_m", 30.0)),
    }

    terrain_constraint_fields = {
        "terrain_landform_field": cfg.get("terrain_landform_field", "landform_type"),
        "terrain_slope_class_field": cfg.get("terrain_slope_class_field", "slope_class"),
        "terrain_aspect_class_field": cfg.get("terrain_aspect_class_field", "aspect_class"),
        "terrain_slope_position_field": cfg.get("terrain_slope_position_field", "slope_position_class"),
    }

    terrain_inputs = {
        "dem_tif": cfg.get("dem_tif"),
        "slope_tif": cfg.get("slope_tif"),
        "aspect_tif": cfg.get("aspect_tif"),
        "landform_tif": cfg.get("landform_tif"),
        "slope_position_tif": cfg.get("slope_position_tif"),
    }

    return {
        "spatial_context_enabled": bool(cfg.get("spatial_context_summary_json")),
        "spatial_context_summary_json": spatial_summary_json,
        "spatial_context_summary": spatial_summary,
        "terrain_inputs": terrain_inputs,
        "terrain_rule_config": terrain_rule_config,
        "terrain_constraint_fields": terrain_constraint_fields,
        "terrain_quartet_fields": {
            "landform_type": terrain_constraint_fields["terrain_landform_field"],
            "slope_class": terrain_constraint_fields["terrain_slope_class_field"],
            "aspect_class": terrain_constraint_fields["terrain_aspect_class_field"],
            "slope_position_class": terrain_constraint_fields["terrain_slope_position_field"],
        },
        "dominant_terrain_profile": spatial_summary.get("terrain_class_summary", {}),
    }

