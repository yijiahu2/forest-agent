from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

from finetune_layer.io_utils import dump_json, load_csv, load_yaml


def save_crop(src_path: str, geom, out_path: Path) -> None:
    with rasterio.open(src_path) as src:
        out_img, out_transform = mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pseudo_csv", required=True)
    parser.add_argument("--replay_good_csv", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"]) / "pseudo_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_image = cfg["input_image"]
    dem_tif = cfg.get("dem_tif")
    slope_tif = cfg.get("slope_tif")
    aspect_tif = cfg.get("aspect_tif")
    shp = cfg["xiaoban_shp"]
    xbh_field = cfg["xiaoban_id_field"]

    pseudo_df = load_csv(args.pseudo_csv)
    replay_df = load_csv(args.replay_good_csv)
    all_df = pd.concat([pseudo_df, replay_df], ignore_index=True)

    gdf = gpd.read_file(shp)
    if xbh_field not in gdf.columns:
        raise ValueError(f"小班 shp 中未找到字段: {xbh_field}")

    records = []
    for _, row in all_df.iterrows():
        xbh = str(row["XBH"])
        split = str(row["split"])

        sub = gdf[gdf[xbh_field].astype(str) == xbh]
        if len(sub) == 0:
            continue

        geom = sub.iloc[0].geometry
        roi_id = f"{split}_{xbh}"

        image_path = out_dir / "images" / f"{roi_id}.tif"
        save_crop(input_image, geom, image_path)

        dem_path = None
        slope_path = None
        aspect_path = None

        if dem_tif:
            dem_path = out_dir / "terrain" / "dem" / f"{roi_id}.tif"
            save_crop(dem_tif, geom, dem_path)

        if slope_tif:
            slope_path = out_dir / "terrain" / "slope" / f"{roi_id}.tif"
            save_crop(slope_tif, geom, slope_path)

        if aspect_tif:
            aspect_path = out_dir / "terrain" / "aspect" / f"{roi_id}.tif"
            save_crop(aspect_tif, geom, aspect_path)

        # 第一版语义 mask 先占位：后续从 stage1 输出或实例结果转语义结果填充
        mask_sem_path = out_dir / "masks_sem" / f"{roi_id}.tif"

        meta = {
            "roi_id": roi_id,
            "xbh": xbh,
            "split": split,
            "tree_count_error_ratio": float(row.get("tree_count_error_ratio", 0.0)),
            "mean_crown_width_error_ratio": float(row.get("mean_crown_width_error_ratio", 0.0)),
            "closure_error_abs": float(row.get("closure_error_abs", 0.0)),
            "mean_slope": None if pd.isna(row.get("mean_slope")) else float(row.get("mean_slope")),
            "relief_elev": None if pd.isna(row.get("relief_elev")) else float(row.get("relief_elev")),
            "dominant_aspect_class": None if pd.isna(row.get("dominant_aspect_class")) else str(row.get("dominant_aspect_class")),
            "image_path": str(image_path),
            "mask_sem_path": str(mask_sem_path),
            "dem_path": None if dem_path is None else str(dem_path),
            "slope_path": None if slope_path is None else str(slope_path),
            "aspect_path": None if aspect_path is None else str(aspect_path),
        }

        meta_path = out_dir / "meta" / f"{roi_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        records.append(meta)

    summary = {
        "num_samples": len(records),
        "output_dir": str(out_dir),
        "note": "第一版仅导出 ROI 图像与 terrain；mask_sem_path 预留给后续 stage1 预测语义掩码填充。",
    }
    dump_json(summary, out_dir / "pseudo_dataset_summary.json")
    print(f"[OK] pseudo dataset built: {out_dir}, n={len(records)}")


if __name__ == "__main__":
    main()