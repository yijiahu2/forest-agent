from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pycocotools import mask as mask_utils

from finetune_layer.io_utils import dump_json, load_csv, load_yaml


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def _read_image_shape(path: Path) -> tuple[int, int]:
    with rasterio.open(path) as src:
        return int(src.height), int(src.width)


def _read_mask_binary(mask_src: Path) -> np.ndarray:
    with rasterio.open(mask_src) as src:
        arr = src.read(1)
    return (arr > 0).astype(np.uint8)


def _save_mask_png(mask_bin: np.ndarray, mask_dst: Path) -> None:
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_bin, mode="L").save(mask_dst)


def _binary_mask_to_coco_annotation(
    mask_bin: np.ndarray,
    image_id: int,
    ann_id: int,
    category_id: int = 1,
) -> dict[str, Any] | None:
    if mask_bin.max() == 0:
        return None

    # COCO RLE 要求 Fortran order
    mask_fortran = np.asfortranarray(mask_bin.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()

    # json 序列化需要把 bytes 转成 str
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }


def _ensure_split(df):
    df = df.copy()
    if "split" not in df.columns:
        df["split"] = "train"

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df.loc[~df["split"].isin(["train", "val", "test"]), "split"] = "train"

    has_val = (df["split"] == "val").any()
    if not has_val and len(df) >= 2:
        df.loc[df.index[-1], "split"] = "val"

    return df


def _build_coco_split(records: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    images = []
    annotations = []
    ann_id = 1

    for rec in records:
        image_id = rec["id"]
        images.append(
            {
                "id": image_id,
                "file_name": rec["image_relpath"],
                "width": rec["width"],
                "height": rec["height"],
            }
        )

        ann = rec.get("annotation")
        if ann is not None:
            ann = dict(ann)
            ann["id"] = ann_id
            annotations.append(ann)
            ann_id += 1

    return {
        "info": {
            "description": f"forest-agent pseudo dataset for TCD semantic fine-tuning ({split_name})",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "tree",
                "supercategory": "vegetation",
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pseudo_dir = Path(cfg["output_dir"]) / "pseudo_dataset"
    manifest_csv = pseudo_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_csv}")

    df = load_csv(manifest_csv)
    df = df[df["has_mask"] == True].copy()  # noqa
    if len(df) == 0:
        raise RuntimeError("No usable samples with masks for external stage1 training.")

    df = _ensure_split(df)

    out_dir = Path(cfg["output_dir"]) / "external_stage1_dataset"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for i, row in df.reset_index(drop=True).iterrows():
        split = str(row["split"]).strip().lower()
        roi_id = str(row["roi_id"])
        image_src = Path(row["image_path"])
        mask_src = Path(row["mask_sem_path"])

        if not image_src.exists():
            raise FileNotFoundError(f"image_path not found: {image_src}")
        if not mask_src.exists():
            raise FileNotFoundError(f"mask_sem_path not found: {mask_src}")

        image_dst = images_dir / f"{roi_id}{image_src.suffix.lower()}"
        mask_dst = masks_dir / f"{roi_id}.png"

        _safe_link_or_copy(image_src, image_dst)

        mask_bin = _read_mask_binary(mask_src)
        _save_mask_png(mask_bin, mask_dst)

        height, width = _read_image_shape(image_src)
        image_id = i + 1

        annotation = _binary_mask_to_coco_annotation(
            mask_bin=mask_bin,
            image_id=image_id,
            ann_id=image_id,
            category_id=1,
        )

        rec = {
            "id": image_id,
            "roi_id": roi_id,
            "split": split,
            "image_src": str(image_src),
            "mask_src": str(mask_src),
            "image_dst": str(image_dst),
            "mask_dst": str(mask_dst),
            "image_relpath": image_dst.name,
            "mask_relpath": mask_dst.name,
            "width": width,
            "height": height,
            "xbh": row.get("XBH"),
            "annotation": annotation,
            "foreground_pixels": int(mask_bin.sum()),
        }

        records.append(rec)
        split_records[split].append(rec)

    train_json = _build_coco_split(split_records["train"], "train")
    val_json = _build_coco_split(split_records["val"], "val")
    test_json = _build_coco_split(split_records["test"], "test")

    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)

    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_json, f, ensure_ascii=False, indent=2)

    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)

    summary = {
        "status": "prepared",
        "dataset_root": str(out_dir),
        "num_samples": len(records),
        "num_train": len(split_records["train"]),
        "num_val": len(split_records["val"]),
        "num_test": len(split_records["test"]),
        "non_empty_annotations_train": sum(r["annotation"] is not None for r in split_records["train"]),
        "non_empty_annotations_val": sum(r["annotation"] is not None for r in split_records["val"]),
        "outputs": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "train_json": str(out_dir / "train.json"),
            "val_json": str(out_dir / "val.json"),
            "test_json": str(out_dir / "test.json"),
        },
        "records": records,
    }
    dump_json(summary, out_dir / "external_dataset_summary.json")
    print(
        f"[OK] TCD external stage1 dataset prepared: {out_dir}, "
        f"train={len(split_records['train'])}, val={len(split_records['val'])}, test={len(split_records['test'])}"
    )


if __name__ == "__main__":
    main()