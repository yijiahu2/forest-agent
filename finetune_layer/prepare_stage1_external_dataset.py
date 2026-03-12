from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pycocotools import mask as mask_utils

from finetune_layer.io_utils import dump_json, load_csv, load_yaml


def _read_image_shape(path: Path) -> tuple[int, int]:
    with rasterio.open(path) as src:
        return int(src.height), int(src.width)


def _read_image_rgb_uint8(image_src: Path) -> np.ndarray:
    with rasterio.open(image_src) as src:
        arr = src.read()  # [C, H, W]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected image array shape: {arr.shape}, file={image_src}")

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    else:
        raise ValueError(f"Invalid band count: {arr.shape[0]}, file={image_src}")

    arr = arr.astype(np.float32)

    out = np.zeros_like(arr, dtype=np.uint8)
    for i in range(3):
        band = arr[i]
        finite = np.isfinite(band)
        if not finite.any():
            continue

        vals = band[finite]
        lo = np.percentile(vals, 2)
        hi = np.percentile(vals, 98)

        if hi <= lo:
            hi = vals.max()
            lo = vals.min()

        if hi <= lo:
            out[i] = 0
        else:
            scaled = (band - lo) / (hi - lo)
            scaled = np.clip(scaled, 0, 1)
            out[i] = (scaled * 255).astype(np.uint8)

    return np.transpose(out, (1, 2, 0))  # [H, W, C]


def _save_image_png(image_src: Path, image_dst: Path) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    rgb = _read_image_rgb_uint8(image_src)
    Image.fromarray(rgb, mode="RGB").save(image_dst)


def _read_mask_binary(mask_src: Path) -> np.ndarray:
    with rasterio.open(mask_src) as src:
        arr = src.read(1)
    return (arr > 0).astype(np.uint8)


def _save_mask_png(mask_bin: np.ndarray, mask_dst: Path) -> None:
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_bin * 255).astype(np.uint8), mode="L").save(mask_dst)


def _binary_mask_to_coco_annotation(
    mask_bin: np.ndarray,
    image_id: int,
    ann_id: int,
    category_id: int = 1,
) -> dict[str, Any] | None:
    if mask_bin.max() == 0:
        return None

    mask_fortran = np.asfortranarray(mask_bin.astype(np.uint8))
    rle = mask_utils.encode(mask_fortran)
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()

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

        image_dst = images_dir / f"{roi_id}.png"
        mask_dst = masks_dir / f"{roi_id}.png"

        _save_image_png(image_src, image_dst)

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