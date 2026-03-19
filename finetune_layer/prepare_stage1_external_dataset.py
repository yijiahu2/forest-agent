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


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap must be smaller than tile_size, got {overlap} >= {tile_size}")

    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _extract_padded_tile(arr: np.ndarray, y: int, x: int, tile_size: int, pad_value: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    y1 = min(y + tile_size, h)
    x1 = min(x + tile_size, w)
    tile = arr[y:y1, x:x1]

    if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
        return tile

    if arr.ndim == 3:
        out = np.full((tile_size, tile_size, arr.shape[2]), pad_value, dtype=arr.dtype)
    else:
        out = np.full((tile_size, tile_size), pad_value, dtype=arr.dtype)
    out[: tile.shape[0], : tile.shape[1]] = tile
    return out


def _iter_tile_samples(
    image_rgb: np.ndarray,
    mask_bin: np.ndarray,
    tile_size: int,
    overlap: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if image_rgb.shape[:2] != mask_bin.shape[:2]:
        raise ValueError(
            f"Image/mask shape mismatch: image={image_rgb.shape[:2]}, mask={mask_bin.shape[:2]}"
        )

    h, w = mask_bin.shape
    y_starts = _tile_starts(h, tile_size, overlap)
    x_starts = _tile_starts(w, tile_size, overlap)

    tiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    tile_idx = 0
    for y in y_starts:
        for x in x_starts:
            image_tile = _extract_padded_tile(image_rgb, y, x, tile_size, pad_value=0)
            mask_tile = _extract_padded_tile(mask_bin, y, x, tile_size, pad_value=0)
            suffix = f"tile_{tile_idx:04d}_y{y}_x{x}"
            tiles.append((suffix, image_tile, mask_tile))
            tile_idx += 1
    return tiles


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

    df["source_split"] = df["split"].astype(str)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df.loc[~df["split"].isin(["train", "val", "test"]), "split"] = "train"

    if not (df["split"] == "train").any():
        df.loc[df.index[0], "split"] = "train"

    train_idx = list(df.index[df["split"] == "train"])
    if not (df["split"] == "val").any() and train_idx:
        val_src = train_idx[-1]
        if len(df) >= 2:
            df.loc[val_src, "split"] = "val"
            train_idx = [idx for idx in train_idx if idx != val_src]

    if not (df["split"] == "test").any() and len(df) >= 3 and train_idx:
        test_src = train_idx[-1]
        df.loc[test_src, "split"] = "test"

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
    tile_size = int(cfg.get("external_dataset_tile_size", 1024))
    tile_overlap = int(cfg.get("external_dataset_tile_overlap", 256))

    out_dir = Path(cfg["output_dir"]) / "external_stage1_dataset"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    image_id = 1
    for _, row in df.reset_index(drop=True).iterrows():
        split = str(row["split"]).strip().lower()
        roi_id = str(row["roi_id"])
        image_src = Path(row["image_path"])
        mask_src = Path(row["mask_sem_path"])

        if not image_src.exists():
            raise FileNotFoundError(f"image_path not found: {image_src}")
        if not mask_src.exists():
            raise FileNotFoundError(f"mask_sem_path not found: {mask_src}")

        image_rgb = _read_image_rgb_uint8(image_src)
        mask_bin = _read_mask_binary(mask_src)
        tile_samples = _iter_tile_samples(
            image_rgb=image_rgb,
            mask_bin=mask_bin,
            tile_size=tile_size,
            overlap=tile_overlap,
        )

        for tile_suffix, image_tile, mask_tile in tile_samples:
            tile_roi_id = f"{roi_id}_{tile_suffix}"
            image_dst = images_dir / f"{tile_roi_id}.png"
            mask_dst = masks_dir / f"{tile_roi_id}.png"

            image_dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image_tile, mode="RGB").save(image_dst)
            _save_mask_png(mask_tile, mask_dst)

            height, width = mask_tile.shape
            annotation = _binary_mask_to_coco_annotation(
                mask_bin=mask_tile,
                image_id=image_id,
                ann_id=image_id,
                category_id=1,
            )

            rec = {
                "id": image_id,
                "roi_id": tile_roi_id,
                "source_roi_id": roi_id,
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
                "foreground_pixels": int(mask_tile.sum()),
            }

            records.append(rec)
            split_records[split].append(rec)
            image_id += 1

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
        "num_source_rois": int(len(df)),
        "has_independent_test_split": bool(len(split_records["test"]) > 0),
        "non_empty_annotations_train": sum(r["annotation"] is not None for r in split_records["train"]),
        "non_empty_annotations_val": sum(r["annotation"] is not None for r in split_records["val"]),
        "non_empty_annotations_test": sum(r["annotation"] is not None for r in split_records["test"]),
        "outputs": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "train_json": str(out_dir / "train.json"),
            "val_json": str(out_dir / "val.json"),
            "test_json": str(out_dir / "test.json"),
        },
        "tiling": {
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
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
