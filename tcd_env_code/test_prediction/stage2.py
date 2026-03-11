import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features
import cv2
import torch
from cellpose import models
from shapely.geometry import shape
import geopandas as gpd

# ---- 论文固定阈值 ----
FLOW_THR = 1.0
CELLPROB_THR = 0.0

# ---- 大图滑窗参数（显存不够可调小 TILE）----
TILE = 1024
OVERLAP = 256
STEP = TILE - OVERLAP

# ---- 数据特性：NoData=256 ----
NODATA_VAL = 256

# ---- 小碎片过滤（像素面积）----
MIN_AREA = 50


def read_rgb_window(ds, win):
    # 读 RGB
    rgb = ds.read([1, 2, 3], window=win)  # CHW
    # 把 nodata=256 的像素识别出来（任一通道=256 就算无效）
    nodata = (rgb == NODATA_VAL).any(axis=0)  # HW bool

    rgb = np.transpose(rgb, (1, 2, 0))  # HWC
    # 你的值域本来就在 0~246，直接转 uint8 最稳
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # nodata 置零（背景）
    if nodata.any():
        rgb[nodata] = 0
    return rgb, nodata


def read_mask_window(ds, win):
    m = ds.read(1, window=win)
    m = (m > 0).astype(np.uint8)
    return m


def remove_small_instances(lbl, min_area):
    if lbl.max() == 0:
        return lbl.astype(np.int32)
    out = np.zeros_like(lbl, dtype=np.int32)
    new_id = 1
    for k in range(1, int(lbl.max()) + 1):
        m = (lbl == k)
        if int(m.sum()) >= min_area:
            out[m] = new_id
            new_id += 1
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def export_instances_to_shp(label_img, transform, crs, out_shp, min_area_px=50):
    """
    将实例ID栅格(0背景)矢量化输出为 Shapefile。
    - label_img: HxW int32，像元值为实例ID
    - transform/crs: 来自原始影像，保证坐标正确
    - out_shp: 输出 shp 路径
    - min_area_px: 过滤小碎片（像素面积）
    """
    mask = label_img > 0

    geoms = []
    ids = []
    areas_px = []

    # 像元面积（单位：坐标单位^2；北向上影像一般 transform.a * transform.e 为负）
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area <= 0:
        pixel_area = 1.0

    # shapes 会输出每个连通块的 polygon；同一个 id 若被切成多块，会有多条记录
    for geom, val in features.shapes(label_img.astype(np.int32), mask=mask, transform=transform):
        vid = int(val)
        if vid <= 0:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue

        area_px = float(poly.area / pixel_area)
        if area_px < float(min_area_px):
            continue

        geoms.append(poly)
        ids.append(vid)
        areas_px.append(area_px)

    if len(geoms) == 0:
        print("[shp] No polygons to write (empty instances).")
        return

    gdf = gpd.GeoDataFrame(
        {"id": ids, "area_px": areas_px},
        geometry=geoms,
        crs=crs
    )

    # 合并同一个 id 的多段多边形，让“每棵树 = 1 条要素”
    gdf = gdf.dissolve(by="id", as_index=False, aggfunc={"area_px": "sum"})

    # 写 Shapefile
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="UTF-8")
    print("Saved:", out_shp, "features:", len(gdf))


def main(in_tif, msem_tif, out_dir, diameter_px=200.0, iou_merge_thr=0.2):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("diameter_px:", diameter_px)

    cp = models.CellposeModel(gpu=(device == "cuda"), pretrained_model="cpsam")

    with rasterio.open(in_tif) as src, rasterio.open(msem_tif) as ms:
        H, W = src.height, src.width
        profile = src.profile.copy()

        inst_full = np.zeros((H, W), dtype=np.int32)
        next_id = 1

        for y in range(0, H, STEP):
            for x in range(0, W, STEP):
                h = min(TILE, H - y)
                w = min(TILE, W - x)
                win = Window(x, y, w, h)

                rgb, nodata = read_rgb_window(src, win)
                msem = read_mask_window(ms, win)

                # Stage1 ROI + nodata 双重约束
                valid = (msem == 1) & (~nodata)
                rgb_masked = rgb.copy()
                rgb_masked[~valid] = 0

                # pad 到 TILE
                pad_h = TILE - h
                pad_w = TILE - w
                if pad_h > 0 or pad_w > 0:
                    rgb_masked = cv2.copyMakeBorder(
                        rgb_masked, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                    )

                # Cellpose-SAM
                masks, flows, styles = cp.eval(
                    rgb_masked,
                    diameter=float(diameter_px),
                    channels=[0, 0],  # 灰度模式更稳
                    flow_threshold=FLOW_THR,
                    cellprob_threshold=CELLPROB_THR,
                )

                masks = masks[:h, :w].astype(np.int32)
                masks = remove_small_instances(masks, MIN_AREA)

                old = inst_full[y:y + h, x:x + w]

                if old.max() == 0:
                    if masks.max() > 0:
                        masks[masks > 0] += (next_id - 1)
                        next_id = int(masks.max()) + 1
                    inst_full[y:y + h, x:x + w] = np.maximum(old, masks)
                    continue

                # offset 新实例
                if masks.max() > 0:
                    masks_off = masks.copy()
                    masks_off[masks_off > 0] += (next_id - 1)
                    new_ids = np.unique(masks_off)
                    new_ids = new_ids[new_ids > 0]
                    tentative_next = int(masks_off.max()) + 1
                else:
                    masks_off = masks
                    new_ids = np.array([], dtype=np.int32)
                    tentative_next = next_id

                overlap = (old > 0) & (masks_off > 0)
                if overlap.any() and new_ids.size > 0:
                    mapping = {}
                    for nid in new_ids:
                        nm = (masks_off == nid)
                        if not (nm & overlap).any():
                            continue
                        cand_old = np.unique(old[nm])
                        cand_old = cand_old[cand_old > 0]
                        best_i, best_oid = 0.0, None
                        for oid in cand_old:
                            val = iou(nm, old == oid)
                            if val > best_i:
                                best_i, best_oid = val, int(oid)
                        if best_oid is not None and best_i >= iou_merge_thr:
                            mapping[int(nid)] = best_oid

                    for nid, oid in mapping.items():
                        masks_off[masks_off == nid] = oid

                merged = old.copy()
                write_new = (merged == 0) & (masks_off > 0)
                merged[write_new] = masks_off[write_new]
                inst_full[y:y + h, x:x + w] = merged

                next_id = max(next_id, tentative_next)
                print(f"[tile] x={x} y={y} tile_inst={int(masks.max())} next_id={next_id}")

        # 写 GeoTIFF
        out_tif = os.path.join(out_dir, "Y_inst.tif")
        profile.update(
            count=1,
            dtype=rasterio.int32,
            nodata=0,
            compress="LZW",
            photometric="MINISBLACK"
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(inst_full.astype(np.int32), 1)

        print("Saved:", out_tif)

        # 导出 Shapefile（单木实例矢量）
        out_shp = os.path.join(out_dir, "Y_inst.shp")
        export_instances_to_shp(
            inst_full,
            transform=src.transform,
            crs=src.crs,
            out_shp=out_shp,
            min_area_px=MIN_AREA
        )

    # 彩色 PNG 预览
    out_png = os.path.join(out_dir, "Y_inst_color.png")
    max_id = int(inst_full.max())
    rng = np.random.default_rng(0)
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    if max_id > 0:
        lut[1:] = rng.integers(0, 255, size=(max_id, 3), dtype=np.uint8)
    color = lut[inst_full]
    cv2.imwrite(out_png, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    print("Saved:", out_png)
    print("instances:", int(inst_full.max()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tif", required=True)
    ap.add_argument("--msem_tif", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--diameter_px", type=float, default=200.0)
    ap.add_argument("--iou_merge_thr", type=float, default=0.2)
    args = ap.parse_args()

    main(
        args.in_tif,
        args.msem_tif,
        args.out_dir,
        diameter_px=args.diameter_px,
        iou_merge_thr=args.iou_merge_thr
    )