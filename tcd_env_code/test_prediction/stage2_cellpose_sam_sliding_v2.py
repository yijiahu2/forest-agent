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

# -----------------------------
# 固定阈值（论文）
# -----------------------------
FLOW_THR = 1.0
CELLPROB_THR = 0.0

# -----------------------------
# 数据特性：NoData=256
# -----------------------------
NODATA_VAL = 256

# -----------------------------
# 小碎片过滤（像素面积）
# -----------------------------
MIN_AREA = 50


def read_rgb_window(ds, win):
    rgb = ds.read([1, 2, 3], window=win)  # CHW
    nodata = (rgb == NODATA_VAL).any(axis=0)  # HW bool
    rgb = np.transpose(rgb, (1, 2, 0))  # HWC
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if nodata.any():
        rgb[nodata] = 0
    return rgb, nodata


def read_mask_window(ds, win):
    m = ds.read(1, window=win)
    return (m > 0).astype(np.uint8)


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


def feather_weight(h, w, overlap):
    """tile 中心权重大、边缘权重小（0~1）"""
    yy, xx = np.mgrid[0:h, 0:w]
    d_top = yy
    d_left = xx
    d_bottom = (h - 1) - yy
    d_right = (w - 1) - xx
    d_edge = np.minimum(np.minimum(d_top, d_bottom), np.minimum(d_left, d_right)).astype(np.float32)
    if overlap <= 0:
        return np.ones((h, w), np.float32)
    wgt = np.clip(d_edge / float(overlap), 0.0, 1.0)
    return wgt


def export_instances_to_shp(label_img, transform, crs, out_shp, min_area_px=50):
    mask = label_img > 0
    geoms, ids, areas_px = [], [], []

    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area <= 0:
        pixel_area = 1.0

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

    gdf = gpd.GeoDataFrame({"id": ids, "area_px": areas_px}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve(by="id", as_index=False, aggfunc={"area_px": "sum"})
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="UTF-8")
    print("Saved:", out_shp, "features:", len(gdf))


def cellpose_eval_safe(cp, img, diameter, flow_thr, cellprob_thr,
                       channels, bsize=256, tile_overlap=0.35, augment=True, niter=0):
    """
    兼容不同 cellpose 版本：尽量传入 bsize/tile_overlap/augment/niter，不支持则自动回退。
    """
    kwargs = dict(
        diameter=float(diameter),
        channels=channels,
        flow_threshold=float(flow_thr),
        cellprob_threshold=float(cellprob_thr),
    )
    # 尝试添加高级参数（版本不支持就回退）
    for k, v in [("bsize", bsize), ("tile_overlap", tile_overlap), ("augment", augment), ("niter", niter)]:
        kwargs[k] = v
    try:
        masks, flows, styles = cp.eval(img, **kwargs)
        return masks
    except TypeError:
        # 回退到基础参数
        kwargs_basic = dict(
            diameter=float(diameter),
            channels=channels,
            flow_threshold=float(flow_thr),
            cellprob_threshold=float(cellprob_thr),
        )
        masks, flows, styles = cp.eval(img, **kwargs_basic)
        return masks


def merge_tile_multiscale(masks_list, iou_merge_thr=0.2, min_area=50):
    """
    将多个尺度得到的 masks 融合到一个 tile mask 中（tile 内部融合）。
    策略：
    - 从小到大依次合并
    - 若新实例与已有实例 IoU >= 阈值，则合并为同一实例（保留已有ID）
    - 否则新增实例ID
    """
    tile = np.zeros_like(masks_list[0], dtype=np.int32)
    next_id = 1

    for masks in masks_list:
        masks = masks.astype(np.int32)
        masks = remove_small_instances(masks, min_area)
        if masks.max() == 0:
            continue

        # 将 masks 的局部ID逐个处理
        for lid in range(1, int(masks.max()) + 1):
            nm = (masks == lid)
            if nm.sum() < min_area:
                continue

            # 找 tile 中可能重叠的候选 old id
            cand = np.unique(tile[nm])
            cand = cand[cand > 0]

            best_iou, best_oid = 0.0, None
            for oid in cand:
                val = iou(nm, tile == oid)
                if val > best_iou:
                    best_iou, best_oid = val, int(oid)

            if best_oid is not None and best_iou >= float(iou_merge_thr):
                tile[nm] = best_oid
            else:
                tile[nm] = next_id
                next_id += 1

    return tile


def assign_global_ids(tile_local, old_global, iou_merge_thr, global_next_id):
    """
    将 tile_local(1..K) 映射到全局ID：
    - 与 old_global(全局实例ID) 在 overlap 区域 IoU>=thr 的，映射到对应 old id
    - 否则分配新全局ID
    """
    tile_global = np.zeros_like(tile_local, dtype=np.int32)

    local_ids = np.unique(tile_local)
    local_ids = local_ids[local_ids > 0]
    if local_ids.size == 0:
        return tile_global, global_next_id

    for lid in local_ids:
        lm = (tile_local == lid)
        cand_old = np.unique(old_global[lm])
        cand_old = cand_old[cand_old > 0]

        best_iou, best_oid = 0.0, None
        for oid in cand_old:
            val = iou(lm, old_global == oid)
            if val > best_iou:
                best_iou, best_oid = val, int(oid)

        if best_oid is not None and best_iou >= float(iou_merge_thr):
            tile_global[lm] = best_oid
        else:
            tile_global[lm] = int(global_next_id)
            global_next_id += 1

    return tile_global, global_next_id


def main(in_tif, msem_tif, out_dir,
         diam_list=(96.0, 160.0, 256.0),
         iou_merge_thr=0.2,
         tile=1536, overlap=384,
         bsize=256, tile_overlap=0.35, augment=True, niter=0,
         use_gray=True):

    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("diam_list:", diam_list)
    print("TILE:", tile, "OVERLAP:", overlap)
    print("cellpose bsize:", bsize, "tile_overlap:", tile_overlap, "augment:", augment, "niter:", niter)

    step = tile - overlap
    assert step > 0, "tile must be > overlap"

    # Cellpose-SAM
    cp = models.CellposeModel(gpu=(device == "cuda"), pretrained_model="cpsam")

    with rasterio.open(in_tif) as src, rasterio.open(msem_tif) as ms:
        H, W = src.height, src.width
        profile = src.profile.copy()

        inst_full = np.zeros((H, W), dtype=np.int32)
        weight_full = np.zeros((H, W), dtype=np.float32)
        next_id = 1

        for y in range(0, H, step):
            for x in range(0, W, step):
                h = min(tile, H - y)
                w = min(tile, W - x)
                win = Window(x, y, w, h)

                rgb, nodata = read_rgb_window(src, win)
                msem = read_mask_window(ms, win)

                valid = (msem == 1) & (~nodata)

                # ---- ROI 软填充：用有效区 median 填 ROI 外，避免黑边假边界 ----
                rgb_in = rgb.copy()
                if valid.any():
                    med = np.median(rgb_in[valid], axis=0).astype(np.uint8)
                    rgb_in[~valid] = med
                else:
                    rgb_in[:] = 0

                # ---- 可选：灰度/彩色输入 ----
                # Cellpose channels 机制与“细胞/核”范式相关；为稳健起见默认用灰度 [0,0]
                if use_gray:
                    img_in = rgb_in
                    channels = [0, 0]
                else:
                    # 仍然传入 RGB，但保持 channels=[0,0]（避免版本差异导致通道解释错误）
                    img_in = rgb_in
                    channels = [0, 0]

                # ---- pad 到 tile（边缘用 reflect，减少边缘伪影）----
                pad_h = tile - h
                pad_w = tile - w
                if pad_h > 0 or pad_w > 0:
                    img_in = cv2.copyMakeBorder(img_in, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                    valid_pad = np.pad(valid.astype(np.uint8), ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0).astype(bool)
                else:
                    valid_pad = valid

                # ---- 多尺度推理 ----
                masks_list = []
                for d in diam_list:
                    masks = cellpose_eval_safe(
                        cp, img_in, diameter=d,
                        flow_thr=FLOW_THR, cellprob_thr=CELLPROB_THR,
                        channels=channels,
                        bsize=bsize, tile_overlap=tile_overlap, augment=augment, niter=niter
                    )
                    masks = masks[:h, :w].astype(np.int32)

                    # 只保留 valid 区域内实例（避免 ROI 外污染）
                    if valid.any():
                        masks[~valid] = 0
                    masks_list.append(masks)

                # ---- tile 内融合（多尺度）----
                tile_local = merge_tile_multiscale(masks_list, iou_merge_thr=iou_merge_thr, min_area=MIN_AREA)
                if tile_local.max() == 0:
                    continue

                # ---- 映射到全局ID（与当前全局 overlap 区匹配）----
                old = inst_full[y:y + h, x:x + w]
                tile_global, next_id = assign_global_ids(tile_local, old, iou_merge_thr=iou_merge_thr, global_next_id=next_id)

                # ---- Feather 权重拼接（中心优先，边缘弱）----
                w_tile = feather_weight(h, w, overlap=overlap)

                # 仅在 valid 区域参与拼接
                w_tile = w_tile * valid.astype(np.float32)

                # 更新规则：权重大者胜
                w_old = weight_full[y:y + h, x:x + w]
                take = (w_tile > w_old) & (tile_global > 0)

                old_updated = old.copy()
                old_updated[take] = tile_global[take]

                w_updated = w_old.copy()
                w_updated[take] = w_tile[take]

                inst_full[y:y + h, x:x + w] = old_updated
                weight_full[y:y + h, x:x + w] = w_updated

                print(f"[tile] x={x} y={y} local_max={int(tile_local.max())} global_next_id={next_id}")

        # 写 GeoTIFF
        out_tif = os.path.join(out_dir, "Y_inst.tif")
        profile.update(count=1, dtype=rasterio.int32, nodata=0, compress="LZW", photometric="MINISBLACK")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(inst_full.astype(np.int32), 1)
        print("Saved:", out_tif)

        # 导出 Shapefile
        out_shp = os.path.join(out_dir, "Y_inst.shp")
        export_instances_to_shp(inst_full, transform=src.transform, crs=src.crs, out_shp=out_shp, min_area_px=MIN_AREA)

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

    # 多尺度 diameter 列表
    ap.add_argument("--diam_list", type=str, default="96,160,256",
                    help='comma-separated diameters, e.g. "80,140,220"')

    # 合并阈值
    ap.add_argument("--iou_merge_thr", type=float, default=0.2)

    # 滑窗参数（3090 24GB 默认更大）
    ap.add_argument("--tile", type=int, default=1536)
    ap.add_argument("--overlap", type=int, default=384)

    # cellpose 内部 tiling 参数
    ap.add_argument("--bsize", type=int, default=256)
    ap.add_argument("--tile_overlap", type=float, default=0.35)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no_augment", action="store_true", default=False)
    ap.add_argument("--niter", type=int, default=0)

    # 输入模式
    ap.add_argument("--use_rgb", action="store_true", default=False,
                    help="pass RGB (still channels=[0,0] for stability); default uses grayscale style")

    args = ap.parse_args()

    diam_list = tuple(float(x) for x in args.diam_list.split(",") if x.strip() != "")
    augment = (not args.no_augment)

    main(
        args.in_tif,
        args.msem_tif,
        args.out_dir,
        diam_list=diam_list,
        iou_merge_thr=args.iou_merge_thr,
        tile=args.tile,
        overlap=args.overlap,
        bsize=args.bsize,
        tile_overlap=args.tile_overlap,
        augment=augment,
        niter=args.niter,
        use_gray=(not args.use_rgb),
    )