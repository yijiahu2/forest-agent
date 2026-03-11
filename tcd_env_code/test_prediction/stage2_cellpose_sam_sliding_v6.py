import os
import math
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
# 固定阈值（论文默认）
# -----------------------------
FLOW_THR = 1.0
CELLPROB_THR = 0.0

# -----------------------------
# 数据特性：NoData=256
# -----------------------------
NODATA_VAL = 256

# -----------------------------
# tile 内基础碎片过滤
# -----------------------------
MIN_AREA_TILE = 50


def check_alignment(src, ms, strict=True):
    ok_size = (src.width == ms.width and src.height == ms.height)
    ok_crs = (src.crs == ms.crs)
    ok_transform = (src.transform == ms.transform)

    if not (ok_size and ok_crs and ok_transform):
        msg = (
            "[WARN] in_tif 与 msem_tif 可能未完全对齐：\n"
            f"  src:  size=({src.width},{src.height}) crs={src.crs} transform={src.transform}\n"
            f"  msem: size=({ms.width},{ms.height}) crs={ms.crs} transform={ms.transform}\n"
            "若 Stage2 出现“掩膜看着对但实例不出”的情况，请先把 M_sem 重采样到 in_tif 网格。"
        )
        print(msg)
        if strict:
            raise RuntimeError("in_tif 与 msem_tif 不同网格/不同尺寸/不同 transform。")


def infer_gsd_m(src, user_gsd_m=None):
    """
    优先使用用户显式传入的 gsd_m。
    否则仅当 CRS 为投影坐标系且单位通常为米时，用 transform 估计。
    """
    if user_gsd_m is not None and float(user_gsd_m) > 0:
        return float(user_gsd_m)

    if src.crs is not None and src.crs.is_projected:
        # 对北向上栅格通常 a 为像元宽，e 为像元高（负号）
        gsd_x = abs(float(src.transform.a))
        gsd_y = abs(float(src.transform.e))
        if gsd_x > 0 and gsd_y > 0:
            return (gsd_x + gsd_y) / 2.0

    raise ValueError(
        "无法可靠推断 GSD。请显式传入 --gsd_m，例如 --gsd_m 0.022701909"
    )


def get_pixel_area_m2(gsd_m):
    return float(gsd_m) * float(gsd_m)


def area_m2_to_equiv_diameter_m(area_m2):
    if area_m2 <= 0:
        return 0.0
    return float(math.sqrt(4.0 * area_m2 / math.pi))


def diameter_m_to_area_px(diam_m, pixel_area_m2):
    """
    把等效圆直径（m）换成像素面积阈值。
    """
    area_m2 = math.pi * (float(diam_m) / 2.0) ** 2
    return float(area_m2 / pixel_area_m2)


def read_rgb_window(ds, win):
    rgb = ds.read([1, 2, 3], window=win)  # CHW
    nodata = (rgb == NODATA_VAL).all(axis=0)  # 更稳：三个通道都=256才视为nodata
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
    yy, xx = np.mgrid[0:h, 0:w]
    d_top = yy
    d_left = xx
    d_bottom = (h - 1) - yy
    d_right = (w - 1) - xx
    d_edge = np.minimum(np.minimum(d_top, d_bottom), np.minimum(d_left, d_right)).astype(np.float32)

    if overlap <= 0:
        return np.ones((h, w), np.float32)
    return np.clip(d_edge / float(overlap), 0.0, 1.0)


def cellpose_eval_safe(cp, img, diameter, flow_thr, cellprob_thr, channels,
                       bsize=256, tile_overlap=0.35, augment=True, niter=0):
    """
    尽量使用 bsize/tile_overlap/augment/niter；
    若当前 cellpose 版本不支持，则自动回退基础参数。
    """
    kwargs = dict(
        diameter=float(diameter),
        channels=channels,
        flow_threshold=float(flow_thr),
        cellprob_threshold=float(cellprob_thr),
        bsize=int(bsize),
        tile_overlap=float(tile_overlap),
        augment=bool(augment),
        niter=int(niter),
    )
    try:
        masks, flows, styles = cp.eval(img, **kwargs)
        return masks
    except TypeError:
        kwargs_basic = dict(
            diameter=float(diameter),
            channels=channels,
            flow_threshold=float(flow_thr),
            cellprob_threshold=float(cellprob_thr),
        )
        masks, flows, styles = cp.eval(img, **kwargs_basic)
        return masks


def merge_tile_multiscale(masks_list, iou_merge_thr=0.25, min_area=50):
    """
    tile 内多尺度融合：
    - IoU >= 阈值 -> 视作同一实例
    - 否则新增实例
    """
    if len(masks_list) == 0:
        return None

    tile = np.zeros_like(masks_list[0], dtype=np.int32)
    next_id = 1

    for masks in masks_list:
        masks = masks.astype(np.int32)
        masks = remove_small_instances(masks, min_area)
        if masks.max() == 0:
            continue

        for lid in range(1, int(masks.max()) + 1):
            nm = (masks == lid)
            if nm.sum() < min_area:
                continue

            cand = np.unique(tile[nm])
            cand = cand[cand > 0]

            best_i, best_oid = 0.0, None
            for oid in cand:
                val = iou(nm, tile == oid)
                if val > best_i:
                    best_i, best_oid = val, int(oid)

            if best_oid is not None and best_i >= float(iou_merge_thr):
                tile[nm] = best_oid
            else:
                tile[nm] = next_id
                next_id += 1

    return tile


def assign_global_ids(tile_local, old_global, iou_merge_thr, global_next_id):
    """
    tile 实例映射到全局 ID：
    - 与已有全局实例 IoU >= 阈值 -> 映射到已有 ID
    - 否则新建全局 ID
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

        best_i, best_oid = 0.0, None
        for oid in cand_old:
            val = iou(lm, old_global == oid)
            if val > best_i:
                best_i, best_oid = val, int(oid)

        if best_oid is not None and best_i >= float(iou_merge_thr):
            tile_global[lm] = best_oid
        else:
            tile_global[lm] = int(global_next_id)
            global_next_id += 1

    return tile_global, global_next_id


def post_filter_instances_by_area_px(inst_full, min_area_px=0.0, max_area_px=0.0, relabel=True):
    """
    全图后处理：按像素面积范围过滤。
    """
    inst = inst_full.astype(np.int32, copy=True)
    max_id = int(inst.max())
    if max_id <= 0:
        return inst

    areas_px = np.bincount(inst.ravel())
    if areas_px.size < max_id + 1:
        areas_px = np.pad(areas_px, (0, max_id + 1 - areas_px.size), mode="constant")
    areas_px[0] = 0

    keep = np.ones(max_id + 1, dtype=bool)
    keep[0] = False

    if min_area_px and min_area_px > 0:
        keep &= (areas_px >= float(min_area_px))
    if max_area_px and max_area_px > 0:
        keep &= (areas_px <= float(max_area_px))

    lut = np.zeros(max_id + 1, dtype=np.int32)
    if relabel:
        kept_ids = np.nonzero(keep)[0]
        lut[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.int32)
    else:
        lut[keep] = np.where(keep)[0].astype(np.int32)

    return lut[inst]


def export_instances_to_shp(label_img, transform, crs, out_shp, pixel_area_m2):
    mask = label_img > 0
    geoms, ids, areas_px, areas_m2, diam_m_list = [], [], [], [], []

    for geom, val in features.shapes(label_img.astype(np.int32), mask=mask, transform=transform):
        vid = int(val)
        if vid <= 0:
            continue

        poly = shape(geom)
        if poly.is_empty:
            continue

        area_m2 = float(poly.area)
        if area_m2 <= 0:
            continue

        area_px = area_m2 / pixel_area_m2
        diam_m = area_m2_to_equiv_diameter_m(area_m2)

        geoms.append(poly)
        ids.append(vid)
        areas_px.append(area_px)
        areas_m2.append(area_m2)
        diam_m_list.append(diam_m)

    if len(geoms) == 0:
        print("[shp] No polygons to write (empty instances).")
        return

    gdf = gpd.GeoDataFrame(
        {
            "id": ids,
            "area_px": areas_px,
            "area_m2": areas_m2,
            "diam_m": diam_m_list,
        },
        geometry=geoms,
        crs=crs
    )

    gdf = gdf.dissolve(
        by="id",
        as_index=False,
        aggfunc={"area_px": "sum", "area_m2": "sum", "diam_m": "max"}
    )
    gdf["diam_m"] = np.sqrt(4.0 * gdf["area_m2"].astype(float) / math.pi)
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="UTF-8")
    print("Saved:", out_shp, "features:", len(gdf))


def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return tuple()
    return tuple(float(x) for x in str(s).split(",") if str(x).strip() != "")


def main(
    in_tif,
    msem_tif,
    out_dir,
    gsd_m=None,
    diam_list=(160.0, 256.0),
    iou_merge_thr=0.25,
    tile=2048,
    overlap=512,
    bsize=256,
    tile_overlap=0.35,
    augment=True,
    niter=0,
    min_crown_diam_m=4.0,
    max_crown_diam_m=10.0,
    strict_align_check=False,
    debug=False,
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    cp = models.CellposeModel(gpu=(device == "cuda"), pretrained_model="cpsam")

    with rasterio.open(in_tif) as src, rasterio.open(msem_tif) as ms:
        check_alignment(src, ms, strict=strict_align_check)

        gsd_m = infer_gsd_m(src, gsd_m)
        pixel_area_m2 = get_pixel_area_m2(gsd_m)

        min_area_px = diameter_m_to_area_px(min_crown_diam_m, pixel_area_m2) if min_crown_diam_m > 0 else 0.0
        max_area_px = diameter_m_to_area_px(max_crown_diam_m, pixel_area_m2) if max_crown_diam_m > 0 else 0.0

        print("gsd_m:", gsd_m)
        print("diam_list:", diam_list)
        print("TILE:", tile, "OVERLAP:", overlap, "STEP:", tile - overlap)
        print("cellpose:", f"bsize={bsize} tile_overlap={tile_overlap} augment={augment} niter={niter}")
        print("diam_filter_m:", f"{min_crown_diam_m} ~ {max_crown_diam_m}")
        print("area_filter_px:", f"{min_area_px:.1f} ~ {max_area_px:.1f}")

        step = tile - overlap
        if step <= 0:
            raise ValueError("tile must be > overlap")

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
                canopy = (msem == 1) & (~nodata)

                if not canopy.any():
                    if debug:
                        print(f"[dbg] x={x} y={y} canopy=0 skip")
                    continue

                if debug:
                    print(f"[dbg] x={x} y={y} canopy_frac={canopy.mean():.3f} use_diams={diam_list}")

                # ROI 外软填充，避免硬黑边
                rgb_in = rgb.copy()
                med = np.median(rgb_in[canopy], axis=0).astype(np.uint8)
                rgb_in[~canopy] = med

                img_in = rgb_in
                channels = [0, 0]

                pad_h = tile - h
                pad_w = tile - w
                if pad_h > 0 or pad_w > 0:
                    img_in = cv2.copyMakeBorder(img_in, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                masks_list = []
                for d in diam_list:
                    masks = cellpose_eval_safe(
                        cp, img_in, diameter=d,
                        flow_thr=FLOW_THR,
                        cellprob_thr=CELLPROB_THR,
                        channels=channels,
                        bsize=bsize,
                        tile_overlap=tile_overlap,
                        augment=augment,
                        niter=niter,
                    )
                    masks = masks[:h, :w].astype(np.int32)
                    masks[~canopy] = 0
                    masks_list.append(masks)

                tile_local = merge_tile_multiscale(
                    masks_list,
                    iou_merge_thr=iou_merge_thr,
                    min_area=MIN_AREA_TILE
                )

                if tile_local is None or tile_local.max() == 0:
                    continue

                old = inst_full[y:y + h, x:x + w]
                tile_global, next_id = assign_global_ids(
                    tile_local, old,
                    iou_merge_thr=iou_merge_thr,
                    global_next_id=next_id
                )

                w_tile = feather_weight(h, w, overlap=overlap) * canopy.astype(np.float32)
                w_old = weight_full[y:y + h, x:x + w]
                take = (w_tile > w_old) & (tile_global > 0)

                if take.any():
                    old_updated = old.copy()
                    old_updated[take] = tile_global[take]
                    inst_full[y:y + h, x:x + w] = old_updated

                    w_updated = w_old.copy()
                    w_updated[take] = w_tile[take]
                    weight_full[y:y + h, x:x + w] = w_updated

        # 全图后处理：按真实 4~10m 约束换算出的像素面积过滤
        inst_full = post_filter_instances_by_area_px(
            inst_full,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            relabel=True
        )

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

        out_shp = os.path.join(out_dir, "Y_inst.shp")
        export_instances_to_shp(
            inst_full,
            transform=src.transform,
            crs=src.crs,
            out_shp=out_shp,
            pixel_area_m2=pixel_area_m2
        )

        out_stats = os.path.join(out_dir, "Y_inst_stats.txt")
        with open(out_stats, "w", encoding="utf-8") as f:
            f.write(f"instances={int(inst_full.max())}\n")
            f.write(f"gsd_m={gsd_m}\n")
            f.write(f"pixel_area_m2={pixel_area_m2:.10f}\n")
            f.write(f"diam_list={','.join(map(str, diam_list))}\n")
            f.write(f"min_crown_diam_m={min_crown_diam_m}\n")
            f.write(f"max_crown_diam_m={max_crown_diam_m}\n")
            f.write(f"min_area_px={min_area_px:.4f}\n")
            f.write(f"max_area_px={max_area_px:.4f}\n")
        print("Saved:", out_stats)
        print("instances:", int(inst_full.max()))

    out_png = os.path.join(out_dir, "Y_inst_color.png")
    max_id = int(inst_full.max())
    rng = np.random.default_rng(0)
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    if max_id > 0:
        lut[1:] = rng.integers(0, 255, size=(max_id, 3), dtype=np.uint8)
    color = lut[inst_full]
    cv2.imwrite(out_png, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print("Saved:", out_png)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--in_tif", required=True)
    ap.add_argument("--msem_tif", required=True)
    ap.add_argument("--out_dir", required=True)

    # 显式 GSD，推荐直接传
    ap.add_argument("--gsd_m", type=float, default=None,
                    help="ground sampling distance in meters/pixel, e.g. 0.022701909")

    # 收窄后的多尺度，避免过分割
    ap.add_argument("--diam_list", type=str, default="160,256",
                    help='comma-separated diameters, e.g. "160,256"')

    ap.add_argument("--iou_merge_thr", type=float, default=0.25)

    # 外部滑窗
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=512)

    # Cellpose 内部 tiling
    ap.add_argument("--bsize", type=int, default=256)
    ap.add_argument("--tile_overlap", type=float, default=0.35)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no_augment", action="store_true", default=False)
    ap.add_argument("--niter", type=int, default=0)

    # 冠幅直径（米）
    ap.add_argument("--min_crown_diam_m", type=float, default=4.0)
    ap.add_argument("--max_crown_diam_m", type=float, default=10.0)

    ap.add_argument("--strict_align_check", action="store_true", default=False)
    ap.add_argument("--debug", action="store_true", default=False)

    args = ap.parse_args()

    diam_list = parse_float_list(args.diam_list)
    augment = (not args.no_augment)

    main(
        in_tif=args.in_tif,
        msem_tif=args.msem_tif,
        out_dir=args.out_dir,
        gsd_m=args.gsd_m,
        diam_list=diam_list,
        iou_merge_thr=args.iou_merge_thr,
        tile=args.tile,
        overlap=args.overlap,
        bsize=args.bsize,
        tile_overlap=args.tile_overlap,
        augment=augment,
        niter=args.niter,
        min_crown_diam_m=args.min_crown_diam_m,
        max_crown_diam_m=args.max_crown_diam_m,
        strict_align_check=args.strict_align_check,
        debug=args.debug,
    )