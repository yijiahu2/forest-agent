import os
import numpy as np
import rasterio
import torch
import cv2

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_PATH = "/home/xth/tcd/models/tcd-segformer-mit-b5"

###############################################
# 参数
###############################################
TILE_SIZE = 1024
OVERLAP = 256
THR = 0.25

# shp后处理参数（按需改）
MIN_AREA_M2 = 0.0     # 小面积过滤阈值（单位：平方米），0表示不过滤
SIMPLIFY_TOL = 0.0    # 简化容差（单位：地图单位，米/度），0表示不简化


###############################################
# 工具函数
###############################################
def to_uint8_rgb(arr):
    rgb = arr[:3].astype(np.float32)

    if rgb.max() > 255:
        out = np.zeros_like(rgb, dtype=np.uint8)
        for c in range(3):
            v = rgb[c]
            lo, hi = np.percentile(v, (1, 99))
            if hi <= lo:
                hi = lo + 1
            v = (v - lo) / (hi - lo)
            v = np.clip(v, 0, 1) * 255
            out[c] = v.astype(np.uint8)
        rgb = out
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return np.transpose(rgb, (1, 2, 0))


###############################################
# SegFormer 推理
###############################################
@torch.no_grad()
def predict_tile(rgb, model, processor, device):
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0]

    probs = torch.softmax(logits, dim=0)
    canopy = probs[1]
    mask = (canopy >= THR).cpu().numpy().astype(np.uint8)
    return mask


###############################################
# 滑窗推理
###############################################
def sliding_window_predict(img, model, processor, device):
    H, W = img.shape[:2]
    step = TILE_SIZE - OVERLAP

    result = np.zeros((H, W), np.uint8)
    weight = np.zeros((H, W), np.float32)

    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = min(y + TILE_SIZE, H)
            x1 = min(x + TILE_SIZE, W)

            tile = img[y:y1, x:x1]

            pad_h = TILE_SIZE - tile.shape[0]
            pad_w = TILE_SIZE - tile.shape[1]

            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(
                    tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
                )

            pred = predict_tile(tile, model, processor, device)
            pred = pred[: y1 - y, : x1 - x]

            result[y:y1, x:x1] += pred
            weight[y:y1, x:x1] += 1

    result = result / np.maximum(weight, 1e-6)
    return (result > 0.5).astype(np.uint8)


###############################################
# 写 tif
###############################################
def write_tif(out_path, data, profile):
    profile = profile.copy()
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress="LZW",
        nodata=0,
        photometric="MINISBLACK",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


###############################################
# ✅ mask -> shp（核心新增）
###############################################
def mask_to_shp(mask, transform, crs, out_shp,
                min_area_m2=0.0, simplify_tol=0.0):
    """
    mask: (H,W) uint8 0/1
    transform/crs: 来自 rasterio dataset
    out_shp: 输出shp路径
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    os.makedirs(os.path.dirname(out_shp), exist_ok=True)

    # 只矢量化值为1的区域
    geom_iter = shapes(
        mask.astype(np.uint8),
        mask=(mask == 1),
        transform=transform
    )

    geoms = []
    vals = []
    for geom, val in geom_iter:
        if int(val) != 1:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        if simplify_tol and simplify_tol > 0:
            g = g.simplify(simplify_tol, preserve_topology=True)
        if g.is_empty:
            continue
        geoms.append(g)
        vals.append(int(val))

    if len(geoms) == 0:
        # 输出一个空shp也行，但更直观是提示
        print("[WARN] No canopy polygons found, skip shp.")
        return

    gdf = gpd.GeoDataFrame(
        {"class": vals},
        geometry=geoms,
        crs=crs
    )

    # 面积过滤（仅当是投影坐标系时更有意义；经纬度面积不是真实m²）
    if min_area_m2 and min_area_m2 > 0:
        try:
            areas = gdf.geometry.area
            gdf = gdf.loc[areas >= float(min_area_m2)].copy()
        except Exception as e:
            print("[WARN] area filter failed (likely geographic CRS):", e)

    gdf.reset_index(drop=True, inplace=True)

    # 写出shp（可同时得到 .shx .dbf .prj）
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
    print("Saved:", out_shp)
    print("Polygons:", len(gdf))


###############################################
# 主函数
###############################################
def main(in_tif, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH, local_files_only=True
    ).to(device).eval()

    with rasterio.open(in_tif) as src:
        img = src.read()
        rgb = to_uint8_rgb(img)
        print("Image size:", rgb.shape)

        mask = sliding_window_predict(rgb, model, processor, device)

        out_tif = os.path.join(out_dir, "M_sem.tif")
        write_tif(out_tif, mask, src.profile)

        # ✅ 新增：输出 shp
        out_shp = os.path.join(out_dir, "M_sem.shp")
        mask_to_shp(
            mask=mask,
            transform=src.transform,
            crs=src.crs,
            out_shp=out_shp,
            min_area_m2=MIN_AREA_M2,
            simplify_tol=SIMPLIFY_TOL
        )

    ########################################
    # PNG 可视化
    ########################################
    png = (mask * 255).astype(np.uint8)
    out_png = os.path.join(out_dir, "M_sem.png")
    cv2.imwrite(out_png, png)

    print("Saved:", out_tif)
    print("Saved:", out_png)
    print("Canopy fraction:", mask.mean())


###############################################
# CLI
###############################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tif", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    main(args.in_tif, args.out_dir)