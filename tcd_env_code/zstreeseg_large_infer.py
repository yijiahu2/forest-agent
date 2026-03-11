import os
import sys
import numpy as np
import rasterio
import torch
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon
import geopandas as gpd
from skimage.measure import find_contours
from scipy import ndimage
from skimage.segmentation import watershed

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from cellpose import models


# =========================
# 参数 (针对密植林优化)
# =========================

MODEL_PATH = "/home/xth/tcd/models/tcd-segformer-mit-b5"

TILE = 1024
OVERLAP = 256

THRESHOLD = 0.5

CELLPOSE_DIAMETER = 30
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = -2


# =========================
# 读取影像
# =========================

def read_rgb(src, window):

    img = src.read([1,2,3], window=window)

    img = np.transpose(img,(1,2,0))

    if img.dtype != np.uint8:

        img = img.astype(np.float32)

        lo,hi = np.percentile(img,(1,99))

        img = (img-lo)/(hi-lo)

        img = np.clip(img,0,1)*255

        img = img.astype(np.uint8)

    return img


# =========================
# SegFormer
# =========================

def load_segformer():

    processor = AutoImageProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    return processor,model,device


@torch.no_grad()
def run_segformer(rgb,processor,model,device):

    inputs = processor(images=rgb,return_tensors="pt").to(device)

    out = model(**inputs)

    logits = out.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=rgb.shape[:2],
        mode="bilinear",
        align_corners=False
    )[0]

    probs = torch.softmax(logits,dim=0)

    canopy = probs[1]

    mask = (canopy>=THRESHOLD).cpu().numpy().astype(np.uint8)

    return mask


# =========================
# Cellpose
# =========================

def load_cellpose():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.CellposeModel(
        gpu=(device=="cuda"),
        pretrained_model="cpsam"
    )

    return model


def run_cellpose(rgb,msem,model):

    rgb = rgb.copy()

    rgb[msem==0] = 0

    kernel = np.ones((3,3),np.uint8)
    rgb = cv2.morphologyEx(rgb,cv2.MORPH_OPEN,kernel)

    masks,flows,styles = model.eval(

        rgb,

        diameter=CELLPOSE_DIAMETER,

        flow_threshold=FLOW_THRESHOLD,

        cellprob_threshold=CELLPROB_THRESHOLD
    )

    return masks


# =========================
# watershed 分裂树冠
# =========================

def split_crowns(mask):

    binary = mask>0

    distance = ndimage.distance_transform_edt(binary)

    labels = watershed(
        -distance,
        mask,
        mask=binary
    )

    return labels


# =========================
# 滑窗
# =========================

def sliding_windows(width,height):

    step = TILE - OVERLAP

    xs = list(range(0,width,step))
    ys = list(range(0,height,step))

    for y in ys:
        for x in xs:

            yield x,y


# =========================
# 实例拼接
# =========================

def merge_instances(global_mask,tile_mask,x,y):

    tile = tile_mask.copy()

    ids = np.unique(tile)

    for i in ids:

        if i==0:
            continue

        tile[tile==i] = global_mask.max()+1

    h,w = tile.shape

    global_mask[y:y+h,x:x+w] = np.maximum(
        global_mask[y:y+h,x:x+w],
        tile
    )


# =========================
# 保存 raster
# =========================

def save_raster(path,arr,profile,dtype):

    if os.path.exists(path):
        os.remove(path)

    p = {
        "driver":"GTiff",
        "height":arr.shape[0],
        "width":arr.shape[1],
        "count":1,
        "dtype":dtype,
        "crs":profile["crs"],
        "transform":profile["transform"],
        "compress":"LZW"
    }

    with rasterio.open(path,"w",**p) as dst:
        dst.write(arr,1)


# =========================
# shp 导出
# =========================

def export_shp(mask,profile,out_shp):

    transform = profile["transform"]

    geoms=[]
    ids=[]

    for i in tqdm(range(1,mask.max()+1)):

        binary = (mask==i).astype(np.uint8)

        contours = find_contours(binary,0.5)

        for c in contours:

            coords=[]

            for y,x in c:

                X,Y = rasterio.transform.xy(transform,int(y),int(x))

                coords.append((X,Y))

            if len(coords)>3:

                poly = Polygon(coords)

                geoms.append(poly)

                ids.append(i)

    gdf = gpd.GeoDataFrame(
        {"id":ids,"geometry":geoms},
        crs=profile["crs"]
    )

    gdf.to_file(out_shp)


# =========================
# 统计
# =========================

def crown_statistics(mask,profile,out_txt):

    pixel_area = abs(profile["transform"][0] * profile["transform"][4])

    ids = np.unique(mask)
    ids = ids[ids!=0]

    areas = []

    for i in ids:

        area_pixels = np.sum(mask==i)

        areas.append(area_pixels * pixel_area)

    tree_count = len(ids)

    mean_area = np.mean(areas)

    with open(out_txt,"w") as f:

        f.write(f"Tree count: {tree_count}\n")
        f.write(f"Mean crown area: {mean_area:.2f} m2\n")

    print("Tree count:",tree_count)


# =========================
# 主流程
# =========================

def main(in_tif,out_dir):

    os.makedirs(out_dir,exist_ok=True)

    processor,segformer,device = load_segformer()

    cellpose = load_cellpose()

    with rasterio.open(in_tif) as src:

        W = src.width
        H = src.height

        profile = src.profile

        M_sem = np.zeros((H,W),np.uint8)

        Y_inst = np.zeros((H,W),np.int32)

        for x,y in tqdm(sliding_windows(W,H)):

            window = rasterio.windows.Window(
                x,y,
                min(TILE,W-x),
                min(TILE,H-y)
            )

            rgb = read_rgb(src,window)

            msem = run_segformer(rgb,processor,segformer,device)

            inst = run_cellpose(rgb,msem,cellpose)

            inst = split_crowns(inst)

            merge_instances(Y_inst,inst,x,y)

            h,w = msem.shape

            M_sem[y:y+h,x:x+w] = np.maximum(
                M_sem[y:y+h,x:x+w],
                msem
            )

    save_raster(os.path.join(out_dir,"M_sem.tif"),M_sem,profile,"uint8")

    save_raster(os.path.join(out_dir,"Y_inst.tif"),Y_inst,profile,"int32")

    export_shp(
        Y_inst,
        profile,
        os.path.join(out_dir,"tree_crowns.shp")
    )

    crown_statistics(
        Y_inst,
        profile,
        os.path.join(out_dir,"crown_stats.txt")
    )

    print("Finished")


if __name__ == "__main__":

    main(sys.argv[1],sys.argv[2])