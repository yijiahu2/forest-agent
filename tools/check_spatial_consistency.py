from pathlib import Path
import rasterio
import geopandas as gpd
from shapely.geometry import box

DOM = "/mnt/e/Learning_documents/DEM/dom197.tif"
DEM = "/mnt/e/Learning_documents/DEM/shanxia_DEM1m.tif"    
SHP = "/mnt/e/Learning_documents/output/shp/xiaoban29pro.shp"  # 改成你的实际路径

def print_raster_info(path):
    with rasterio.open(path) as src:
        print("=" * 90)
        print("RASTER:", path)
        print("CRS:", src.crs)
        print("BOUNDS:", src.bounds)
        print("WIDTH, HEIGHT:", src.width, src.height)
        return src.crs, src.bounds

def print_vector_info(path):
    gdf = gpd.read_file(path)
    print("=" * 90)
    print("VECTOR:", path)
    print("CRS:", gdf.crs)
    print("BOUNDS:", tuple(gdf.total_bounds))
    print("FEATURE COUNT:", len(gdf))
    return gdf.crs, gdf.total_bounds, gdf

def bounds_intersect(b1, b2):
    # b = (left, bottom, right, top)
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

def main():
    dom_crs, dom_bounds = print_raster_info(DOM)
    dem_crs, dem_bounds = print_raster_info(DEM)
    shp_crs, shp_bounds, shp_gdf = print_vector_info(SHP)

    dom_box = box(dom_bounds.left, dom_bounds.bottom, dom_bounds.right, dom_bounds.top)

    print("\n[1] CRS consistency")
    print("DOM == DEM CRS:", dom_crs == dem_crs)
    print("DOM == SHP CRS:", str(dom_crs) == str(shp_crs) or shp_gdf.to_crs(dom_crs).crs == dom_crs)

    print("\n[2] Bounds intersection")
    dom_bounds_tuple = (dom_bounds.left, dom_bounds.bottom, dom_bounds.right, dom_bounds.top)
    dem_bounds_tuple = (dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    shp_bounds_tuple = tuple(shp_bounds)

    print("DOM intersects DEM:", bounds_intersect(dom_bounds_tuple, dem_bounds_tuple))
    print("DOM intersects SHP bounds:", bounds_intersect(dom_bounds_tuple, shp_bounds_tuple))

    print("\n[3] Geometry intersection after reprojection")
    shp_in_dom_crs = shp_gdf.to_crs(dom_crs)
    shp_union = shp_in_dom_crs.unary_union
    print("DOM intersects SHP geometry:", dom_box.intersects(shp_union))

    print("\n[4] Containment checks")
    print("DOM within DEM:", (
        dom_bounds.left >= dem_bounds.left and
        dom_bounds.right <= dem_bounds.right and
        dom_bounds.bottom >= dem_bounds.bottom and
        dom_bounds.top <= dem_bounds.top
    ))

if __name__ == "__main__":
    main()