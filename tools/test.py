from geo_layer.terrain_constraints import (
    classify_slope_class,
    classify_aspect_class,
    classify_landform_type,
    classify_slope_position_class,
)

print("坡度测试")
for v in [3, 10, 20, 30, 40, 50]:
    print(v, classify_slope_class(v))

print("坡向测试")
tests = [
    (10, 10),   # 北坡
    (45, 10),   # 东北坡
    (90, 10),   # 东坡
    (135, 10),  # 东南坡
    (180, 10),  # 南坡
    (225, 10),  # 西南坡
    (270, 10),  # 西坡
    (315, 10),  # 西北坡
    (100, 3),   # 无坡向
]
for aspect, slope in tests:
    print(aspect, slope, classify_aspect_class(aspect, slope))

print("地貌测试")
for elev, relief in [(50, 10), (80, 100), (150, 80), (300, 120), (700, 200), (1500, 300)]:
    print(elev, relief, classify_landform_type(elev, relief))

print("坡位测试")
tests2 = [
    (2, 0.1, None, None),
    (20, 0.9, 0.5, None),
    (20, 0.7, 0.2, None),
    (20, 0.5, 0.0, None),
    (20, 0.2, -0.4, None),
]
for slope, rel, tpi, fa in tests2:
    print(slope, rel, tpi, fa, classify_slope_position_class(slope, rel, tpi, fa))






python -m geo_layer.spatial_context \
  --dom_tif /mnt/e/Learning_documents/DEM/dom197.tif \
  --dem_tif /mnt/e/Learning_documents/DEM/shanxia_DEM1m.tif \
  --xiaoban_shp /mnt/e/Learning_documents/output/shp/xiaoban29pro.shp \
  --out_dir /mnt/e/Learning_documents/output/test_spatial_context \
  --xiaoban_id_field XBH \
  --tree_count_field LMSL \
  --crown_field PJGF \
  --closure_field YBD \
  --area_ha_field MJ_hm2