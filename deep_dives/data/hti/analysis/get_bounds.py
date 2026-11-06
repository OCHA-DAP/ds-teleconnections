import sys, ocha_stratus as stratus
out = sys.argv[1]
for lvl in (0, 1):
    gdf = stratus.load_shp_from_blob("ds-aa-hti-hurricanes/raw/codab/hti.shp.zip", shapefile=f"hti_adm{lvl}.shp", stage="dev")
    print(lvl, gdf.crs, gdf.shape, [c for c in gdf.columns if "PCODE" in c.upper() or "_FR" in c.upper()][:6])
    if lvl == 1: print(gdf[[c for c in gdf.columns if c.upper() in ("ADM1_PCODE","ADM1_FR","ADM1_EN")]].to_string())
    print(gdf.total_bounds)
    gdf.to_file(f"{out}/hti_adm{lvl}.gpkg")
