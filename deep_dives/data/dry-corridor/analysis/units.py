"""Units for the Dry Corridor page: 3 countries (adm0), 54 departments (adm1) and the three framework groups."""
import geopandas as gpd, pandas as pd
AOI = {"GT_AOI": ("GTM", ["GT20", "GT19", "GT21", "GT02"], "Guatemala framework departments"),
       "HN_AOI": ("HND", ["HN07", "HN08"], "Honduras framework departments"),
       "SV_AOI": ("SLV", ["SV01", "SV12"], "El Salvador framework departments")}
CNAME = {"GTM": "Guatemala", "HND": "Honduras", "SLV": "El Salvador"}
def load_units(bdir):
    parts = []
    for iso in ("GTM", "HND", "SLV"):
        a0 = gpd.read_file(f"{bdir}/{iso.lower()}_adm0.gpkg")
        a1 = gpd.read_file(f"{bdir}/{iso.lower()}_adm1.gpkg")
        parts.append(gpd.GeoDataFrame({"unit": a0.ADM0_PCODE, "name": CNAME[iso], "iso3": iso, "level": 0}, geometry=a0.geometry, crs=a0.crs))
        parts.append(gpd.GeoDataFrame({"unit": a1.ADM1_PCODE, "name": a1.ADM1_ES, "iso3": iso, "level": 1}, geometry=a1.geometry, crs=a1.crs))
        for k, (i3, pcs, nm) in AOI.items():
            if i3 == iso:
                g = a1[a1.ADM1_PCODE.isin(pcs)]; assert len(g) == len(pcs), (k, g.ADM1_PCODE.tolist())
                parts.append(gpd.GeoDataFrame({"unit": [k], "name": [nm], "iso3": [iso], "level": [9]}, geometry=[g.union_all()], crs=a1.crs))
    return pd.concat(parts, ignore_index=True)
