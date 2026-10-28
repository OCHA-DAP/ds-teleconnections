"""Digitize INAMEH's map of rainy-season onset (Mapa de inicio de la temporada lluviosa) into five zone polygons.

Input: inameh_inicio_temporada_lluviosa.png (INAMEH, Coordinación de Meteorología Aplicada; shared by a colleague, 2026-10).
Output: ven_inameh_rainy_season_zones.geojson, read by ven_deep_dive.py. Run from the repo root:

    uv run python deep_dives/data/ven_inameh_zones.py

How: each pixel is classed by hue and saturation (relief shading changes brightness, not hue) against the five legend
colours; a 9×9-pixel majority filter and nearest-neighbour fill remove labels and lines; the map is georeferenced by fitting a
lon/lat scale and offset that best overlaps the zone pixels with the country outline (CODAB, IoU 0.63; the fit is in GEOREF);
water drawn in light blue (lakes, Guri, sea round Margarita) is pulled out of zones it cannot belong to, per INAMEH's legend.
Checked against the map, and against the legend where it names a state (e.g. Mérida 100% B, Falcón 53% D / 46% E, Nueva
Esparta mostly E; its small islands Coche and Cubagua are lost at 0.025°), and against the CHIRPS climatology (rains start
in the order A, B, C, D, E). Edges are approximate: use it on the 0.5° grid, not for maps of
municipalities.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
LEGEND = {"A": (185, 222, 209), "B": (138, 202, 220), "C": (99, 169, 169), "D": (142, 179, 222), "E": (77, 135, 210)}
GEOREF = (136.75, 150.0, 34.35, 35.45)   # pixel column = x0 + sx·(lon + 70); row = y0 − sy·(lat − 10)
FRAME = (20, 25, 555, 500)               # map frame in the image (left, top, right, bottom), legend excluded
RES = 0.025


def classify(img: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(img).convert("RGB"))
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(float)
    h, s, v = hsv[..., 0] / 255 * 360, hsv[..., 1] / 255, hsv[..., 2] / 255
    proto = np.asarray(Image.fromarray(np.array([list(LEGEND.values())], np.uint8)).convert("HSV")).astype(float)[0]
    ph, ps = proto[:, 0] / 255 * 360, proto[:, 1] / 255
    d = np.hypot((np.abs(((h[..., None] - ph + 180) % 360) - 180)) / 30.0, np.abs(s[..., None] - ps) / 0.15)
    zone = np.where((d.min(-1) < 1.2) & (v > 0.25) & (s > 0.08), d.argmin(-1), -1)
    out = np.full_like(zone, -1)
    l, t, r, b = FRAME
    out[t:b, l:r] = zone[t:b, l:r]
    cnt = np.stack([ndimage.uniform_filter((out == k).astype(float), 9) for k in range(len(LEGEND))])
    maj = np.where(cnt.max(0) > 0, cnt.argmax(0), -1)
    idx = ndimage.distance_transform_edt(maj < 0, return_distances=False, return_indices=True)
    return maj[idx[0], idx[1]]


def zone_grid(pixels: np.ndarray, states: gpd.GeoDataFrame) -> np.ndarray:
    x0, y0, sx, sy = GEOREF
    lon = np.arange(-74, -59, RES) + RES / 2
    lat = np.arange(13, 0, -RES) - RES / 2
    lon2, lat2 = np.meshgrid(lon, lat)
    h, w = pixels.shape
    zg = pixels[np.clip(np.round(y0 - sy * (lat2 - 10)).astype(int), 0, h - 1),
                np.clip(np.round(x0 + sx * (lon2 + 70)).astype(int), 0, w - 1)]
    # where each zone can be, from INAMEH's legend: A the far south and the south-west Apure/Barinas strip; B the west
    # (Andes, Apure, Barinas, south of Lake Maracaibo) and southern Bolívar; E the northern coast and islands
    wrong = (((zg == 0) & (lat2 > 8.0)) | ((zg == 1) & ~(((lon2 < -67) & (lat2 < 9.3)) | (lat2 < 6.8))) | ((zg == 4) & (lat2 < 10.0)))
    good = (zg >= 0) & ~wrong
    j = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
    zg = np.where(wrong, zg[j[0], j[1]], zg)
    inside = rasterize(list(states.geometry), out_shape=zg.shape, transform=from_origin(-74, 13, RES, RES), fill=0) > 0
    return np.where(inside, zg, -1).astype("int16")


def main() -> None:
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    import ven_deep_dive as V
    zg = zone_grid(classify(HERE / "inameh_inicio_temporada_lluviosa.png"), V.states())
    tr = from_origin(-74, 13, RES, RES)
    rows = [dict(zone=z, geometry=unary_union([shape(g) for g, v in shapes(zg, mask=zg == k, transform=tr) if v == k]).simplify(0.01))
            for k, z in enumerate(LEGEND)]
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(HERE / "ven_inameh_rainy_season_zones.geojson", driver="GeoJSON")
    print("wrote", HERE / "ven_inameh_rainy_season_zones.geojson")


if __name__ == "__main__":
    main()
