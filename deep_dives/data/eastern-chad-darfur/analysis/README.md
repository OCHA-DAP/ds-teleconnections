# Eastern Chad and Darfur 2026 — analysis

Produces `../page_data.json`, the only input of `ecd_deep_dive.py`. Run on 17 November 2026.

Area: Chad's Ouaddaï, Wadi Fira and Sila; Sudan's five Darfur states (`aoi.py`, COD-AB from FieldMaps).
Work files go to `cache/eastern-chad-darfur/` (gitignored; override with `ECD_WORK`).

Needs: the dev DB tunnel (`~/bin/db-tunnel up`; `db.py` points stratus at it), dev blob access through
`ocha-stratus`, and the packed 0.5° rainfall cubes of `ds-precip-intercomparison` (`cache/packed/*.npz`)
cloned next to this repo. Extra packages beyond the repo's: `rasterio rioxarray psycopg2-binary shapely`.

```bash
cd deep_dives/data/eastern-chad-darfur/analysis
W=../../../../cache/eastern-chad-darfur; mkdir -p $W/raw/codab
curl -so $W/raw/codab/sdn.shp.zip https://data.fieldmaps.io/cod/originals/sdn.shp.zip
curl -so $W/raw/codab/tcd.shp.zip https://data.fieldmaps.io/cod/originals/tcd.shp.zip
# WorldPop 2018 1 km clipped to the AOI box (dev blob raster container) -> $W/raw/worldpop_2018_bbox.tif
R="uv run --with rasterio --with rioxarray --with psycopg2-binary --with shapely python"
$R 01_rainfall.py     # CHIRPS v2/v3 (+ERA5, IMERG Late) per side and state; CHC for Aug-Oct 2026
./fetch_veg.sh $W     # FAO ASIS + JRC ASAP raw pulls (ASAP's server is slow)
$R 02_vegetation.py   # season values per side
$R 03_yields.py       # FEWS NET Data Warehouse crop statistics, Darfur
$R 04_fewsnet.py      # FEWS NET maps (regional archive to 2021, FDW packages after) -> population shares
$R 05_ipc_ch.py       # CH (HDX consolidated file) + IPC (team mirror ipc.population)
$R 06_hnrp.py         # HNRP severity + people in need (team mirror hpc.*)
$R 07_build_data.py   # -> ../page_data.json
```

Each script's docstring states its choices. Gotchas found on the way:

- IPC's June–September 2026 projection classifies 48% of Darfur's population while the `all` row stays at
  full scope: divide by the phase sum (05 does) and report the coverage. IPC calls Central Darfur
  "Central Darfur State" from 2026.
- CH (HDX): keep `usethisperiod == 'Y'`; refugee and returnee rows have no p-code; Dar Tama and Iriba were
  not analysed in March 2026, so HDX uses the November 2025 projection for them.
- FEWS NET: maps before April 2011 use FEWS NET's own severity scale (left out); the regional shapefile
  archive ends mid-2021 and the Data Warehouse packages answer `[]` for an exact date even when a round
  exists under another date; no maps from December 2024 to October 2025.
- FAO ASIS reports Darfur as the three pre-2012 states and Wadi Fira as Biltine. JRC ASAP: Sudan
  `country_id` 165, Chad 184.
- Darfur yields: the level of the series shifts several times, so yields are % of the surrounding years
  (±5, same source). 2014–2015 and 2018–2019 look wrong.
- d3-geo needs clockwise rings (07 orients them).
