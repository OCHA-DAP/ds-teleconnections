# Eastern Chad and Darfur 2026 — analysis

Produces `processed/page_data.json` in the work folder, the only input of `ecd_deep_dive.py`. Run on 17 November 2026.

Area: Chad's Ouaddaï, Wadi Fira and Sila; Sudan's five Darfur states (`aoi.py`, COD-AB from FieldMaps).
Work files go to `cache/eastern-chad-darfur/` (gitignored; override with `ECD_WORK`). **No data is committed**: the
whole work folder is mirrored on the dev blob at `projects/ds-teleconnections/eastern-chad-darfur/` with the same
relative paths. `uv run python blob_sync.py pull` restores it; `blob_sync.py push` after any rerun. The builders fetch
the files they need from blob when they are not local.

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
P="uv run --with rasterio --with pystac-client --with planetary-computer --with geopandas python"
$P 08_ndvi.py         # MODIS NDVI greening 2001-2026 from Planetary Computer (~2,000 tile reads, cached)
$R 09_greening.py     # detrended greening: per side/year, per state 2026, the 2026 map
$R 07_build_data.py   # -> $W/processed/page_data.json
uv run python blob_sync.py push   # mirror the work folder on the dev blob
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
- Planetary Computer MODIS: pick items by id (`MOD13A1.*`), not `platform`; overview level 1 of a
  500 m tile = about 2 km. Detrended greening follows the Chad evaluation-villages recipe
  (cell-normalised, trend fitted per unit over 2001–2025).
- d3-geo needs clockwise rings (07 orients them); maps are equirectangular so the greening raster
  lines up with the outlines.

## Restricted conflict page (`ecd_conflict.py`, builder `ecdc`)

Seasons and local violence, /enso/eastern-chad-darfur-conflict/, encrypted with staticrypt
(`SITE_PASSWORD`). Built from ACLED events, whose terms limit redistribution: **nothing derived from
them is committed** (no data, no findings in the template or commit messages). Everything stays in
`$W/processed/conflict/`:

```
uv run python 10_acled.py      # ACLED Chad + Sudan, all years (ACLED_USERNAME/ACLED_PASSWORD) -> $W/raw/acled/
uv run python 12_cast.py       # ACLED CAST forecasts, Chad + Sudan                         -> $W/raw/acled/
# UCDP GED 25.1 (https://ucdp.uu.se/downloads/ged/ged251-csv.zip, read zipped)  -> $W/raw/ucdp/
uv run python 11_conflict.py   # needs 09_greening outputs; -> $W/processed/conflict/conflict_data.json
# $W/processed/conflict/text.json: curated text (hand-check, literature, forecasting models), by hand
uv run python blob_sync.py push
cd ../../../.. && ECD_WORK=$W SITE_PASSWORD=... uv run python ecd_conflict.py --encrypt
```

Gotchas:
- ACLED API pages with a cursor (`next_cursor`), not `page=`; `inter1/inter2` are text labels.
- ACLED's militia labels for the same kind of attack change over time, so classes exclude only formal
  war parties (`conflict_classes.py`). Camels as transport ("riding camels") are not livestock.
- Coverage, labelling and CAST notes that come from looking at the data are in `text.json` (`notes`),
  not here: this repo is public.
- The conflict data and text exist only in the work folder and on blob: push after every change.

### Pooled Sudano-Sahelian test (section "The wider Sahel" on the restricted page)

Design written first: `$W/processed/belt/prespec_2026-11-18.md` (with its amendments); hand-check in
`$W/processed/belt/handcheck_2026-11-18.md`.

```
uv run python 10_acled.py belt                       # ACLED for the other belt countries -> $W/raw/acled/acled_belt.parquet
P="uv run --with pystac-client --with planetary-computer --with rasterio --with geopandas python"
$P 08_ndvi.py belt                                   # MODIS greening, 4 km, belt box (~3,700 reads, ~10 min)
./fetch_belt_veg.sh $W                               # ASIS (ASI, VHI) + ASAP (cropland, rangeland) admin-1 series
# provider boundaries -> $W/raw/belt/asap/gaul1_asap_v06.zip (ASAP download page), $W/raw/belt/gaul/g2015_2014_1.zip (FAO GAUL 2015)
# CHIRPS v2 0.5 deg cube (projects/ds-precip-intercomparison/processed/grid05/chirps_v2.nc) -> $W/raw/belt/chirps_v2_grid05.nc
uv run --with h5netcdf python 13_belt_cells.py       # cells, mask, rain, own + upstream greening
uv run python 14_belt_conflict.py                    # tests, per country, matrix -> $W/processed/belt/belt_data.json (~10 min)
uv run python blob_sync.py push
```
Gotchas: one belt-wide STAC search times out (08 searches in 10-degree chunks); FAO ASIS csvs are Latin-1;
ASIS units are GAUL 2015 admin-1 codes (map them with the GAUL boundaries, not names); a cell's % of trend is
meaningless where the line runs near zero (13 drops those series).

### Contact-zone test (section "Where herders meet farmers" on the restricted page)

McGuirk & Nunn's (2025) nearest-neighbour design, reproduced from their replication data and then carried to
this page's data and outcome step by step. Design written first: `$W/processed/belt/prespec_2026-11-19_contact.md`.

```
# replication package (CC-BY 4.0), Zenodo record 13308366 -> $W/raw/mcguirk_nunn/replication_v2.1.zip,
#   data + do-file + README unzipped (-j) into $W/raw/mcguirk_nunn/x/
# UCDP Non-State 25.1 (ucdp-nonstate-251-csv.zip) -> $W/raw/ucdp/nonstate/
# nearest neighbour per cell (15 builds $W/raw/mcguirk_nunn/n1_identity.parquet if missing): each cell's
#   n1_g_prec_gpcc series (1989-2018) matched to the one territory with the same g_prec_gpcc series
uv run python 15_contact.py --null     # permutation null only: the detectable effect, before any estimate
uv run python 15_contact.py            # calibration, bridge, primary, variations -> $W/processed/belt/contact_data.json (~5 min)
uv run python blob_sync.py push
```
Gotchas: the package has no South Sudan (its "Sudan" is pre-2011) and no neighbour names; the page shows herding
scores only, never ethnic territory names (they stay in `$W/processed/belt/contact_neighbours.csv`).
