# ds-teleconnections

A global survey of how major climate modes drive seasonal rainfall, country by country.

**Live site:** https://ocha-dap.github.io/ds-teleconnections/ — a landing page with two products:

| Path | What |
|---|---|
| [`/survey/`](https://ocha-dap.github.io/ds-teleconnections/survey/) | the global teleconnection survey (this script's report) |
| [`/enso-map/`](https://ocha-dap.github.io/ds-teleconnections/enso-map/) | ENSO and seasonal rainfall, season by season — interactive world map of the pixel ENSO correlation for each of the 12 seasons (see [ENSO season map](#enso-season-map)) |
| [`/enso/`](https://ocha-dap.github.io/ds-teleconnections/enso/) | ENSO country deep dives — per-country evidence reviews (see [Country deep dives](#country-deep-dives)) |

At two spatial resolutions — per country and per 0.25° grid cell — the project measures the correlation between seasonal rainfall (ERA5) and six climate-mode indices — **ENSO (Niño3.4), IOD, TNA, TSA, AMM, PDO** — across every rolling 3-month season and a range of lead times, then renders the results as an interactive set of choropleth maps. It is a manager-facing product for communicating where, when, and how strongly ENSO and other modes shape humanitarian rainfall risk.

The entire analysis and report generation lives in a single script: [`teleconnection_survey.py`](teleconnection_survey.py).

---

## Quick start

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). Data access uses [`ocha-stratus`](https://github.com/OCHA-DAP/ocha-stratus) and needs the team's Azure/DB credentials in the environment.

```bash
uv sync
uv run python teleconnection_survey.py
```

On run the script will:
1. Download the six climate indices from NOAA PSL (cached under `cache/`), and draw the time-series panel for each from NOAA's latest files (see below).
2. **Country pass** — query ERA5 country-level monthly precipitation from the team DB (`public.era5`, `adm_level=0`).
3. **Pixel pass** — download the ERA5 monthly precipitation COGs from the `raster` blob container, subset to the map viewport, and cache the stack to `cache/era5_pixel/` (~850 MB, one-time; later runs reuse it).
4. Run the correlation analysis for both resolutions × both lag caps (3- and 6-month).
5. Write all PNG maps to `docs/survey/maps/` and the report to `docs/survey/index.html`.
6. Write intermediate correlation tables to `out/` (git-ignored).

The two passes are independent and have different access requirements — the country pass needs the prod
Postgres DB, the pixel pass needs only blob. Either can be skipped to regenerate just the other half
(the report is rebuilt from whatever PNGs are on disk):

```bash
uv run python teleconnection_survey.py --skip-adm0   # pixel maps + report only
uv run python teleconnection_survey.py --skip-px     # country maps + report only
```

The time-series panels beside each map show where each index stands now, so they are drawn from NOAA's
latest files, kept separately in `cache/live/` (re-downloaded when more than a week old), never from the
pinned `cache/{index}.data` copies that the correlations and deep dives use. NOAA revises these series
(Niño3.4 moved to ERSST v6 in 2026), so the two must not be mixed. Two panels use a different series
from the maps (`LIVE_INDEX_SOURCES`): PDO comes from NOAA NCEI's ERSST v5 PDO, because PSL's PDO (the
original University of Washington series) ended in Aug 2025 (NCEI's moves with it, r = 0.98 since 2010, but
reads about 0.5 lower since 1981 and 0.6–0.7 lower since 2010); the AMM has no current source at all (its
only publisher, Univ. of Wisconsin, ends Feb 2026 because the NCEP-NCAR reanalysis it is computed from
stopped updating), so its panel stays at Feb 2026 and says so. `cache/` is not committed: the pinned
copies exist only on the machine that ran the analysis, and on a fresh clone the first run downloads NOAA's
current files into that slot. To bring just the panels up to date — NOAA only, no DB or blob access (stops
with an error if NOAA can't be reached):

```bash
uv run python teleconnection_survey.py --ts-only     # re-download the indices, redraw docs/survey/maps/ts_*.png
```

Open `docs/survey/index.html` in a browser to view the report locally (`docs/index.html` is the site landing page).

---

## Data sources

| Source | What | Where |
|---|---|---|
| **ERA5** (country) | Country-level monthly mean precipitation (mm/day), `adm_level=0` | Team DB `public.era5` via `ocha-stratus` |
| **ERA5** (pixel) | Monthly total precipitation COGs, 0.25° global | Blob `raster/era5/monthly/processed/precip_reanalysis_v*.tif` |
| **Niño3.4** | ENSO SST anomaly (5°N–5°S, 120–170°W) | NOAA PSL `nina34.anom.data` |
| **IOD (DMI)** | Indian Ocean Dipole Mode Index (HadISST) | NOAA PSL `dmi.had.long.data` |
| **TNA** | Tropical North Atlantic SST anomaly | NOAA PSL `tna.data` |
| **TSA** | Tropical South Atlantic SST anomaly | NOAA PSL `tsa.data` |
| **AMM** | Atlantic Meridional Mode | NOAA PSL `amm.data` |
| **PDO** | Pacific Decadal Oscillation | NOAA PSL `pdo.data` |
| **Boundaries** | Natural Earth 110m admin-0 (cached locally) | downloaded on first run |

ERA5 values are **mm/day** and used as-is — a trimester value is the mean of its three monthly means (not a sum). Somaliland is merged into Somalia; Western Sahara (ESH) is shown as its own entity; small island states are drawn as dots.

Analysis period: **1981–2025**. The pixel pass covers every ERA5 land cell inside the map viewport (−100°–180° E, −36°–56° N) — 163,453 cells of the 369 × 1120 subset; ocean cells are not analysed.

---

## Methodology

The report itself contains a full **Methodology** section at the bottom; this is a summary.

### Spatial resolution (Country vs Pixel) — in-app toggle
The same method runs at two units of analysis:

- **Country (ADM0):** ERA5 admin-0 area-weighted means from the team DB — one series per country, 153 countries.
- **Pixel (0.25°):** the ERA5 monthly COGs read directly from blob, with the whole pipeline run independently on each land grid cell. No spatial smoothing or pooling between cells.

The pixel view exposes sub-national structure a national mean averages away — countries spanning more than one rainfall regime (Kenya, Ethiopia, Indonesia, Brazil) cancel opposing signals, and a signal confined to one basin disappears. It costs two things:

- **No field-significance correction.** Each cell is one ~45-year series, so a few percent of cells pass p < 0.05 by chance. Read spatially coherent regions as signal and isolated cells as noise.
- **An extra aridity filter** the country pass does not need: a cell–season is analysed only if its climatological mean exceeds **0.25 mm/day**, dropping hyper-arid interiors (Sahara, Rub' al Khali, Taklamakan) where correlations against near-zero rainfall are numerically large but meaningless. These are drawn in the "arid — no wet season" shade.

Two rendering differences follow from the resolution: both-signs cells are **hatched** rather than split diagonally, and the pixel dominant-mode map shows only the top mode, only where it leads the runner-up by ≥ 0.10 in |*r*| (a single cell does not average enough area for the arg-max over six collinear modes to be stable).

Implementation note: the pixel pass is fully vectorised over cells — one Pearson call per (index, lag) across the whole grid, and the partial pass uses batched precision matrices rather than per-cell regressions. Both are validated against the country-pass implementations (`stats.pearsonr` and `_partial_corr`) to ~1e-8.

### Seasons
All **12 rolling 3-month windows** are assessed per country (NDJ, DJF, JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND). Year labels use the first month of the window (NDJ/DJF carry the previous-December year, e.g. DJF 2024 = Dec 2024 – Feb 2025).

### Rainy-season filter
A country–trimester pair is analysed only if that trimester's climatological mean rainfall is ≥ 25% of the country's annual mean. The annual mean is computed from the four **non-overlapping canonical** trimesters (DJF + MAM + JAS + OND), which cover each calendar month exactly once (summing all 12 rolling windows would triple-count).

### Correlation sweep
For each country × trimester × index, Pearson *r* is computed across lags up to the selected maximum (the index leads rainfall). The lag with the highest |*r*| is the "best lag" for that combination. Significance is p < 0.05 (two-tailed). |*r*| < 0.30 → no reliable signal (gray); |*r*| ≥ 0.5 → strong (darker shade; Cohen's 'large', aligned with the seas5-skill app's high-skill threshold).

### Lag cap (3 vs 6 months) — in-app toggle
- **3 months (default):** the index may lead rainfall by at most one preceding non-overlapping season. Keeps the relationship within the same evolving event; the more forecast-relevant view.
- **6 months:** additionally admits prior-season relationships (e.g. a previous winter's ENSO state predicting the following monsoon), which can have the *opposite* sign to the concurrent signal.

> **Note on monotonicity.** The **Total** view is nested (6-month ⊇ 3-month), so a country's signal there never disappears when widening the cap. The **Unique signal (partial)** view is *not* nested: partial correlation removes the other modes at *their own* best lags, and those control lags change with the cap, so a country's unique signal can legitimately appear, vanish, or flip sign between the two views. This is expected, not a bug.

### Total association vs Unique signal — in-app toggle
- **Total:** raw pairwise Pearson *r* (includes variance shared with other modes). Use this when a single index alone is your predictor.
- **Unique signal:** partial correlation holding the other modes constant (residuals method). PDO gets its own unique signal but is excluded from the control set (never regressed out of the other modes), because as a low-frequency mode collinear with ENSO it would otherwise absorb genuine ENSO signal. Shrinkage from Total to Unique reflects shared variance, not absent signal. Suppressor cases (significant in partial but not total) are shown.

### Other panels
- **Dominant climate mode:** each country (or cell) coloured by the index with the strongest significant total correlation (|*r*| ≥ 0.30). At country resolution a split diagonal adds a second, different mode when it acts on a non-overlapping trimester; at pixel resolution only the top mode is shown, and only where it clears the runner-up by ≥ 0.10. Respects the lag toggle.
- **ENSO composites (El Niño / La Niña):** mean rainfall anomaly (in SDs) in each country's (or cell's) headline trimester, classified by the **concurrent** trimester's Niño3.4 (≥ ±0.5). Rendered at both resolutions. The two maps are independent — ENSO impacts are asymmetric.
- **Index collinearity:** 6×6 Pearson matrix among the indices (e.g. AMM–TNA ≈ 0.81), which explains where Total and Unique diverge.

---

## ENSO season map

`enso_season_map.py` writes the data behind [`/enso-map/`](https://ocha-dap.github.io/ds-teleconnections/enso-map/),
an interactive map (`docs/enso-map/index.html`, WebGL 2, no libraries) of the Niño3.4 correlation with ERA5 rainfall for
each of the 12 rolling seasons, per 0.25° land cell. The survey's pixel ENSO map shows each cell's
strongest season; this map shows every season.

The method is the survey's pixel pass, called unchanged (`pixel_trimester_rainfall_era5`, `pixel_sweep`):
total Pearson *r* at the best lag of 0–3 months, 1981–2025, the same rainy-season rule and p < 0.05. Only
two inputs differ: the area is all land from 60°S to 84°N (180°W–180°E) instead of the survey viewport,
and the land mask is Natural Earth 1:10m land plus minor islands, so small islands get cells. Each run
checks itself against the survey and stops without writing if they disagree: on the cells both grids
share, the strongest significant rainy season must equal `out/corr_px_display_l3.npz` (last run: 162,754
shared cells, 96,723 with a value on both, largest difference 0.0; 699 survey cells lie off this grid's
finer coastline).

```bash
uv run python enso_season_map.py      # -> docs/enso-map/data/ (~6 MB)
```

The first run downloads the global ERA5 monthly COGs from the prod raster blob into
`cache/era5_pixel_global/` (~2.4 GB download, ~1.8 GB on disk); later runs are offline and take under a
minute. Niño3.4 is the survey's pinned `cache/nino34.data`. Rerun it whenever the survey's pixel pass is
rerun, so the two stay in step. `--cache-dir` / `--survey-npz` point at another checkout's cache and
`out/` (useful from a worktree).

**Projection (adaptive, equal-area).** With the whole world in view the map is Equal Earth (spherical
form), the equal-area projection the UN General Assembly encouraged in September 2026, with its central
meridian on the view centre: dragging east-west turns the globe, across 180° too. Zooming in blends it
into a Lambert azimuthal equal-area projection centred on the view (true to shape at the centre, north
up), by the share of the world's width in view (Equal Earth above 55%, azimuthal below 15%; after Jenny
2012, adaptive composite map projections). Both ends are equal-area; in the blend, areas are close to true
in the middle of the view but can be off by up to about a third towards the corners at mid zoom, most at
high latitudes (Fable review: area scale 0.61 at the top corners at z 3.7 centred 38.5°N). On tall frames
the blend waits until the poles have left the frame. Zoom out stops at the whole world. Vertically the centre is held so that Equal
Earth's pole line stays at or beyond the frame (otherwise its back seam opens into a gap over the pole at
mid zooms); high latitudes can be centred once zoomed in.

Rendering is the page's own (no map library, no Web Mercator tiles): `projCore()` is the one projection
implementation and the WebGL 2 vertex shader is a line-for-line port. Cells, the ocean (2° quads) and
land beyond the data grid (`polar.bin.gz`, Antarctica) are instanced quads whose corners are offsets from
their own centre, clamped at the seam; borders (Natural Earth 1:50m from `basemap.json`, per-vertex
trigonometry precomputed), the world's edge, the cell highlight and names (Natural Earth label points and
zooms, placed greedily without overlaps) are drawn on a 2D canvas above. Hover, zoom anchoring and fitting
a country invert the projection numerically (exact at either end, Newton between, retried from the other
seed; round trips agree to 1e-8 px). Dragging keeps the grabbed place under the pointer with damped,
backtracking steps, and re-grabs whatever is under the pointer where the map cannot follow (latitude
limit, world's edge), so nothing jumps. Cell edges are computed in exact degrees in the shader, so
neighbours share bit-identical edges; edges along the 180° cut and the South Pole (Natural Earth's
polygon cuts) are not stroked as borders. `?debug` exposes the projection as `window.ENSO_MAP` for such checks. Needs WebGL 2.

Data files (the binaries are gzip, decompressed in the page with `DecompressionStream`; Pages compresses
the JSON in transit): `meta.json` (grid, seasons,
thresholds, cross-check result, a content hash used to bust caches), `land.bin.gz` (land mask, one bit per
grid cell, row-major from 84°N/180°W), `annual.bin.gz` (uint16 annual sum of the four canonical seasons'
mean rainfall, 0.01 mm/day, rounded up), `countries.json` (Natural Earth 1:50m names and bounding boxes for the
zoom box), `basemap.json` (see above), `polar.bin.gz` (land mask beyond the data grid: 23 rows from
89.75°N, then 119 from 60.25°S, 1,440 columns, one bit per 0.25° cell) and one `{season}.bin.gz` per season holding, per land cell, the season's mean rainfall
(uint16, 0.01 mm/day, rounded down so the 0.25 mm/day test reads the same as in Python), *r* × 100
(int8, truncated towards zero so the printed value never crosses a class edge; −128 = none) and a flags byte (bits 0–1 best lag, 2 rainy season,
3 p < 0.05, 4 *r* computed, 5–7 the survey's colour class wherever p < 0.05). The "Drier months: Show"
switch colours seasons that fail the 25% rule but average at least 0.25 mm/day, hatched; by default the
map shows exactly what the survey's rule allows.

---

## Country deep dives

`enso_deep_dive.py` builds the per-country ENSO evidence reviews published under
[`/enso/`](https://ocha-dap.github.io/ds-teleconnections/enso/). Each country is one TOML file in
`deep_dives/` holding the curated narrative (why the survey catalogue graded the country as it did,
what the literature supports, references); the script adds the reproducible ERA5 evidence from the
same cached pixel stack and NOAA indices the survey uses — seasonal cycle, pixel-level Niño3.4
correlation maps, an ENSO-phase rainfall history, El Niño composite and drought hit-rate maps, and
the phase/drought tables — and writes `docs/enso/<slug>/index.html` plus `docs/enso/index.html`.

```bash
uv run python enso_deep_dive.py              # rebuild every country + the index
uv run python enso_deep_dive.py --only eri   # one country
```

Needs `cache/era5_pixel/` (built by a survey run; blob access only — the deep-dive script then appends any newer
ERA5 months from the prod raster blob into `cache/era5_pixel/monthly_ext.npy`, so the current season is analysed
without touching the survey's cache) and, for the country-level
cross-check table, the survey's `out/corr_*_l3.parquet`. No DB access. To add a country, copy
`deep_dives/eri.toml` or `deep_dives/mwi.toml`, edit, run, and commit `docs/enso/`.

TOML knobs worth knowing: `headline_season` drives the drought analysis; `map_seasons` lists the
correlation-map panels in the order given (chronological reads best when the sign flips within a
season, as in Malawi); `[[zones]]` split the country by headline-season climatology (`min_mm_day`,
`max_mm_day`) and/or latitude (`min_lat`, `max_lat`; a cell centred on a bound goes south), and
`zone_history = true` adds a per-zone phase-history figure; `zones_analysis = false` keeps zones for the
seasonal cycle and locator map only and runs the rest nationally (Zimbabwe). Each page also gets a SEAS5 skill-by-zone
figure and table: for one issuance month (`skill_issued_month`, default one month before the headline
season starts), the median pixel Pearson r per zone for every trimester that issuance covers, binned
low / moderate / high with the seas5-skill app's thresholds 0.30 / 0.50 and drawn under the rainy-season
climatology, plus the return period of that issuance's current forecast anomaly per zone (dry above / wet
below the axis, with the app's 3- and 10-year alert bands) and an auto-written alert summary. `food_security = "fews"` adds a FEWS NET section (current situation and
the latest round's projections on FEWS NET's own units, from the ds-fewsnet-mirror public site JSON and the
dev-blob unit geometry, cached under `cache/fews_*`), drawn on FEWS NET's own polygons. Every page also gets
a by-province section built from the team's per-admin ERA5 raster stats (`public.era5`, prod, admin-1 monthly
means; CODAB polygons via `ocha_stratus.codab`) — headline-season r and El Niño hit-rate; `adm1 = false` drops it. FEWS NET is never aggregated: it is drawn at
its reported units, and those unit outlines are overlaid on the drought rasters. Needs `DSCI_AZ_DB_*` creds and
`PGSSLMODE=require`; results are cached as parquet under `cache/`. Read from the app's detrended skill cube, fetched once from the DEV blob into
`cache/skill_stats_grid_detrended.nc` (1.2 GB); `skill = false` drops the section. `season_table = true` adds a season-by-season table (rainfall rank, ENSO phase, CERF drought
allocations from the `aa.cerf_allocation` mirror + drought-period supplement, FEWS NET's public population-in-Phase-3+
series from FDW and the share of FEWS NET units in Phase 3+/4+ from the mirror), plus a row for the season being forecast. A page can be reordered or trimmed with `section_order` (block names:
verdict, summary, before, era5, drought, adm1, skill, fews, after, refs) and re-headed with a `[titles]` table;
`show_adm0`, `skill_intro`, `fews_intro`, `adm1_intro` switch the boilerplate off (see Zimbabwe, which is
written around three questions). The Niño3.4 *analysis* series is pinned in `cache/nino34.data` (NOAA CPC ERSST v5,
identical to the PSL series the survey was built on); the live PSL series (ERSST v6 since 2026, ~+0.2 °C) is
fetched weekly into `cache/nino34_latest.data` and used only for the current-ENSO-state line. Top-level keys must sit above the first
`[table]` header or TOML swallows them.

**Gaza and the West Bank** share one page, `/enso/gaza-west-bank/` (`deep_dives/gaza-west-bank.toml`, `builder = "levant-combined"`),
built by their own module, `levant_deep_dive.py`, which `enso_deep_dive.py` dispatches to (or run it directly:
`uv run python levant_deep_dive.py gaza-west-bank`). Each area is analysed from its own TOML in `deep_dives/parts/` (`gaza.toml`,
`west-bank.toml`; the dispatcher does not build files there as pages), with figures in `docs/enso/gaza-west-bank/<area>/`, and the
page interleaves the two section by section (`render_blocks` → `render_combined`). The combined TOML holds what they share: the
summary, the forecast text, the regional-maps text (maps drawn once, both areas outlined), the caveats and the index-card verdict.
The former `/enso/gaza/` and `/enso/west-bank/` URLs redirect (`redirect_from`). What differs between the two areas (COD-AB polygon,
ERA5 cells, GPCC cells, rain gauge, IMERG and SEAS5 boxes, labels) sits in each part's `[area]` table; Gaza's values are the defaults. Gaza is three ERA5 cells and
its winter hazard is rain, wind, sea and cold rather than drought, so the page reuses the series' chrome, SEAS5 skill figure and
pixel correlation maps but adds: four independent rainfall records (ERA5 hourly point series for the three cells from the CDS
`reanalysis-era5-single-levels-timeseries` dataset, 1950–; the GPCC 1° combined product via NOAA PSL NCSS, 1891–; IMERG late v7
daily from the prod raster blob, 1998–; GHCN-Daily Beer Sheva, 1921–2016), a 31-year running-correlation stationarity test
(HadISST Niño3.4 before 1950), daily metrics by ENSO phase, and the last three winters' daily rainfall against dated impacts
curated in the TOML (`[[impacts]]`, `[[prewar]]`, public UN/cluster sources only). Downloads cache under `cache/gaza/`; CDS
needs a key in `~/.cdsapirc` (the CDS endpoint is forced, whatever URL the file names).

The West Bank page adds a farming section from its `[agri]` table (`levant_agri.py`): three agro-ecological zones defined by
governorate (`[[agri.zones]]`), their ERA5 rain, spring MODIS NDVI per zone (MOD13Q1 v061 COGs on the Microsoft Planetary
Computer, anonymous SAS token, cached to `cache/westbank/modis_ndvi_zones.parquet`), NOAA STAR province-averaged vegetation
health (1982–), FAOSTAT yields for Palestine (subset committed as `deep_dives/data/pse_faostat_crops.csv`; `fetch_faostat`
rebuilds it from the bulk file) and PCBS olive presses and 2021 census cereal area by governorate
(`deep_dives/data/pcbs_*.csv`, with a source URL per row). Crop year Y follows the October Y−1 to April Y rainy season.

SEAS5 on this page: skill and return periods come from the seas5-skill app's pixel cube, which is recomputed the day after a
new forecast lands. When the cube's forecast is older than the latest issuance in the team database, `seas5_db_trimesters` takes
the return periods from the admin-1 raster statistics instead (prod `public.seas5` / `public.era5`, the app's detrended method,
fully forecast windows only; cached in `cache/gaza/*_adm1_db.parquet`), keeps the cube's hindcast skill, and the page says so.
The database is reachable only inside the VNet or through the laptop tunnel (`DSCI_AZ_DB_PROD_HOST`). Delete
`cache/skill_stats_grid_detrended.nc` to pick up the refreshed cube; the page then reads it again (the build says so), and the
SEAS5 paragraphs, summary item 1 and key message 1 need re-checking against the regenerated tables, since pixel medians can
differ slightly from the area-mean stand-in.
`skill_issued_month` in each part's TOML selects the issuance, and the SEAS5 paragraphs there are written for it.

---

## The web app

`docs/survey/index.html` is a self-contained static page (no external dependencies):

- **Resolution** toggle — Country (ADM0) / Pixel (0.25°) (default Country).
- **View** toggle — Total association / Unique signal (default Unique).
- **Max lag** toggle — 3 months / 6 months (default 3).
- The three toggles compose freely; section notes and legends switch with the resolution.
- Maps support inline scroll-to-zoom (scroll to zoom on cursor, drag to pan, double-click to reset); the time-series panel and legend stay fixed beside each map.

---

## Outputs

```
docs/
  index.html                     site landing page (hand-edited cards; forwards old #section links to survey/)
  assets/                        landing-page CSS + hero (HDX v2 tokens, from the team's pages template)
  enso/                          ENSO country deep dives (generated by enso_deep_dive.py from deep_dives/*.toml)
  survey/index.html              generated survey report
  survey/maps/
    map_{l3,l6}_{total,partial}_{index}.png      per-index correlation maps (country)
    map_px_{l3,l6}_{total,partial}_{index}.png   per-index correlation maps (pixel)
    map_dominant_{l3,l6}.png                     dominant-mode maps (country)
    map_px_dominant_{l3,l6}.png                  dominant-mode maps (pixel)
    ts_{index}.png                               index time-series panels (latest NOAA values; --ts-only)
    enso_elnino.png / enso_lanina.png            ENSO composites (country)
    enso_px_elnino.png / enso_px_lanina.png      ENSO composites (pixel)
    index_corr_matrix.png                        collinearity heatmap
out/                              intermediate correlation tables (git-ignored)
  corr_{total,partial}_{l3,l6}.parquet
  corr_display_{total,partial}_{l3,l6}.parquet
  corr_px_display_{l3,l6}.npz                    per-cell display arrays + grid metadata
cache/era5_pixel/                 cached ERA5 monthly grid stack (~850 MB, git-ignored)
  monthly.npy / meta.json
cache/era5_pixel_global/          the same for the ENSO season map, 60°S–84°N (~1.8 GB, git-ignored)
docs/enso-map/                    ENSO season map (index.html, WebGL 2; data/ from enso_season_map.py)
```

---

## Deployment

GitHub Pages serves the contents of `docs/` on the `feature/era5-ghpages` branch via [`.github/workflows/pages.yml`](.github/workflows/pages.yml). The PNGs and `index.html` are committed; the Action just publishes them. To update the live site:

```bash
uv run python teleconnection_survey.py
git add docs/
git commit -m "Refresh report"
git push
```

---

## Caveats

- Pearson *r* assumes a linear, stationary relationship; teleconnections can be non-linear and non-stationary, and 1981–2025 correlations may not represent future conditions.
- ERA5 is a model-based reanalysis; in data-sparse regions the precipitation field leans more on the model background than on observations.
- No multiple-testing correction (no FDR/Bonferroni) — with hundreds of country–trimester combinations per index, some false positives are expected at p < 0.05. This matters far more at pixel resolution, where there are ~163k cells: only spatially coherent regions should be read as signal.
- Partial correlations do not cleanly partition variance when predictors are collinear; the Unique-signal view indicates whether a mode contributes *beyond* the others, not a clean attribution.
- Country resolution is limited to the 153 countries with ERA5 admin-0 extracts. Pixel resolution covers all land in the map viewport, so it includes territory outside those 153 countries — and conversely drops small island states that do not occupy a 0.25° land cell (these are drawn as dots on the country maps).
- Country and pixel results are not expected to match cell-for-country: a national mean and a grid cell are different random variables, and where a country spans opposing regimes the country map can show no signal while the pixel map shows strong signal of both signs.

---

## Repository layout

```
teleconnection_survey.py     all analysis + report generation (single file)
enso_season_map.py           data for the ENSO season map (docs/enso-map/), reusing the survey's pixel pass
pyproject.toml / uv.lock     dependencies (numpy, pandas, geopandas, scipy, matplotlib, ocha-stratus)
docs/                        generated GitHub Pages site (committed)
.github/workflows/pages.yml  Pages deployment
cache/                       downloaded indices + boundaries (git-ignored)
cache/era5_pixel/            cached ERA5 0.25° monthly grid stack (git-ignored)
out/                         intermediate parquet / npz tables (git-ignored)
```
