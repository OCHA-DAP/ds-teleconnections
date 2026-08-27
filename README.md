# ds-teleconnections

A global survey of how major climate modes drive seasonal rainfall, country by country.

**Live site:** https://ocha-dap.github.io/ds-teleconnections/

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
1. Download the six climate indices from NOAA PSL (cached under `cache/`).
2. **Country pass** — query ERA5 country-level monthly precipitation from the team DB (`public.era5`, `adm_level=0`).
3. **Pixel pass** — download the ERA5 monthly precipitation COGs from the `raster` blob container, subset to the map viewport, and cache the stack to `cache/era5_pixel/` (~850 MB, one-time; later runs reuse it).
4. Run the correlation analysis for both resolutions × both lag caps (3- and 6-month).
5. Write all PNG maps to `docs/maps/` and the report to `docs/index.html`.
6. Write intermediate correlation tables to `out/` (git-ignored).

The two passes are independent and have different access requirements — the country pass needs the prod
Postgres DB, the pixel pass needs only blob. Either can be skipped to regenerate just the other half
(the report is rebuilt from whatever PNGs are on disk):

```bash
uv run python teleconnection_survey.py --skip-adm0   # pixel maps + report only
uv run python teleconnection_survey.py --skip-px     # country maps + report only
```

Open `docs/index.html` in a browser to view the report locally.

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
For each country × trimester × index, Pearson *r* is computed across lags up to the selected maximum (the index leads rainfall). The lag with the highest |*r*| is the "best lag" for that combination. Significance is p < 0.05 (two-tailed). |*r*| < 0.30 → no reliable signal (gray); |*r*| ≥ 0.45 → strong (darker shade).

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

## The web app

`docs/index.html` is a self-contained static page (no external dependencies):

- **Resolution** toggle — Country (ADM0) / Pixel (0.25°) (default Country).
- **View** toggle — Total association / Unique signal (default Unique).
- **Max lag** toggle — 3 months / 6 months (default 3).
- The three toggles compose freely; section notes and legends switch with the resolution.
- Maps support inline scroll-to-zoom (scroll to zoom on cursor, drag to pan, double-click to reset); the time-series panel and legend stay fixed beside each map.

---

## Outputs

```
docs/
  index.html                     generated report
  maps/
    map_{l3,l6}_{total,partial}_{index}.png      per-index correlation maps (country)
    map_px_{l3,l6}_{total,partial}_{index}.png   per-index correlation maps (pixel)
    map_dominant_{l3,l6}.png                     dominant-mode maps (country)
    map_px_dominant_{l3,l6}.png                  dominant-mode maps (pixel)
    ts_{index}.png                               index historical time-series panels
    enso_elnino.png / enso_lanina.png            ENSO composites (country)
    enso_px_elnino.png / enso_px_lanina.png      ENSO composites (pixel)
    index_corr_matrix.png                        collinearity heatmap
out/                              intermediate correlation tables (git-ignored)
  corr_{total,partial}_{l3,l6}.parquet
  corr_display_{total,partial}_{l3,l6}.parquet
  corr_px_display_{l3,l6}.npz                    per-cell display arrays + grid metadata
cache/era5_pixel/                 cached ERA5 monthly grid stack (~850 MB, git-ignored)
  monthly.npy / meta.json
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
pyproject.toml / uv.lock     dependencies (numpy, pandas, geopandas, scipy, matplotlib, ocha-stratus)
docs/                        generated GitHub Pages site (committed)
.github/workflows/pages.yml  Pages deployment
cache/                       downloaded indices + boundaries (git-ignored)
cache/era5_pixel/            cached ERA5 0.25° monthly grid stack (git-ignored)
out/                         intermediate parquet / npz tables (git-ignored)
```
