# ds-teleconnections

A global survey of how major climate modes drive seasonal rainfall, country by country.

**Live site:** https://ocha-dap.github.io/ds-teleconnections/

For each country the project measures the correlation between seasonal rainfall (ERA5) and six climate-mode indices — **ENSO (Niño3.4), IOD, TNA, TSA, AMM, PDO** — across every rolling 3-month season and a range of lead times, then renders the results as an interactive set of choropleth maps. It is a manager-facing product for communicating where, when, and how strongly ENSO and other modes shape humanitarian rainfall risk.

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
2. Query ERA5 country-level monthly precipitation from the team DB (`public.era5`, `adm_level=0`).
3. Run the correlation analysis for both lag caps (3- and 6-month).
4. Write all PNG maps to `docs/maps/` and the report to `docs/index.html`.
5. Write intermediate correlation tables to `out/` (git-ignored).

Open `docs/index.html` in a browser to view the report locally.

---

## Data sources

| Source | What | Where |
|---|---|---|
| **ERA5** | Country-level monthly mean precipitation (mm/day), `adm_level=0` | Team DB `public.era5` via `ocha-stratus` |
| **Niño3.4** | ENSO SST anomaly (5°N–5°S, 120–170°W) | NOAA PSL `nina34.anom.data` |
| **IOD (DMI)** | Indian Ocean Dipole Mode Index (HadISST) | NOAA PSL `dmi.had.long.data` |
| **TNA** | Tropical North Atlantic SST anomaly | NOAA PSL `tna.data` |
| **TSA** | Tropical South Atlantic SST anomaly | NOAA PSL `tsa.data` |
| **AMM** | Atlantic Meridional Mode | NOAA PSL `amm.data` |
| **PDO** | Pacific Decadal Oscillation | NOAA PSL `pdo.data` |
| **Boundaries** | Natural Earth 110m admin-0 (cached locally) | downloaded on first run |

ERA5 values are **mm/day** and used as-is — a trimester value is the mean of its three monthly means (not a sum). Somaliland is merged into Somalia; Western Sahara (ESH) is shown as its own entity; small island states are drawn as dots.

Analysis period: **1981–2025**.

---

## Methodology

The report itself contains a full **Methodology** section at the bottom; this is a summary.

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
- **Dominant climate mode:** each country coloured by the index with the strongest significant total correlation (|*r*| ≥ 0.30); split diagonal adds a second, different mode when it acts on a non-overlapping trimester. Respects the lag toggle.
- **ENSO composites (El Niño / La Niña):** mean rainfall anomaly (in SDs) in each country's headline trimester, classified by the **concurrent** trimester's Niño3.4 (≥ ±0.5). The two maps are independent — ENSO impacts are asymmetric.
- **Index collinearity:** 6×6 Pearson matrix among the indices (e.g. AMM–TNA ≈ 0.81), which explains where Total and Unique diverge.

---

## The web app

`docs/index.html` is a self-contained static page (no external dependencies):

- **View** toggle — Total association / Unique signal (default Unique).
- **Max lag** toggle — 3 months / 6 months (default 3).
- Maps support inline scroll-to-zoom (scroll to zoom on cursor, drag to pan, double-click to reset); the time-series panel and legend stay fixed beside each map.

---

## Outputs

```
docs/
  index.html                     generated report
  maps/
    map_{l3,l6}_{total,partial}_{index}.png   per-index correlation maps
    map_dominant_{l3,l6}.png                  dominant-mode maps
    ts_{index}.png                            index historical time-series panels
    enso_elnino.png / enso_lanina.png         ENSO composites
    index_corr_matrix.png                     collinearity heatmap
out/                              intermediate correlation tables (git-ignored)
  corr_{total,partial}_{l3,l6}.parquet
  corr_display_{total,partial}_{l3,l6}.parquet
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
- No multiple-testing correction (no FDR/Bonferroni) — with hundreds of country–trimester combinations per index, some false positives are expected at p < 0.05.
- Partial correlations do not cleanly partition variance when predictors are collinear; the Unique-signal view indicates whether a mode contributes *beyond* the others, not a clean attribution.
- Coverage is limited to the 153 countries with ERA5 admin-0 extracts.

---

## Repository layout

```
teleconnection_survey.py     all analysis + report generation (single file)
pyproject.toml / uv.lock     dependencies (numpy, pandas, geopandas, scipy, matplotlib, ocha-stratus)
docs/                        generated GitHub Pages site (committed)
.github/workflows/pages.yml  Pages deployment
cache/                       downloaded indices + boundaries (git-ignored)
out/                         intermediate parquet tables (git-ignored)
```
