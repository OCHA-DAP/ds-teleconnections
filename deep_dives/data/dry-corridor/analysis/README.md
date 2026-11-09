# Dry Corridor 2027 outlook (Primera and Postrera): analysis scripts

Run once on 9 November 2026 to build `/enso/dry-corridor/` (page builder: `cadc_deep_dive.py` at the repo root). They are
working scripts: each takes a work directory (`SP`) as its first argument and reads and writes there. `common.py` and
`units.py` are `exec`'d from `SP/scripts/`, so copy this folder to `SP/scripts/` first and run the scripts from the repo
root with `uv run python -I`. Database and blob reads use ocha-stratus with the connection settings from the environment
(read-only).

Index files: `common.py` reads the NOAA PSL Niño3.4 series (ERSSTv6, published on a 1981–2010 base, psl.noaa.gov/data/
correlation/nina34.anom.data) from `NINO34_FILE` (default `cache/live/nino34.data`) and re-bases it to 1991–2020 month by
month. `season_models.py` and `synth.py` read the PSL tropical North Atlantic index (psl.noaa.gov/data/correlation/
tna.data, HadISST, 1991–2020) from `TNA_FILE` (default `SP/idx/tna.data`).

| Step | Script | Output in `SP` |
|---|---|---|
| Boundaries (COD-AB admin 0 and 1, team blob) | `get_bounds.py SP/bounds` | `bounds/{gtm,hnd,slv}_adm{0,1}.gpkg` |
| ERA5 monthly by country and department (team database) | `get_era5.py SP` | `era5_cadc_all.parquet` |
| CHIRPS monthly (IRI Data Library, never the CHC server) | `dl_chirps_monthly.py SP/raw_m/chirps_cadc_monthly.nc` | the netCDF |
| CHIRPS daily, 1 April–30 November 1981–2026 (IRI Data Library) | `dl_chirps_daily.py SP/chirps_d <y0> <y1>` (three year ranges in parallel) | `chirps_d/chirps_cadc_an_<year>.nc` |
| Area weights (fractional cell coverage) for countries, departments and the framework groups (`units.py`) | `weights.py <grid nc> SP/bounds <out> SP/scripts`, once for the monthly grid and once for a daily file | `weights_monthly.parquet`, `weights_daily.parquet` |
| CHIRPS monthly by unit | `chirps_monthly_units.py <nc> SP/weights_monthly.parquet SP/chirps_monthly_units.parquet` | `chirps_monthly_units.parquet` |
| NMME (IRI Data Library): rainfall, Niño3.4 and tropical Atlantic forecasts from November starts, skill and calibration | `nmme/00_meta.py` … `04_enso_analogs.py` in order; they take `SP/nmme` (holding `raw/` and `out/`), the bounds, the CHIRPS monthly file, the ERA5 parquet and the index files (see each script's usage line) | `nmme/out/` → `results/nmme/` |
| ENSO inputs for the regressions | written by hand from `nmme/out/nino34_2027_calibrated.csv` and `tna_2027_calibrated.csv` (the "regression" calibration; see the note in the file) | `enso_inputs.json` |
| Seasonal totals, regressions, past seasons after strong El Niño winters | `season_models.py SP SP/enso_inputs.json` | `season_results.csv`, `season_totals.csv` |
| Sub-season composites by ENSO situation | `composites.py SP` | `composites.csv`, `subseason_after.csv` |
| Onset, canícula dry spell, end of rains, Sep–Oct dry spell per CHIRPS cell | `daily_metrics.py SP` | `daily_metrics_pixels.parquet` |
| Timing by unit and year, composites | `daily_agg.py SP` | `daily_metrics_units.parquet` (→ `results/timing_by_unit_year.csv`), `timing_composites.csv` |
| Onset sensitivity (simpler rule) and by ENSO situation | `onset_check.py SP` | `onset_check.csv` |
| EM-DAT drought records by ENSO situation (team blob snapshot) | `emdat.py SP` | `emdat_droughts.csv` (not committed: EM-DAT terms) |
| Map outlines | `geo.py SP` | `geo.json` |
| Page data | `synth.py SP` | `page_data.json` → `../page_data.json` |
| Regression figures quoted in the text | `results_summary.py SP` | `regression_summary.json` → `results/` |

Printed, not saved: the CHIRPS/ERA5 agreement and the sub-season correlations with Niño3.4 (run inline on the season
totals), the correlation of onset with March–May Niño3.4 (inline on `daily_metrics_units.parquet`), and the check that the
Pacific–Atlantic gradient cross-validates at least as well as Niño3.4 and TNA as two separate predictors.

Notes: CHIRPS daily for 2026 was not available when the page was built, so the timing analysis covers 1981–2025 and the
2026 figures on the page come from the monthly files. The canícula dry-spell metric in `daily_metrics.py` is computed but
not used on the page: CHIRPS records rain on too many July–August days for spell lengths to be meaningful.
