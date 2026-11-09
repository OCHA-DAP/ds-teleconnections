# Haiti spring 2027 outlook: analysis scripts

Run once on 6 November 2026 to build `/enso/hti/` (page builder: `hti_deep_dive.py` at the repo root). They were written
as working scripts: each takes a work directory (`SP`) as its first argument, reads its inputs from there and writes its
outputs there. The shared helpers `common.py` and `onset_lib.py` are `exec`'d from `SP` (`exec(open(f"{SP}/common.py"))`),
so copy both into the work directory first, and run the scripts from the repo root. Database reads use ocha-stratus with
the connection settings from the environment.

Index files: `common.py` reads the NOAA PSL Niño3.4 series (ERSSTv6, 1981–2010 base, psl.noaa.gov/data/timeseries/month/DS/NINO34/)
from `cache/live/nino34.data` unless `NINO34_FILE` says otherwise; `enso_dept.py` also reads PSL's TNA index
(`cache/live/tna.data`, or `TNA_FILE`). `predictors.py` reads `SP/idx/nao.txt` (CPC monthly NAO) and `SP/idx/tna_latest.data`
(PSL TNA).

| Step | Script | Output in `SP` |
|---|---|---|
| Boundaries | `get_bounds.py SP` | `bounds/hti_adm0.gpkg`, `bounds/hti_adm1.gpkg` |
| ERA5 + SEAS5 by department (team database) | `get_era5_seas5_adm1.py SP` | `era5_hti_all.parquet`, `seas5_hti_all.parquet` |
| IMERG daily (team database) | `get_imerg.py SP`, `get_imerg_adm2.py SP` | `imerg_hti.parquet`, `imerg_hti_adm2_fj.parquet` |
| CHIRPS monthly (IRI Data Library, never the CHC server) | `dl_chirps.py`, then `chirps_monthly_units.py`, area weights `weights.py` | `chirps_monthly_units.parquet` |
| CHIRPS daily Feb–Jul 1981–2026 (IRI Data Library) | `bash dl_all.sh SP` → `dl_chirps_fj.py` | `chirps_fj/chirps_hti_fj_<year>.nc` |
| Daily-grid area weights | `weights_daily.py` | `weights_chirps_daily.parquet` |
| ENSO and department statistics | `enso_dept.py`, `predictors.py`, `xcheck.py`, `monthly_pct.py` | `enso_dept_stats.csv` + printed checks |
| Regressions and past-spring comparisons (winter Niño3.4 ~ N(mu, sd)) | `loo_models.py SP 2.7 0.45` (also 2.3, 3.0) | `loo_results.csv` → `results/loo_results_<mu>.csv` |
| SEAS5 November run by department (run from a ds-seas5-skill checkout) | `seas5_dept.py SP` | `results/seas5_dept_out.txt` |
| Onset, dry spells, false starts per CHIRPS cell | `onset_chirps.py SP` (rules in `onset_lib.py`) | `onset_chirps_pixels.parquet` |
| Onset by department + strong-El-Niño composites | `onset_agg.py SP` | `onset_chirps_adm1.parquet`, `results/onset_chirps_composites.csv` |
| IMERG onset check | `onset_imerg_adm2.py SP` | `onset_imerg_adm1.parquet` |
| Department synthesis table | `dept_table.py SP` | `results/dept_synthesis.csv` |
| Page data (department figures + simplified outlines) | `map_data.py SP` | `map_data.json` → `../page_data.json` |
| NMME (IRI Data Library) | `nmme/01_download.py` … `04_enso_analogs.py`, run in order; they take one directory holding `scripts/` (these files), `raw/` and `out/`, and read the CHIRPS monthly subset from the work directory | `out/` → `results/nmme/` |

`national_springs.json` holds the national CHIRPS and ERA5 March–June totals (% of the 1991–2020 average) and the
December–February Niño3.4 by year, as drawn in the page's scatter plot.

Printed, not saved: the JFM NAO correlation with May–June rain (−0.37, `predictors.py`), the ERA5/CHIRPS ratio and correlation
(`xcheck.py`), the monthly 2026 percentages (`monthly_pct.py`) and the false-start check after allowing for the seasonal
total (run inline on `onset_chirps_adm1.parquet`: regress national false starts on the total, then read the residuals of
the six strong-El-Niño springs). `map_data.py` sets the NMME national figures (125%, 91–172%) by hand from
`results/nmme/03_stdout.txt` and the agreement classes by hand; the page builder checks the classes against their rule.

Notes: IMERG 2026 is dropped in `common.py` (its values break from the earlier record). `hti_aso_era5.py` and
`hti_mam_seas5.py` are the earlier checks of the 2026 August–October season and the SEAS5 March–May forecast.
