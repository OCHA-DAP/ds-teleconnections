# Fiji, Solomon Islands, Tonga and Vanuatu: 2026/27 wet season (analysis scripts)

Run on 17 November 2026 to build `/enso/pacific/` (page builder: `pac_deep_dive.py` at the repo root). Each script takes a
work directory `SP` (inputs go to `SP/data/`). Run from the repo root with `uv run python -I`. Database and blob reads are
read-only, through ocha-stratus with the connection settings from the environment.

| Step | Script | Output |
|---|---|---|
| Database tables (SEAS5 and ERA5 admin 0/1, CERF), NOAA indices, COD-AB boundaries, GHCN-Daily gauges | `get_inputs.py SP` | `SP/data/*`, `SP/cerf_pacific.csv` |
| SEAS5 November starts (lead 0–5, 1981–2026) and ERA5 monthly, windowed to 2°N–26°S, 145°E–165°W (prod raster blob) | `dl_rasters.py SP/data/rasters/seas5 seas5`, then `dl_rasters.py SP/data/rasters/era5 era5` | `SP/data/rasters/{seas5,era5}/*.nc` |
| CHIRPS v2 July–October 1981–2026 for Tonga and Fiji (CHC COGs, one windowed read at a time: CHC rate-limits parallel reads) | `chc_window.py SP/data/chc/ton_jaso.npz -176.3 -173.6 -22.4 -15.5 7,8,9,10 1981 2026`, and the same for `fji_e` (176.8–180°E) and `fji_w` (180–178°W) | `SP/data/chc/*.npz` |
| CHIRPS check of the July–October rain so far against ERA5 | `chc_units.py SP` | `SP/data/chirps_jaso.json` (read by `synth.py` when present) |
| Page data and tables | `synth.py SP deep_dives/data/pacific` | `page_data.json`, `results/*.csv` |

`dl_chirps.py` and `chirps_units.py` (CHIRPS from the IRI Data Library) were tried first; the IRI server did not answer
in time on 17 November 2026, so the page uses the CHC route above. The Fiji CHC fetch hit CHC's rate limit (HTTP 429) on
17 November 2026 and was stopped; only Tonga's CHIRPS check is on the page (Fiji has gauges).

Definitions used throughout:

- Forecast percentages are of the 1991–2020 mean November SEAS5 forecast for the same months and area (no detrending);
  observed percentages are of the 1991–2020 ERA5 mean. Ranks are among the 46 November forecasts 1981–2026.
- Skill is the correlation between the November forecast and ERA5 over 1982–2025.
- El Niño share: straight-line fits of the November–April forecast (and of ERA5) on Niño3.4 known at the start of
  November (OISST October from `sstoi.indices`, PSL ERSST v6 October, ONI August–October), 1982–2025, each with and
  without a linear year term. Neutral year = the fit at Niño3.4 = 0 for 2026; El Niño share = the Niño3.4 term at the
  2026 value over the gap between the neutral year and the 2026 forecast, capped at 100%. "Share of variation" = squared
  correlation after removing a linear trend from both series.
- Observed rainfall that past forecasts point to: linear fit of ERA5 % on forecast %, 1982–2025, 80% prediction interval
  (with the fit's own uncertainty, since 2026 lies outside the range of past forecasts).
- Solomon Islands groups are equal-weight means of provinces: south-east SB05, SB08, SB09; centre SB04, SB06, SB07;
  north-west SB01, SB02, SB03. Honiara (SB10) has no ERA5 cell; Ha'apai (TO3) has no SEAS5 cell.
- Fiji gauges: GHCN-Daily, months with at most two missing days; November–April totals only when all six months pass.
