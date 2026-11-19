"""Conditional check: how did the Nov-start NMME MJJA / SO rainfall signal verify after past El Nino winters?

For Nov starts whose model NDJ Nino3.4 (MME mean of model ensemble means) was >= 1.0, list the MME rainfall
index next to the observed CHIRPS (and ERA5) % of the 1991-2020 mean, per country.

Usage: uv run python -I 04_enso_analogs.py <nmme_dir>
Reads out/{ensmean_seasonal_totals, obs_seasonal_totals, nino34_all_starts_members, nino34_obs_monthly,
live_models}.csv; writes out/enso_analog_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) / "out"
CLIM = (1991, 2020)
HC = (1991, 2026)
ens = pd.read_csv(OUT / "ensmean_seasonal_totals.csv")
obs = pd.read_csv(OUT / "obs_seasonal_totals.csv")
nino = pd.read_csv(OUT / "nino34_all_starts_members.csv")
no = pd.read_csv(OUT / "nino34_obs_monthly.csv")
live = pd.read_csv(OUT / "live_models.csv").groupby("season").model.apply(list).to_dict()

# model-predicted NDJ Nino3.4 (MME mean of model ensemble means), indexed by rainfall target year
n = nino[nino.season == "NDJ"].groupby(["model", "start_year"]).nino34_anom.mean().unstack(0).mean(1)
n.index = n.index + 1
# observed NDJ (Nov-Jan before the target year), 1991-2020 base
key = no.set_index(["year", "month"]).nino34_9120
obs_ndj = {y: np.mean([key.get((y - 1, 11), np.nan), key.get((y - 1, 12), np.nan), key.get((y, 1), np.nan)])
           for y in range(1983, 2028)}

rows = []
for cc in sorted(obs.country.unique()):
    for s, models in live.items():
        e = ens[(ens.country == cc) & (ens.method == "frac") & (ens.season == s) & ens.model.isin(models)]
        w = e.pivot(index="target_year", columns="model", values="ensmean")[models]
        w = w[w.index >= HC[0]].dropna()
        z = np.log1p(w)
        hc = z.loc[: HC[1]]
        zz = ((z - hc.mean()) / hc.std()).mean(1)  # equal weight per model, standardised log1p ens means
        zz = (zz - zz.loc[: HC[1]].mean()) / zz.loc[: HC[1]].std()
        pct_model = (100 * w / w.loc[CLIM[0]:CLIM[1]].mean()).mean(1)  # each model vs its own 1991-2020 mean
        o = obs[(obs.country == cc) & (obs.season == s)].set_index("target_year")
        cm = o.loc[CLIM[0]:CLIM[1], ["chirps_mm", "era5_mm"]].mean()
        oh = o.loc[HC[0]:HC[1]]
        for y in zz.index:
            ch = o.chirps_mm.get(y, np.nan)
            er = o.era5_mm.get(y, np.nan)
            rows.append(dict(
                country=cc, season=s, target_year=y, n_models=len(models),
                mme_nino34_NDJ=n.get(y, np.nan), obs_nino34_NDJ=obs_ndj.get(y, np.nan),
                mme_rain_index_z=zz[y], mme_rain_rank=f"{int((zz.loc[:HC[1]] < zz[y]).sum()) + 1}/{len(zz.loc[:HC[1]]) + (y > HC[1])}",
                mme_rain_pct_model_clim=pct_model[y],
                chirps_mm=ch, chirps_pct_9120=100 * ch / cm.chirps_mm,
                chirps_rank=(f"{int((oh.chirps_mm.dropna() < ch).sum()) + 1}/{oh.chirps_mm.notna().sum()}"
                             if np.isfinite(ch) else ""),
                era5_pct_9120=100 * er / cm.era5_mm if np.isfinite(er) else np.nan))
d = pd.DataFrame(rows)
d.to_csv(OUT / "enso_analog_check.csv", index=False)
el = d[d.mme_nino34_NDJ >= 1.0]
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)
print("Analog years: Nov starts with model MME NDJ Nino3.4 >= 1.0 (target year = following year); 2027 = forecast.")
print("mme_rain_index_z: standardised MME rainfall index (0 = model normal, negative = drier); ranks 1 = driest.")
for (cc, s), g in el.groupby(["country", "season"]):
    print(f"\n--- {cc} {s} ---")
    print(g.drop(columns=["country", "season"]).round(2).to_string(index=False))
print("\nSummary over analog years with observations (target <= 2026):")
for (cc, s), g in el[el.target_year <= HC[1]].groupby(["country", "season"]):
    g = g.dropna(subset=["chirps_pct_9120"])
    r = np.corrcoef(g.mme_rain_index_z, g.chirps_pct_9120)[0, 1] if len(g) > 3 else np.nan
    print(f"{cc} {s}: n={len(g)} years {g.target_year.tolist()}; mean CHIRPS {g.chirps_pct_9120.mean():.0f}% "
          f"(range {g.chirps_pct_9120.min():.0f}-{g.chirps_pct_9120.max():.0f}%), "
          f"below 90%: {int((g.chirps_pct_9120 < 90).sum())}, mean ERA5 {g.era5_pct_9120.mean():.0f}%; "
          f"mean MME index z {g.mme_rain_index_z.mean():+.2f}; r(index, CHIRPS%) {r:.2f}")
