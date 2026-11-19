"""Conditional check: how did the Nov-start NMME MAMJ signal verify in past El Nino winters?

Usage: uv run python -I 04_enso_analogs.py <nmme_agent_dir>
Writes out/enso_analog_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) / "out"
LIVE = ["CanESM5", "GEM5.2-NEMO", "CESM1", "CCSM4", "CFSv2"]
ens = pd.read_csv(OUT / "ensmean_seasonal_totals.csv")
obs = pd.read_csv(OUT / "obs_seasonal_totals.csv")
nino = pd.read_csv(OUT / "nino34_all_starts_members.csv")

# model-predicted NDJ Nino3.4 (MME mean of model ens means), by target (spring) year
n = (nino[nino.season == "NDJ"].groupby(["model", "start_year"]).nino34_anom.mean()
     .unstack(0)[[m for m in LIVE if m in nino.model.unique()]].mean(1))
n.index = n.index + 1
rows = []
for s in ["MAM", "AMJ", "MAMJ"]:
    w = (ens[(ens.method == "frac") & (ens.season == s) & ens.model.isin(LIVE)]
         .pivot(index="target_year", columns="model", values="ensmean")[LIVE])
    w = w[(w.index >= 1991)]
    z = np.log1p(w)
    hc = z.loc[:2026]
    zz = ((z - hc.mean()) / hc.std()).mean(1)
    pct_m = zz.loc[:2026].rank(pct=True)
    o = obs[obs.season == s].set_index("target_year").loc[1991:2026]
    pc = o.chirps_mm.rank(pct=True)
    pe = o.era5_mm.rank(pct=True)
    for y in zz.index:
        rows.append(dict(season=s, target_year=y, mme_nino34_NDJ=n.get(y, np.nan), mme_index=zz[y],
                         mme_pct=pct_m.get(y, (zz.loc[:2026] < zz[y]).mean()),
                         chirps_mm=o.chirps_mm.get(y, np.nan), chirps_pct=pc.get(y, np.nan),
                         era5_pct=pe.get(y, np.nan)))
d = pd.DataFrame(rows)
d.to_csv(OUT / "enso_analog_check.csv", index=False)
el = d[(d.mme_nino34_NDJ >= 1.0)]
pd.set_option("display.width", 200)
print(el.round(2).to_string(index=False))
for s in ["MAM", "AMJ", "MAMJ"]:
    e = el[(el.season == s) & (el.target_year <= 2026)]
    print(s, "El Nino (model NDJ>=1.0) springs:", len(e), "CHIRPS below-tercile count:",
          int((e.chirps_pct <= 1 / 3).sum()), "ERA5 below-tercile:", int((e.era5_pct <= 1 / 3).sum()),
          "mean CHIRPS pct:", round(e.chirps_pct.mean(), 2))
