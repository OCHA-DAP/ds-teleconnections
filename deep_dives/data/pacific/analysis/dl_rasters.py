"""Windowed reads of SEAS5 (Nov issuances, lt0-5) and ERA5 monthly COGs over the SW Pacific (prod raster blob).
Region: lat 2N-26S, lon 145E-165W (two pieces across the dateline). Writes npz to data/rasters/."""
import ocha_stratus as stratus, numpy as np, xarray as xr, sys, os, re
from concurrent.futures import ThreadPoolExecutor
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
cc=stratus.get_container_client(container_name='raster', stage='prod')
def win(da):
    da=da.squeeze(drop=True)
    a=da.sel(y=slice(2,-26),x=slice(145,180)); b=da.sel(y=slice(2,-26),x=slice(-180,-165))
    b=b.assign_coords(x=b.x+360)
    return xr.concat([a,b],dim='x')
def get(name):
    da=stratus.open_blob_cog(name, stage='prod', container_name='raster', container_client=cc)
    w=win(da).load(); return w
which=sys.argv[2]
if which=='seas5':
    names=[f'seas5/monthly/processed/precip_em_i{y}-11-01_lt{l}.tif' for y in list(range(1981,2026))+[2026] for l in range(6)]
    def job(n):
        out=f"{OUT}/{os.path.basename(n)[:-4]}.nc"
        if os.path.exists(out): return
        try: get(n).to_netcdf(out)
        except Exception as ex: print('fail',n,ex)
    with ThreadPoolExecutor(8) as ex: list(ex.map(job,names))
else:
    allnames=stratus.list_container_blobs(name_starts_with='era5/monthly/processed/', container_name='raster', stage='prod')
    print(allnames[:3], len(allnames))
    pat=re.compile(r'(\d{4})-(\d{2})-01')
    want=[]
    for n in allnames:
        m=pat.search(n)
        if not m: continue
        y,mo=int(m.group(1)),int(m.group(2))
        if 1981<=y<=2026 and n.endswith('.tif'): want.append(n)
    print(len(want), want[:2])
    def job(n):
        out=f"{OUT}/{os.path.basename(n)[:-4]}.nc"
        if os.path.exists(out): return
        try: get(n).to_netcdf(out)
        except Exception as ex: print('fail',n,ex)
    with ThreadPoolExecutor(8) as ex: list(ex.map(job,want))
print('done', which)
