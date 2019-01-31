import xarray as xr
import numpy as np

filenames = ["air.2m.nc", "mslet.nc", "shum.2m.nc", "uwnd.10m.nc", "vwnd.10m.nc"]
varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
files = map(lambda x: "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/{}".format(x), filenames)

d = dict()
for f, v in zip(files, varnames):
    with xr.open_dataset(f, cache=False, chunks={'time': 100}) as ds:
        d[v] = (float(ds[v].mean().compute().values), float(ds[v].std().compute().values))
print(d)
