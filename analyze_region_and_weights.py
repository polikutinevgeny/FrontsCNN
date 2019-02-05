import xarray as xr
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

filenames = ["air.2m.nc", "mslet.nc", "shum.2m.nc", "uwnd.10m.nc", "vwnd.10m.nc"]
varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
files = list(map(lambda x: "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/{}".format(x), filenames))
truth_filename = "plotted_fronts_fat.nc"

with xr.open_dataset(truth_filename) as f, xr.open_dataset(files[0]) as f2:
    dates = sorted(list(set(f.time.values) & set(f2.time.values)))

# mean and std of field values
# d = dict()
# for f, v in zip(files, varnames):
#     with xr.open_dataset(f, cache=False, chunks={'time': 100}) as ds:
#         d[v] = (float(ds[v].sel(time=dates).mean().compute().values), float(ds[v].std().compute().values))
# print(d)

# boundaries of fronts
with xr.open_dataset(truth_filename) as ds:
    ds.load()
    a = ds.fronts.sel(time=dates).values[:, ::-1, :]
# t = np.array(np.nonzero(a))
# print(t[1].min())
# print(t[1].max())
# print(t[2].min())
# print(t[2].max())

# with xr.open_dataset(truth_filename) as ds:
#     ds.load()
#     a = ds.fronts.sel(time=dates).values[:, ::-1, :]
unique, counts = np.unique(a, return_counts=True)
counts = counts.sum() / counts
counts = counts / counts.max()
print(dict(zip(unique, counts)))
