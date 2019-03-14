import xarray as xr
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
file = "/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/narr.nc"
truth_filename = "/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/plotted_fronts_fat_binary.nc"

with xr.open_dataset(truth_filename) as f, xr.open_dataset(file) as f2:
    dates = sorted(list(set(f.time.values) & set(f2.time.values)))

# mean and std of field values
# d = dict()
# with xr.open_dataset(file, cache=False, chunks={'time': 100}) as ds:
#     for v in varnames:
#         d[v] = (float(ds[v].sel(time=dates).mean().compute().values), float(ds[v].std().compute().values))
# print(d)

# boundaries of fronts
with xr.open_dataset(truth_filename) as ds:
    ds.load()
    a = ds.fronts.sel(time=dates).values[:, ::-1, :]
t = np.array(np.nonzero(a.sum(axis=0) > 50))
print(t[0].min())
print(t[0].max())
print(t[1].min())
print(t[1].max())

# with xr.open_dataset(truth_filename) as ds:
#     ds.load()
#     a = ds.fronts.sel(time=dates).values[:, ::-1, :]
# unique, counts = np.unique(a, return_counts=True)
# counts = counts.sum() / counts
# counts = counts / counts.max()
# print(dict(zip(unique, counts)))

# s = a.sum(axis=0) > 20
# from cartopy import crs as ccrs
# from matplotlib import pyplot as plt
# proj = ccrs.LambertConformal(
#         central_latitude=50,
#         central_longitude=-107,
#         false_easting=5632642.22547,
#         false_northing=4612545.65137,
#         standard_parallels=(50, 50),
#         cutoff=-30
# )
# with xr.open_dataset("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/air.2m.nc") as example:
#     lat = example.lat.values
#     lon = example.lon.values
#     lon = (lon + 220) % 360 - 180  # Shift due to problems with crossing dateline in cartopy
# ax = plt.axes(projection=proj)
# shift = ccrs.PlateCarree(central_longitude=-40)
# ax.set_extent((0, 1.129712e+07, 0, 8959788), crs=proj)
# plt.pcolormesh(lon, lat, s, zorder=-1, transform=shift)
# ax.coastlines()
# ax.gridlines(draw_labels=True)
# plt.show()