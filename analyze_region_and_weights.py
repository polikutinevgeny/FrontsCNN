import xarray as xr
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
####################################################
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.examples.waves import sample_data

# proj = ccrs.LambertConformal(
#         central_latitude=50,
#         central_longitude=-107,
#         false_easting=5632642.22547,
#         false_northing=4612545.65137,
#         standard_parallels=(50, 50),
#         cutoff=-30
#     )
proj = ccrs.PlateCarree(central_longitude=-107)
ax = plt.axes(projection=proj)
# ax.set_extent((0, 1.129712e+07, 0, 8959788), crs=proj)

with xr.open_dataset("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/air.2m.nc") as example:
    lats = example.lat.values
    lons = example.lon.values
    data = example.air[0].values
# lons, lats, data = sample_data(shape=(20, 40))
lons = (lons + 220) % 360 - 180
print(lons.max(), lons.min())

plt.contourf(lons, lats, lons, levels=20, transform=ccrs.PlateCarree(central_longitude=-40))

ax.coastlines()
ax.gridlines()

plt.show()
exit()
#########################################################
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
