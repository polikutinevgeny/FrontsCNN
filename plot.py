import matplotlib.colors
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt

from crop import crop_center, crop_2d


def plot_results(x, y_true, y_pred, name, onehot, in_size):
    proj = ccrs.LambertConformal(
        central_latitude=50,
        central_longitude=-107,
        false_easting=5632642.22547,
        false_northing=4612545.65137,
        standard_parallels=(50, 50),
        cutoff=-30
    )
    f = plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 2, 1, projection=proj)
    ax.set_title("Prediction")
    plot_fronts(x, np.argmax(y_pred, axis=-1) if onehot else y_pred, proj, ax, in_size)
    ax = plt.subplot(1, 2, 2, projection=proj)
    ax.set_title("Ground truth")
    plot_fronts(x, np.argmax(y_true, axis=-1) if onehot else y_true, proj, ax, in_size)
    plt.savefig(name)
    plt.close(f)
    # plt.show()


def plot_fronts(x, y, proj, ax, in_size):
    with xr.open_dataset("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/air.2m.nc") as example:
        lat = crop_center(crop_2d(example.lat.values), in_size)
        lon = crop_center(crop_2d(example.lon.values), in_size)
        lon = (lon + 220) % 360 - 180  # Shift due to problems with crossing dateline in cartopy
    shift = ccrs.PlateCarree(central_longitude=-40)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)
    ax.set_extent((0, 1.129712e+07, 0, 8959788), crs=proj)
    plt.contourf(lon, lat, x[..., 0], levels=20, transform=shift)
    plt.contour(lon, lat, x[..., 1], levels=20, transform=shift)
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), 'red', 'blue', 'green', 'purple'])
    plt.pcolormesh(lon, lat, y, cmap=cmap, zorder=10, transform=shift)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
