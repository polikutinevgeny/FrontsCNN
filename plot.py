import matplotlib.colors
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from confusion_matrix import plot_confusion_matrix

from crop import crop_center, crop_2d
from main import test_dataset


def plot_results(x, y_true, y_pred, name, onehot, in_size, date):
    proj = ccrs.LambertConformal(
        central_latitude=50,
        central_longitude=-107,
        false_easting=5632642.22547,
        false_northing=4612545.65137,
        standard_parallels=(50, 50),
        cutoff=-30
    )
    f = plt.figure(figsize=(16, 8))
    f.suptitle("Fronts at {}".format(date), fontsize=16)
    ax = plt.subplot(1, 2, 1, projection=proj)
    ax.set_title("Prediction")
    plot_fronts(x, np.argmax(y_pred, axis=-1) if onehot else y_pred, proj, ax, in_size)
    ax = plt.subplot(1, 2, 2, projection=proj)
    ax.set_title("Ground truth")
    plot_fronts(x, np.argmax(y_true, axis=-1) if onehot else y_true, proj, ax, in_size)
    plt.savefig(name)
    plt.close(f)


def plot_fronts(x, y, proj, ax, in_size):
    with xr.open_dataset("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/air.2m.nc") as example:
        lat = crop_center(crop_2d(example.lat.values), in_size)
        lon = crop_center(crop_2d(example.lon.values), in_size)
        lon = (lon + 220) % 360 - 180  # Shift due to problems with crossing dateline in cartopy
    shift = ccrs.PlateCarree(central_longitude=-40)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)
    ax.set_extent((2.0e+6, 1.039e+07, 6.0e+5, 8959788), crs=proj)
    plt.contourf(lon, lat, x[..., 0], levels=20, transform=shift)
    plt.contour(lon, lat, x[..., 1], levels=20, transform=shift, colors='black', linewidths=0.5)
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), 'red', 'blue', 'green', 'purple'])
    plt.pcolormesh(lon, lat, y, cmap=cmap, zorder=10, transform=shift)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    hot = mpatches.Patch(facecolor='red', label='Тёплый фронт', alpha=1)
    cold = mpatches.Patch(facecolor='blue', label='Холодный фронт', alpha=1)
    stat = mpatches.Patch(facecolor='green', label='Стационарный фронт', alpha=1)
    occl = mpatches.Patch(facecolor='purple', label='Фронт окклюзии', alpha=1)
    ax.legend(handles=[hot, cold, stat, occl], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
              prop={'size': 12})


def plot_fronts_far_east(x, y, name, onehot, in_size, date, bw=False):
    proj = ccrs.LambertConformal(
        central_latitude=50,
        central_longitude=130,
        false_easting=5632642.22547,
        false_northing=4612545.65137,
        standard_parallels=(50, 50),
        cutoff=-30
    )
    f = plt.figure(figsize=(8, 8))
    f.suptitle("Fronts at {}".format(date), fontsize=16)
    ax = plt.subplot(1, 1, 1, projection=proj)
    y = np.argmax(y, axis=-1) if onehot else y
    with xr.open_dataset("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/air.2m.nc") as example:
        lat = crop_center(crop_2d(example.lat.values), in_size)
        lon = crop_center(crop_2d(example.lon.values), in_size)  # Steal lat/lon from NARR
        lon = ((lon + 220) % 360 - 180 + 237) % 360  # Shift due to problems with crossing dateline in cartopy
    shift = ccrs.PlateCarree(central_longitude=-40)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)
    ax.set_extent((2.0e+6, 1.039e+07, 6.0e+5, 8959788), crs=proj)
    plt.contour(lon, lat, x[..., 1], levels=20, transform=shift, colors='black', linewidths=0.5)
    if bw:
        plt.pcolor(lon, lat, np.ma.masked_not_equal(y, 1), hatch="||||", alpha=0., transform=shift, zorder=100)
        plt.pcolor(lon, lat, np.ma.masked_not_equal(y, 2), hatch="----", alpha=0., transform=shift, zorder=100)
        plt.pcolor(lon, lat, np.ma.masked_not_equal(y, 3), hatch="oooo", alpha=0., transform=shift, zorder=100)
        plt.pcolor(lon, lat, np.ma.masked_not_equal(y, 4), hatch="++++", alpha=0., transform=shift, zorder=100)
        hot = mpatches.Patch(facecolor='white', label='Тёплый фронт', hatch="||||", alpha=1)
        cold = mpatches.Patch(facecolor='white', label='Холодный фронт', hatch="----", alpha=1)
        stat = mpatches.Patch(facecolor='white', label='Стационарный фронт', hatch="oooo", alpha=1)
        occl = mpatches.Patch(facecolor='white', label='Фронт окклюзии', hatch="++++", alpha=1)
        ax.legend(handles=[hot, cold, stat, occl], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
                  prop={'size': 12})
    else:
        plt.contourf(lon, lat, x[..., 0], levels=20, transform=shift)
        cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), 'red', 'blue', 'green', 'purple'])
        plt.pcolormesh(lon, lat, y, cmap=cmap, zorder=10, transform=shift)
        hot = mpatches.Patch(facecolor='red', label='Тёплый фронт', alpha=1)
        cold = mpatches.Patch(facecolor='blue', label='Холодный фронт', alpha=1)
        stat = mpatches.Patch(facecolor='green', label='Стационарный фронт', alpha=1)
        occl = mpatches.Patch(facecolor='purple', label='Фронт окклюзии', alpha=1)
        ax.legend(handles=[hot, cold, stat, occl], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
                  prop={'size': 12})
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.savefig(name)
    plt.close(f)


def plot_conf_matrix(y_true, y_pred, binary=False, normalize=True):
    if binary:
        plot_confusion_matrix(y_true, y_pred, ["No front", "Front"], normalize=normalize)
    else:
        plot_confusion_matrix(y_true, y_pred, ["No front", "Warm", "Cold", "Stationary", "Occlusion"],
                              normalize=normalize)
    plt.show()


def plot_sample(dataset, model, prefix, in_size, binary=False):
    x, y_true = dataset[0]
    dates = dataset.get_dates(0)
    if binary:
        y_pred = model.predict(x)[..., 0] > 0.5
        y_true = y_true[..., 0]
    else:
        y_pred = model.predict(x, batch_size=1)
    for i in range(x.shape[0]):
        plot_results(x[i], y_true[i], y_pred[i], "{}/{}".format(prefix, i), not binary, in_size, dates[i])


def plot_filtered(dataset, model, in_size, prefix, filter_func, binary=False):
    for m, i in zip(dataset, range(len(dataset))):
        x, y = m
        r = model.evaluate(x, y, verbose=0)
        d = test_dataset.get_dates(i)[0]
        if filter_func(r[1]):
            pred = model.predict(x)
            if binary:
                pred = pred > 0.5
                y[0] = y[0, ..., 0]
            plot_results(x[0], y[0], pred[0], "{2}/{0}_{1:.2f}.png".format(i, r[1], prefix), binary, in_size, d)


def plot_metrics_histogram(dataset, model, prefix):
    d = [[] for _ in model.keras_model.metrics_names]
    for (x, y) in dataset:
        r = model.evaluate(x, y, verbose=0)
        for i, j in zip(d, r):
            i.append(j)
    for n, i in zip(model.metrics_names, d):
        plt.hist(i, bins=100)
        plt.title(n)
        plt.savefig("{}/{}".format(prefix, n))
        plt.close()