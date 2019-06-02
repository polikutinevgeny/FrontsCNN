import numpy as np
import xarray as xr
from dataset import Dataset
from config import Config
from utils import load_indexing
from crop import crop_center, crop_2d
from sklearn.metrics import jaccard_similarity_score


def equivalent_potential_temperature(temperature, specific_humidity, pressure):
    e = pressure / (622 + specific_humidity)
    tl = 2840 / (3.5 * np.log(temperature) - np.log(e) - 4.805) + 55
    return temperature * (1000 / pressure) ** (0.2854 * (1 - 0.28 * 0.001 * specific_humidity)) * \
           np.exp((3.376 / tl - 0.00254) * specific_humidity * (1 + 0.81 * 0.001 * specific_humidity))


def gradient_lcc(field):
    return np.gradient(field, 32.463)


def abs_lcc(dx, dy):
    return np.sqrt(dx ** 2 + dy ** 2)


def thermal_front_parameter(temperature):
    gx, gy = gradient_lcc(temperature)
    ga = abs_lcc(gx, gy)
    ggx, ggy = gradient_lcc(ga)
    return - (ggx * gx + ggy * gy) / ga


def coriolis_parameter(lat):
    return 2 * 7.2921 * 10 ** -5 * np.sin(np.deg2rad(lat))


def iou_numpy(outputs: np.array, labels: np.array):
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    return (intersection + 1e-12) / (union + 1e-12)


class HewsonCriterion:
    def __init__(self, tfp_threshold, grad_threshold):
        self.tfp_threshold = tfp_threshold / 10000
        self.grad_threshold = grad_threshold / 100

    def apply(self, temperature):
        tfp = thermal_front_parameter(temperature)
        temp_grad = abs_lcc(*gradient_lcc(temperature))
        grad = temp_grad + 1 / np.sqrt(2) * 32.463 * abs_lcc(*gradient_lcc(temp_grad))
        return np.logical_and(tfp > self.tfp_threshold, grad > self.grad_threshold)


class ParfittCriterion:
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, temperature, vorticity, coriolis):
        grad = abs_lcc(*gradient_lcc(temperature))
        return vorticity * grad / (0.45 / 100 * coriolis) > self.threshold


class HewsonVorticityCriterion:
    def __init__(self, tfp_threshold, grad_threshold, vorticity_threshold):
        self.tfp_threshold = tfp_threshold / 10000
        self.grad_threshold = grad_threshold / 100
        self.vorticity_threshold = vorticity_threshold

    def apply(self, temperature, vorticity, coriolis):
        tfp = thermal_front_parameter(temperature)
        temp_grad = abs_lcc(*gradient_lcc(temperature))
        grad = temp_grad + 1 / np.sqrt(2) * 32.463 * abs_lcc(*gradient_lcc(temp_grad))
        vort = vorticity / coriolis
        return np.logical_and.reduce(
            (tfp > self.tfp_threshold, grad > self.grad_threshold, vort > self.vorticity_threshold))


config = Config(
    in_shape=(256, 256),
    n_classes=5,
    varnames=["T", "Q", "VO"],
    filename="/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/ERA5_parameter/era5_regridded.nc",
    truth_filename="/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/plotted_fronts_fat_binary.nc",
    batch_size=1,
    n_dims=3,
    binary=True,
    standardize=False
)

train, val, test = load_indexing("indexing.npz")

crit = ParfittCriterion(1)
# crit = HewsonVorticityCriterion(0.75, 1.5, 0.8)
# crit = HewsonCriterion(0.75, 3)

with xr.open_dataset("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/air.2m.nc") as example:
    lat = crop_center(crop_2d(example.lat.values), config.in_shape)
    lon = crop_center(crop_2d(example.lon.values), config.in_shape)
    lon = (lon + 220) % 360 - 180  # Shift due to problems with crossing dateline in cartopy
coriolis = coriolis_parameter(lat)
with Dataset(test, config) as test_dataset:
    for i in range(len(test_dataset)):
        x, y_true = test_dataset[i]
        y_true = y_true[0, ..., 0]
        etemp = equivalent_potential_temperature(x[0, ..., 0], x[0, ..., 1], 900)
        y_pred = crit.apply(etemp, x[0, ..., 2], coriolis)
        # y_pred = crit.apply(etemp)
        from plot import plot_results
        plot_results(etemp, y_true, y_pred, "plots/criterion/parfitt 1/{}.png".format(i), config.in_shape, test_dataset.get_dates(i)[0],
                     bw=True, binary=True)
        if i > 16:
            break
        # import matplotlib.colors
        # from matplotlib import pyplot as plt
        # from cartopy import crs as ccrs
        # proj = ccrs.LambertConformal(
        #     central_latitude=50,
        #     central_longitude=-107,
        #     false_easting=5632642.22547,
        #     false_northing=4612545.65137,
        #     standard_parallels=(50, 50),
        #     cutoff=-30
        # )
        # f = plt.figure(figsize=(8, 8))
        # ax = plt.subplot(1, 1, 1, projection=proj)
        # shift = ccrs.PlateCarree(central_longitude=-40)
        # ax.set_xmargin(0.1)
        # ax.set_ymargin(0.1)
        # ax.set_extent((2.0e+6, 1.039e+07, 6.0e+5, 8959788), crs=proj)
        # plt.contourf(lon, lat, etemp, levels=20, transform=shift)
        # cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), 'purple'])
        # plt.pcolormesh(lon, lat, y_pred, cmap=cmap, zorder=10, transform=shift)
        # ax.coastlines()
        # ax.gridlines(draw_labels=True)
        # plt.show()
