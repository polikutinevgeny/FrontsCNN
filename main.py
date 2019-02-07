import math
import operator

# from model import Deeplabv3
from keras.optimizers import Adam
import numpy as np
import keras.backend as K
import keras.utils
import xarray as xr
from iou import mean_iou as onehot_mean_iou
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors
from segmentation_models import PSPNet, FPN, Linknet, Unet
from segmentation_models.losses import jaccard_loss, dice_loss
from segmentation_models.metrics import iou_score, f_score
from keras.utils import to_categorical
import threading
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

# import tensorflow as tf
# tf.enable_eager_execution()

# deeplab_model = Deeplabv3(input_shape=(277, 349, 5), classes=5, weights=None, backbone='mobilenetv2')
# deeplab_model = PSPNet(backbone_name="resnext50", input_shape=(240, 336, 5), classes=5, activation='softmax', encoder_weights=None)

deeplab_model = FPN(backbone_name="resnet34", input_shape=(256, 320, 5), classes=5, encoder_weights=None)

optimizer = Adam(lr=5e-4)

logdir = "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@Resnet34_Fat/"

# class_weights = {  # on 100 km wide fronts
#     0: 0.0031339743280361424,
#     1: 0.9212158330297883,
#     2: 0.32085642770167977,
#     3: 0.3145959219100464,
#     4: 1.0
# }
class_weights = {  # on 300 km wide fronts
    0: 0.008277938265164113,
    1: 0.9239984235226252,
    2: 0.3345056364872783,
    3: 0.3148883509015734,
    4: 1.0
}
w = list(class_weights.values())


def weighted_jaccard_loss(t, p):
    return jaccard_loss(t, p, class_weights=w)


def weighted_iou_score(t, p):
    return iou_score(t, p, class_weights=w)


def weighted_f_score(t, p):
    return f_score(t, p, class_weights=w)


deeplab_model = keras.models.load_model("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@Resnet34_Fat/weights167-0.800797.hdf5",
                                        custom_objects={
                                            "weighted_jaccard_loss": weighted_jaccard_loss,
                                            "weighted_iou_score": weighted_iou_score,
                                            "weighted_f_score": weighted_f_score
                                        })
# K.set_value(deeplab_model.optimizer.lr, 1e-4)

# deeplab_model.compile(
#     optimizer=optimizer,
#     loss=weighted_jaccard_loss,
#     metrics=[
#         weighted_iou_score,
#         weighted_f_score,
#         'categorical_accuracy',
#         'categorical_crossentropy'
#     ]
# )

# in_size = (240, 336)
in_size = (256, 320)

mean_std = {  # calculated on available dates
    'air': (284.80401611328125, 14.781936645507812),
    'mslet': (101505.328125, 927.8274536132812),
    'shum': (0.009227436035871506, 0.005920239724218845),
    'uwnd': (-0.2842417359352112, 5.01072883605957),
    'vwnd': (-0.3550795614719391, 4.506356716156006)
}
# boundaries = {  # on 100 km wide fronts
#     'min_x': 0,
#     'max_x': 340,
#     'min_y': 18,
#     'max_y': 276,
# }

boundaries = {  # on 300 km wide fronts
    'min_x': 0,
    'max_x': 344,
    'min_y': 15,
    'max_y': 276,
}


def normalize(a, name):
    mean, std = mean_std[name]
    return (a - mean) / std


def crop_center(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def crop_boundaries(imgs):
    return imgs[:, boundaries['min_y']:boundaries['max_y'] + 1, boundaries['min_x']:boundaries['max_x'] + 1, :]


def crop_2d(inp):
    return inp[boundaries['min_y']:boundaries['max_y'] + 1, boundaries['min_x']:boundaries['max_x'] + 1]


# filenames = ["air.2m.nc", "mslet.nc", "shum.2m.nc", "uwnd.10m.nc", "vwnd.10m.nc"]
varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
# files = list(map(lambda x: "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/{}".format(x), filenames))
# files = list(map(lambda x: "./data/{}".format(x), filenames))
filename = "/run/media/polikutin/Fast Data/NARR/narr_unc.nc"
truth_filename = "/run/media/polikutin/Fast Data/NARR/plotted_fronts_fat.nc"


def split_dates():
    import random
    random.seed(0)
    with xr.open_dataset(truth_filename) as f, xr.open_dataset(filename) as f2:
        return train_test_split(list(set(f.time.values) & set(f2.time.values)), random_state=0, shuffle=True,
                                test_size=0.2)


class Dataset(keras.utils.Sequence):
    def __init__(self, dates, filename, varnames, truth_filename, batch_size):
        self.dates = dates
        self.filename = filename
        self.varnames = varnames
        self.truth_filename = truth_filename
        self.file = None
        self.variables = None
        self.batch_size = batch_size
        self.len = int(math.ceil(len(dates) / batch_size))
        self.lock = threading.Lock()

    def __enter__(self):
        self.file = xr.open_dataset(self.filename, cache=False)
        self.variables = []
        for v in self.varnames:
            self.variables.append(self.file[v])
        self.truth_file = xr.open_dataset(self.truth_filename, cache=False)
        self.truth_variable = self.truth_file.fronts
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        self.truth_file.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        with self.lock:
            dates = sorted(self.dates[index * self.batch_size:(index + 1) * self.batch_size])
            x = np.concatenate(
                [np.expand_dims(normalize(v.sel(time=dates), v.name).fillna(0).values, -1) for v in self.variables],
                axis=-1)
            y = to_categorical(self.truth_variable.sel(time=dates)[:, ::-1, :].values)
        x = crop_center(crop_boundaries(x), (len(dates), *in_size, 5))
        y = crop_center(crop_boundaries(y), (len(dates), *in_size, 5))
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.dates)


def plot_results(x, y_true, y_pred, name):
    proj = ccrs.LambertConformal(
        central_latitude=50,
        central_longitude=-107,
        false_easting=5632642.22547,
        false_northing=4612545.65137,
        standard_parallels=(50, 50),
        cutoff=-30
    )
    plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 2, 1, projection=proj)
    plot_fronts(x, np.argmax(y_pred, axis=-1), proj, ax)
    ax = plt.subplot(1, 2, 2, projection=proj)
    plot_fronts(x, np.argmax(y_true, axis=-1), proj, ax)
    plt.savefig("plots/{}".name)
    plt.show()


def plot_fronts(x, y, proj, ax):
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


train, val = split_dates()
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=logdir + "weights{epoch:02d}-{val_loss:.6f}.hdf5"),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5),
    # keras.callbacks.EarlyStopping(patience=20),
    keras.callbacks.TensorBoard(log_dir=logdir)
]
with Dataset(train, filename, varnames, truth_filename, 16) as train_dataset, \
        Dataset(val, filename, varnames, truth_filename, 16) as val_dataset:
    # from timeit import default_timer as timer
    # t = timer()
    # for i in range(100):
    #     tmp = train_dataset[i]
    # print((timer() - t) / 100)
    temp_x, temp_y = val_dataset[0]
    # deeplab_model.fit_generator(
    #     generator=train_dataset,
    #     validation_data=val_dataset,
    #     use_multiprocessing=False,
    #     verbose=1,
    #     callbacks=callbacks,
    #     shuffle=False,
    #     workers=0,
    #     epochs=1000,
    # )

y_pred = deeplab_model.predict(temp_x)

for i in range(temp_x.shape[0]):
    plot_results(temp_x[i], temp_y[i], y_pred[i], i)
