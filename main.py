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

logdir = "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@Resnet34_RAM_Fat/"

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


deeplab_model = keras.models.load_model("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@Resnet34_RAM_Fat/weights342-0.881963.hdf5",
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


filenames = ["air.2m.nc", "mslet.nc", "shum.2m.nc", "uwnd.10m.nc", "vwnd.10m.nc"]
varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
# files = list(map(lambda x: "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/{}".format(x), filenames))
files = list(map(lambda x: "./data/{}".format(x), filenames))
truth_filename = "plotted_fronts_fat.nc"


def split_dates():
    import random
    random.seed(0)
    with xr.open_dataset(truth_filename) as f, xr.open_dataset(files[0]) as f2:
        return train_test_split(list(set(f.time.values) & set(f2.time.values)), random_state=0, shuffle=True,
                                test_size=0.2)


class Dataset(keras.utils.Sequence):
    def __init__(self, dates, filenames, varnames, truth_filename, batch_size):
        self.dates = dates
        self.filenames = filenames
        self.varnames = varnames
        self.truth_filename = truth_filename
        self.files = None
        self.variables = None
        self.batch_size = batch_size
        self.len = int(math.ceil(len(dates) / batch_size))

    def __enter__(self):
        self.files = []
        self.variables = []
        for f, v in zip(self.filenames, self.varnames):
            self.files.append(xr.open_dataset(f, cache=False))
            self.variables.append(self.files[-1][v])
        self.truth_file = xr.open_dataset(self.truth_filename, cache=False)
        self.truth_variable = self.truth_file.fronts
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self.files:
            f.close()
        self.truth_file.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dates = sorted(self.dates[index * self.batch_size:(index + 1) * self.batch_size])
        # x = np.array([np.dstack([normalize(v.sel(time=d), v.name).fillna(0).values for v in self.variables]) for d in dates])
        # y = to_categorical(np.array([self.truth_variable.sel(time=d).values[::-1, ::] for d in dates]))
        x = np.concatenate(
            [np.expand_dims(normalize(v.sel(time=dates).fillna(0).values, v.name), -1) for v in self.variables],
            axis=-1)
        y = to_categorical(self.truth_variable.sel(time=dates)[:, ::-1, :].values)
        # if x.shape != (self.batch_size, 277, 349, 5):
        #     raise Exception("What the fuck")
        x = crop_center(crop_boundaries(x), (len(dates), *in_size, 5))
        y = crop_center(crop_boundaries(y), (len(dates), *in_size, 5))
        return x, y
        # np.array([np.dstack([crop_center(normalize(v.sel(time=d), v.name).fillna(0).values, (240, 336, 5)) for v in self.variables]) for d in dates]),\
        # to_categorical(np.array([crop_center(self.truth_variable.sel(time=d).values, (240, 336))[::-1, ::] for d in dates]))

    def on_epoch_end(self):
        np.random.shuffle(self.dates)


def plot_fronts(x, y):
    plt.imshow(x[..., 0])
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0), 'red', 'blue', 'green', 'purple'])
    plt.imshow(y, cmap=cmap, interpolation='nearest', origin='lower')
    plt.show()


train, val = split_dates()
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=logdir + "weights{epoch:02d}-{val_loss:.6f}.hdf5"),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5),
    # keras.callbacks.EarlyStopping(patience=20),
    keras.callbacks.TensorBoard(log_dir=logdir)
]
with Dataset(train, files, varnames, truth_filename, 32) as train_dataset, \
        Dataset(val, files, varnames, truth_filename, 32) as val_dataset:
    temp_x, temp_y = val_dataset[0]
    # x_train = np.random.randn(40 * 32, *in_size, 5)
    # y_train = np.random.randn(40 * 32, *in_size, 5)
    # for i in range(40):
    #     print("{} train".format(i))
    #     x_train[i * 32:(i + 1) * 32], y_train[i * 32:(i + 1) * 32] = train_dataset[i]
    # x_val = np.random.randn(10 * 32, *in_size, 5)
    # y_val = np.random.randn(10 * 32, *in_size, 5)
    # for i in range(10):
    #     print("{} val".format(i))
    #     x_val[i * 32:(i + 1) * 32], y_val[i * 32:(i + 1) * 32] = val_dataset[i]
    # np.savez("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/data_ram_fat(FPN).npz", x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    # deeplab_model.fit_generator(
    #     generator=train_dataset,
    #     validation_data=val_dataset,
    #     use_multiprocessing=False,
    #     verbose=1,
    #     callbacks=callbacks,
    #     steps_per_epoch=100,
    #     validation_steps=20,
    #     shuffle=False,
    #     workers=0,
    #     epochs=1000,
    # )
# print(x_train.shape, y_train.shape)
# f = np.load("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/data_ram.npz")
# x_train, y_train, x_val, y_val = f["arr_0"], f["arr_1"], f["arr_2"], f["arr_3"]
# f = np.load("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/data_ram(FPN).npz")
# x_train, y_train, x_val, y_val = f["x_train"], f["y_train"], f["x_val"], f["y_val"]
# f = np.load("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/data_ram_fat(FPN).npz")
# x_train, y_train, x_val, y_val = f["x_train"], f["y_train"], f["x_val"], f["y_val"]
# deeplab_model.fit(x=x_train, y=y_train, batch_size=16, epochs=1000, callbacks=callbacks, validation_data=(x_val, y_val))

y_pred = deeplab_model.predict(temp_x)
for i in range(temp_x.shape[0]):
    plot_fronts(temp_x[i], np.argmax(y_pred[i], axis=-1))
    plot_fronts(temp_x[i], np.argmax(temp_y[i], axis=-1))
# plot_fronts(x_train[0], np.argmax(y_train[0], axis=-1))