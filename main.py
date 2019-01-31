import math
from model import Deeplabv3
from keras.optimizers import Adam
import numpy as np
import keras.backend as K
import keras.utils
import xarray as xr
from iou import mean_iou as onehot_mean_iou

deeplab_model = Deeplabv3(input_shape=(277, 349, 5), classes=5, weights=None, backbone='mobilenetv2')
optimizer = Adam(lr=1e-3)


def loss(y_true, y_pred):
    n_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(K.cast(y_true, "int32"), n_classes)
    return K.categorical_crossentropy(y_true, y_pred)


def mean_iou(y_true, y_pred):
    n_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(K.cast(y_true, "int32"), n_classes)
    return onehot_mean_iou(y_true, y_pred)


deeplab_model.compile(optimizer=optimizer, loss=loss, metrics=[mean_iou])


mean_std = {
    'air': (284.6689758300781, 14.781936645507812),
    'mslet': (101502.453125, 927.8274536132812),
    'shum': (0.009169235825538635, 0.005920239724218845),
    'uwnd': (-0.2797855734825134, 5.01072883605957),
    'vwnd': (-0.3674562871456146, 4.506356716156006)
}
filenames = ["air.2m.nc", "mslet.nc", "shum.2m.nc", "uwnd.10m.nc", "vwnd.10m.nc"]
varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
files = map(lambda x: "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/{}".format(x), filenames)


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
            self.files.append(xr.open_dataset(f))
            self.variables.append(self.files[-1][v])
        self.truth_file = xr.open_dataset(self.truth_filename)
        self.truth_variable = self.truth_file.fronts

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self.files:
            f.close()
        self.truth_file.close()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dates = self.dates[index * self.batch_size:(index + 1) * self.batch_size]
        return np.array([np.dstack((v[d] for v in self.variables)) for d in dates])

    def on_epoch_end(self):
        np.random.shuffle(self.dates)