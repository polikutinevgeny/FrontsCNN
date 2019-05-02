import math
import threading

import keras.utils
import numpy as np
import xarray as xr
from skimage.transform import rescale, downscale_local_mean
from keras.utils import to_categorical

import imgaug as ia

from crop import crop_center, crop_boundaries
from normalize import standardize, normalize


class Dataset(keras.utils.Sequence):
    def __init__(self, dates, filename, varnames, truth_filename, batch_size, in_size, onehot, add_mask, mask_model=None, augment=False, return_truth=True):
        self.dates = dates
        self.filename = filename
        self.varnames = varnames
        self.truth_filename = truth_filename
        self.file = None
        self.variables = None
        self.batch_size = batch_size
        self.len = int(math.ceil(len(dates) / batch_size))
        self.lock = threading.Lock()
        self.in_size = in_size
        self.onehot = onehot
        self.add_mask = add_mask
        self.mask_model = mask_model
        self.augment = augment
        self.return_truth = return_truth

    def __enter__(self):
        self.file = xr.open_dataset(self.filename, cache=False, engine='netcdf4')
        self.variables = []
        for v in self.varnames:
            self.variables.append(self.file[v])
        if self.return_truth:
            self.truth_file = xr.open_dataset(self.truth_filename, cache=False)
            self.truth_variable = self.truth_file.fronts
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if self.return_truth:
            self.truth_file.close()

    def __len__(self):
        return self.len

    def get_dates(self, index):
        return sorted(self.dates[index * self.batch_size:(index + 1) * self.batch_size])

    def __getitem__(self, index):
        with self.lock:
            dates = self.get_dates(index)
            x = np.concatenate(
                [np.expand_dims(standardize(v.sel(time=dates), v.name).fillna(0).values, -1) for v in self.variables],
                axis=-1)
            # x = np.concatenate(
            #     [np.expand_dims(normalize(v.sel(time=dates), v.name).fillna(0).values, -1) for v in self.variables],
            #     axis=-1)
            if self.return_truth:
                y = self.truth_variable.sel(time=dates)[:, ::-1, :].values
                if self.onehot:
                    y = to_categorical(y, 5)
                else:
                    y = np.expand_dims(y, axis=-1)
                # y = np.expand_dims((y == 1).astype(np.int), axis=-1)
        x = crop_center(crop_boundaries(x), (len(dates), *self.in_size, 5))
        if self.return_truth:
            if self.onehot:
                y = crop_center(crop_boundaries(y), (len(dates), *self.in_size, 5))
            else:
                y = crop_center(crop_boundaries(y), (len(dates), *self.in_size, 1))
        if self.add_mask:
            if self.mask_model:
                mask = self.mask_model.predict(x)
                x = np.append(x, mask, axis=-1)
            else:
                x = np.append(x, y[..., 1:].sum(axis=-1, keepdims=True), axis=-1)
        if self.augment:
            tx = []
            ty = []
            indices = np.random.randint(low=0, high=64, size=(x.shape[0], 2))
            for i in range(x.shape[0]):
                sx, sy = indices[i]
                tx.append(x[i, sx:sx + 192, sy:sy + 192, :])
                ty.append(y[i, sx:sx + 192, sy:sy + 192, :])
            x, y = np.array(tx), np.array(ty)
        if self.return_truth:
            return x, y
        else:
            return x

    def on_epoch_end(self):
        np.random.shuffle(self.dates)
