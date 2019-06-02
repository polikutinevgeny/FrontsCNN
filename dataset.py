import math

import keras.utils
import numpy as np
import xarray as xr
from keras.utils import to_categorical

from crop import crop_center, crop_boundaries
from normalize import standardize


class Dataset(keras.utils.Sequence):
    def __init__(self, dates, config):
        self.dates = dates
        self.filename = config.filename
        self.varnames = config.varnames
        self.truth_filename = config.truth_filename
        self.file = None
        self.truth_file = None
        self.variables = None
        self.truth_variable = None
        self.batch_size = config.batch_size
        self.len = int(math.ceil(len(self.dates) / self.batch_size))
        self.in_size = config.in_shape
        self.onehot = not config.binary
        self.standardize = config.standardize

    def __enter__(self):
        self.file = xr.open_dataset(self.filename, cache=False, engine='netcdf4')
        self.variables = []
        for v in self.varnames:
            self.variables.append(self.file[v])
        if self.truth_filename:
            self.truth_file = xr.open_dataset(self.truth_filename, cache=False)
            self.truth_variable = self.truth_file.fronts
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if self.truth_filename:
            self.truth_file.close()

    def __len__(self):
        return self.len

    def get_dates(self, index):
        return sorted(self.dates[index * self.batch_size:(index + 1) * self.batch_size])

    def get_x(self, index):
        return self[index][0]

    def get_y(self, index):
        return self[index][1]

    def __getitem__(self, index):
        dates = self.get_dates(index)
        x = np.concatenate(
            [
                np.expand_dims((standardize(v.sel(time=dates), v.name) if self.standardize else v.sel(time=dates))
                               .fillna(0).values, -1)
                for v in self.variables
            ],
            axis=-1)
        if x.ndim == 5:
            x = x.squeeze(axis=1)
        x = crop_center(crop_boundaries(x), (len(dates), *self.in_size, len(self.varnames)))
        if self.truth_filename:
            y = self.truth_variable.sel(time=dates)[:, ::-1, :].values
            if self.onehot:
                y = to_categorical(y, 5)
            else:
                y = np.expand_dims(y, axis=-1)
            if self.onehot:
                y = crop_center(crop_boundaries(y), (len(dates), *self.in_size, 5))
            else:
                y = crop_center(crop_boundaries(y), (len(dates), *self.in_size, 1))
            return x, y
        return x

    def on_epoch_end(self):
        np.random.shuffle(self.dates)
