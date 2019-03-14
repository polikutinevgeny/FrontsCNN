import math
import threading

import keras.utils
import numpy as np
import xarray as xr
from keras.utils import to_categorical

from crop import crop_center, crop_boundaries
from normalize import normalize


class Dataset(keras.utils.Sequence):
    def __init__(self, dates, filename, varnames, truth_filename, batch_size, in_size, onehot):
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

    def __enter__(self):
        self.file = xr.open_dataset(self.filename, cache=False, engine='netcdf4')
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
            y = self.truth_variable.sel(time=dates)[:, ::-1, :].values
            if self.onehot:
                y = to_categorical(y)
            else:
                y = np.expand_dims(y, axis=-1)
        x = crop_center(crop_boundaries(x), (len(dates), *self.in_size, 5))
        if self.onehot:
            y = crop_center(crop_boundaries(y), (len(dates), *self.in_size, 5))
        else:
            y = crop_center(crop_boundaries(y), (len(dates), *self.in_size, 1))
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.dates)
