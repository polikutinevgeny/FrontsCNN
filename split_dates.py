import xarray as xr
from sklearn.model_selection import train_test_split


def split_dates(truth_filename, data_filename):
    import random
    random.seed(0)
    with xr.open_dataset(truth_filename) as f, xr.open_dataset(data_filename) as f2:
        return train_test_split(list(set(f.time.values) & set(f2.time.values)), random_state=0, shuffle=True,
                                test_size=0.2)