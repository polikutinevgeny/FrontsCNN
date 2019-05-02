from config import Config
from dataset import Dataset
from plot import plot_fronts_far_east
from utils import trained_models
import xarray as xr

config = Config(
    model=trained_models['deeplab'],
    class_weights=1,
    in_shape=(256, 256),
    n_classes=5,
    varnames=["air", "mslet", "shum", "uwnd", "vwnd"],
    filename="/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/era5_lcc.nc",
    truth_filename=None,
    batch_size=16,
    binary=False,
    regularizer=None
)

with xr.open_dataset(config.filename) as f:
    dates = f.time.values
with Dataset(dates, config) as d:
    x = d[0]
    dates = d.get_dates(0)
    result = config.model.predict(x)
    for i in range(result.shape[0]):
        plot_fronts_far_east(x[i], result[i], "plots/plots_far_east_drom/deeplab_w{}.png".format(i), config.binary,
                             config.in_shape, dates[i])
