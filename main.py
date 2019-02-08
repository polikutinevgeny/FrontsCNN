# from model import Deeplabv3
from keras.optimizers import Adam
import keras.utils
import xarray as xr

from dataset import Dataset
from sklearn.model_selection import train_test_split
from segmentation_models import FPN
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score, f_score

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

# import tensorflow as tf
# tf.enable_eager_execution()

# deeplab_model = Deeplabv3(input_shape=(277, 349, 5), classes=5, weights=None, backbone='mobilenetv2')
# deeplab_model = PSPNet(backbone_name="resnext50", input_shape=(240, 336, 5), classes=5, activation='softmax', encoder_weights=None)


class Config:
    model = FPN(backbone_name="inceptionv3", input_shape=(256, 320, 5), classes=5, encoder_weights=None)
    optimizer = Adam(lr=5e-4)
    logdir = "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@Inceptionv3_2/"
    class_weights = list({  # on 300 km wide fronts
                             0: 0.008277938265164113,
                             1: 0.9239984235226252,
                             2: 0.3345056364872783,
                             3: 0.3148883509015734,
                             4: 1.0
                         }.values())
    # in_size = (240, 336)
    in_size = (256, 320)
    # filenames = ["air.2m.nc", "mslet.nc", "shum.2m.nc", "uwnd.10m.nc", "vwnd.10m.nc"]
    varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
    # files = list(map(lambda x: "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomData2/NARR/{}".format(x), filenames))
    # files = list(map(lambda x: "./data/{}".format(x), filenames))
    filename = "/run/media/polikutin/Fast Data/NARR/narr_unc.nc"
    truth_filename = "/run/media/polikutin/Fast Data/NARR/plotted_fronts_fat.nc"

    def __init__(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=weighted_jaccard_loss,
            metrics=[
                weighted_iou_score,
            ]
        )


def weighted_jaccard_loss(t, p):
    return jaccard_loss(t, p, class_weights=Config.class_weights)


def weighted_iou_score(t, p):
    return iou_score(t, p, class_weights=Config.class_weights)


def weighted_f_score(t, p):
    return f_score(t, p, class_weights=Config.class_weights)


# Config.model = keras.models.load_model("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@Resnet34_Fat/weights167-0.800797.hdf5",
#                                         custom_objects={
#                                             "weighted_jaccard_loss": weighted_jaccard_loss,
#                                             "weighted_iou_score": weighted_iou_score,
#                                             "weighted_f_score": weighted_f_score
#                                         })
# K.set_value(Config.model.optimizer.lr, 1e-4)


def split_dates():
    import random
    random.seed(0)
    with xr.open_dataset(Config.truth_filename) as f, xr.open_dataset(Config.filename) as f2:
        return train_test_split(list(set(f.time.values) & set(f2.time.values)), random_state=0, shuffle=True,
                                test_size=0.2)


train, val = split_dates()
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=Config.logdir + "weights{epoch:02d}-{val_loss:.6f}.hdf5"),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5),
    # keras.callbacks.EarlyStopping(patience=20),
    keras.callbacks.TensorBoard(log_dir=Config.logdir)
]
with Dataset(train, Config.filename, Config.varnames, Config.truth_filename, 16, Config.in_size) as train_dataset, \
        Dataset(val, Config.filename, Config.varnames, Config.truth_filename, 16, Config.in_size) as val_dataset:
    # from timeit import default_timer as timer
    # t = timer()
    # for i in range(100):
    #     tmp = train_dataset[i]
    # print((timer() - t) / 100)
    # temp_x, temp_y = val_dataset[0]
    Config.model.fit_generator(
        generator=train_dataset,
        validation_data=val_dataset,
        use_multiprocessing=False,
        verbose=1,
        callbacks=callbacks,
        shuffle=False,
        workers=0,
        epochs=1000,
    )

# y_pred = deeplab_model.predict(temp_x)
#
# for i in range(temp_x.shape[0]):
#     plot_results(temp_x[i], temp_y[i], y_pred[i], "plots/{}".format(i))
