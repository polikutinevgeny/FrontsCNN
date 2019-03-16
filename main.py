from keras.engine.saving import model_from_json
from keras.optimizers import Adam
import keras.utils

from dataset import Dataset
from segmentation_models import FPN, PSPNet, Unet, Linknet
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score, f_score
import numpy as np
import keras.backend as K

from plot import plot_results
from segnet import segnet
from keras_contrib.applications.densenet import DenseNetFCN
from pspnet import PSPNet50
from deeplabv3plus import Deeplabv3

from keras_contrib.losses import jaccard_distance


# def fronts_loss(t, p):
#     return jaccard_loss(t[..., 1:], K.softmax(p[..., 1:]))


def weighted_jaccard_loss(t, p):
    return jaccard_loss(t, p, class_weights=Config.class_weights)


def weighted_iou_score(t, p):
    return iou_score(t, p, class_weights=Config.class_weights)


def weighted_f_score(t, p):
    return f_score(t, p, class_weights=Config.class_weights)


def iou_metric_all(t, p):
    return iou_score(t, K.one_hot(K.argmax(p, axis=-1), 5))\


def iou_metric_fronts(t, p):
    # p = cutoff(p)
    return iou_score(t[..., 1:], K.one_hot(K.argmax(p, axis=-1), 5)[..., 1:])


def iou_metric_hot(t, p):
    # p = cutoff(p)
    return iou_score(t[..., 1:2], K.one_hot(K.argmax(p), 5)[..., 1:2])


def iou_metric_cold(t, p):
    # p = cutoff(p)
    return iou_score(t[..., 2:3], K.one_hot(K.argmax(p), 5)[..., 2:3])


def iou_metric_stationary(t, p):
    # p = cutoff(p)
    return iou_score(t[..., 3:4], K.one_hot(K.argmax(p), 5)[..., 3:4])


def iou_metric_occlusion(t, p):
    # p = cutoff(p)
    return iou_score(t[..., 4:5], K.one_hot(K.argmax(p), 5)[..., 4:5])


def cutoff(p):
    empty = K.cast(p[..., 0] > 0.5, K.floatx())
    r0 = p[..., 0] * empty
    return K.concatenate([K.expand_dims(r0), p[..., 1:]], axis=-1)


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss


def set_regularization(model,
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None):
    for layer in model.layers:
        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer
        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer
        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer
    out = model_from_json(model.to_json())
    out.set_weights(model.get_weights())
    return out


# binary_weights = list({0: 0.0685456944293724, 1: 1.0}.values())
binary_weights = 1
class_weights = [
     0.008277938265164113,
     0.9239984235226252,
     0.3345056364872783,
     0.3148883509015734,
     1.0
]


class Config:
    model = FPN(backbone_name="resnet34", input_shape=(None, None, 5), classes=5, encoder_weights=None)
    # model = PSPNet(backbone_name="resnext50", input_shape=(240, 336, 5), classes=5, activation='softmax', encoder_weights=None)
    # model = Unet(backbone_name="inceptionv3", encoder_weights=None, input_shape=(256, 256, 5))
    # model = Linknet(backbone_name="resnet34", input_shape=(256, 256, 5), classes=5, encoder_weights=None, activation="softmax")
    # model = segnet(input_shape=(256, 256, 5), n_labels=5)
    # model = DenseNetFCN(input_shape=(256, 256, 5), weights=None, classes=5, nb_layers_per_block=4, growth_rate=13, dropout_rate=0.2)
    # model = PSPNet50(input_shape=(256, 256, 5), n_labels=5)
    # model = Deeplabv3(weights=None, input_shape=(256, 256, 5), classes=5)
    optimizer = Adam(lr=5e-4)
    logdir = "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/FPN@resnet34/"
    class_weights = class_weights
    # in_size = (240, 336)
    # in_size = (256, 320)
    in_size = (256, 256)  # new crop boundaries
    varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
    filename = "/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/narr.nc"
    truth_filename = "/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/plotted_fronts_fat.nc"
    batch_size = 16
    onehot = True
    regularizer = None
    add_mask = False
    mask_model = None


Config.mask_model = keras.models.load_model("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Unet@inceptionv3_Fat_Binary/weights154-0.380389.hdf5",
                                        custom_objects={
                                            "weighted_jaccard_loss": weighted_jaccard_loss,
                                            "weighted_iou_score": weighted_iou_score,
                                            "weighted_f_score": weighted_f_score
                                        })


# Config.model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=(256, 256, 5), activation='relu'),
#     keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=(256, 256, 5), activation='relu'),
#     keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=(256, 256, 5), activation='relu'),
#     keras.layers.Conv2D(filters=5, kernel_size=(5, 5), padding="same", input_shape=(256, 256, 5), activation='softmax')
# ])

# set_regularization(Config.model, kernel_regularizer=Config.regularizer, bias_regularizer=Config.regularizer)
# Config.model.load_weights("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Deeplabv3plus/weights48-0.517574.hdf5")
Config.model.compile(
        optimizer=Config.optimizer,
        loss=weighted_jaccard_loss,
        metrics=[
            # weighted_iou_score,
            # weighted_f_score,
            iou_metric_all,
            iou_metric_fronts,
            iou_metric_hot,
            iou_metric_cold,
            iou_metric_stationary,
            iou_metric_occlusion
        ]
    )

# Config.model = keras.models.load_model(#"/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_mask/weights55-0.277438.hdf5",
#                                         "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Deeplabv3plus/weights101-0.504555.hdf5",
#                                         custom_objects={
#                                             "weighted_jaccard_loss": weighted_jaccard_loss,
#                                             "weighted_iou_score": weighted_iou_score,
#                                             "weighted_f_score": weighted_f_score
#                                         })
# Config.model.compile(
#         optimizer=Config.optimizer,
#         loss=weighted_jaccard_loss,
#         metrics=[
#             weighted_iou_score,
#             weighted_f_score,
#             iou_metric_all,
#             iou_metric_fronts,
#             iou_metric_hot,
#             iou_metric_cold,
#             iou_metric_stationary,
#             iou_metric_occlusion
#         ]
#     )
# K.set_value(Config.model.optimizer.lr, 5e-4)

Config.mask_model._make_predict_function()

indexing = np.load("indexing.npz")
train, val, test = indexing["train"], indexing["val"], indexing["test"]

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=Config.logdir + "weights{epoch:02d}-{val_loss:.6f}.hdf5", save_best_only=True, monitor="val_iou_metric_fronts", mode="max"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_iou_metric_fronts", factor=0.5, mode='max'),
    # keras.callbacks.EarlyStopping(patience=20),
    keras.callbacks.TensorBoard(log_dir=Config.logdir)
]
with Dataset(train, Config.filename, Config.varnames, Config.truth_filename, Config.batch_size,
             Config.in_size, onehot=Config.onehot, add_mask=Config.add_mask, mask_model=Config.mask_model) as train_dataset, \
        Dataset(val, Config.filename, Config.varnames, Config.truth_filename, Config.batch_size,
                Config.in_size, onehot=Config.onehot, add_mask=Config.add_mask, mask_model=Config.mask_model) as val_dataset:
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
        initial_epoch=0
    )
    # r = Config.model.evaluate_generator(val_dataset)
    # print(*zip(Config.model.metrics_names, r))

# y_pred = Config.model.predict(temp_x)[..., 0] > 0.5
# temp_y = temp_y[..., 0]
# mask = mask_model.predict(temp_x)
# temp_x = np.append(temp_x, mask, axis=-1)
# y_pred = Config.model.predict(temp_x)
# for i in range(temp_x.shape[0]):
#     plot_results(temp_x[i], temp_y[i], y_pred[i], "plots_mask_unet/{}".format(i), Config.onehot, Config.in_size)
