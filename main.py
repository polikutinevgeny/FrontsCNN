from keras.engine.saving import model_from_json
from keras.optimizers import *
import keras.utils
from tqdm import tqdm

from dataset import Dataset
from segmentation_models import FPN, PSPNet, Unet, Linknet
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score, f_score
import numpy as np
import scipy.stats as st
import keras.backend as K

from plot import plot_results, plot_conf_matrix, plot_fronts_far_east
from segnet import segnet
from keras_contrib.applications.densenet import DenseNetFCN
from pspnet import PSPNet50
from deeplabv3plus import Deeplabv3

from keras_contrib.losses import jaccard_distance
from keras.metrics import categorical_accuracy
from sklearn.metrics import confusion_matrix


def weighted_jaccard_loss(t, p):
    return jaccard_loss(t, p, class_weights=Config.class_weights)


def weighted_iou_score(t, p):
    return iou_score(t, p, class_weights=Config.class_weights)


def weighted_f_score(t, p):
    return f_score(t, p, class_weights=Config.class_weights)


def iou_metric_binary(t, p):
    p = K.cast(p > 0.5, K.floatx())
    return iou_score(t, p)


def iou_metric_all(t, p):
    return iou_score(t, K.one_hot(K.argmax(p, axis=-1), 5))


def iou_metric_fronts(t, p):
    return iou_score(t[..., 1:], K.one_hot(K.argmax(p, axis=-1), 5)[..., 1:])


def iou_metric_hot(t, p):
    return iou_score(t[..., 1:2], K.one_hot(K.argmax(p), 5)[..., 1:2])


def iou_metric_cold(t, p):
    return iou_score(t[..., 2:3], K.one_hot(K.argmax(p), 5)[..., 2:3])


def iou_metric_stationary(t, p):
    return iou_score(t[..., 3:4], K.one_hot(K.argmax(p), 5)[..., 3:4])


def iou_metric_occlusion(t, p):
    return iou_score(t[..., 4:5], K.one_hot(K.argmax(p), 5)[..., 4:5])


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def wcce(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return wcce


def mixed_loss_gen():
    wcce = weighted_categorical_crossentropy(Config.class_weights)

    def mixed_loss(y_true, y_pred):
        return wcce(y_true, y_pred) * 5 + jaccard_loss(y_true, y_pred)
    return mixed_loss


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
    # model = FPN(backbone_name="resnet34", input_shape=(None, None, 5), classes=5, encoder_weights=None)
    # model = PSPNet(backbone_name="resnext50", input_shape=(240, 336, 5), classes=5, activation='softmax', encoder_weights=None)
    # model = Unet(backbone_name="resnet34", encoder_weights=None, input_shape=(256, 256, 5))
    # model = Linknet(backbone_name="resnet18", input_shape=(256, 256, 5), classes=5, encoder_weights=None, activation="softmax")
    # model = segnet(input_shape=(256, 256, 5), n_labels=5)
    # model = DenseNetFCN(input_shape=(256, 256, 5), weights=None, classes=5, nb_layers_per_block=4, growth_rate=13, dropout_rate=0.2)
    # model = PSPNet50(input_shape=(256, 256, 5), n_labels=5)
    model = Deeplabv3(weights=None, input_shape=(256, 256, 5), classes=5)
    # model = None
    optimizer = Adam(lr=5e-4)
    logdir = "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_mixed_loss/"
    class_weights = class_weights
    # in_size = (240, 336)
    # in_size = (256, 320)
    in_size = (256, 256)
    # in_size = (255, 255)
    varnames = ["air", "mslet", "shum", "uwnd", "vwnd"]
    filename = "/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/narr.nc"
    truth_filename = "/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/plotted_fronts_fat.nc"
    batch_size = 16
    onehot = True
    regularizer = None
    add_mask = False
    mask_model = None
    augment = False


# Config.mask_model = keras.models.load_model("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Unet@inceptionv3_Fat_Binary/weights154-0.380389.hdf5",
#                                         custom_objects={
#                                             "weighted_jaccard_loss": weighted_jaccard_loss,
#                                             "weighted_iou_score": weighted_iou_score,
#                                             "weighted_f_score": weighted_f_score
#                                         })


# Config.model = keras.models.Sequential([
#     keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=(255 / 3, 255 / 3, 5), activation='relu'),
#     keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=(255 / 3, 255 / 3, 5), activation='relu'),
#     keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=(255 / 3, 255 / 3, 5), activation='relu'),
#     keras.layers.Conv2D(filters=5, kernel_size=(5, 5), padding="same", input_shape=(255 / 3, 255 / 3, 5), activation='softmax')
# ])

# set_regularization(Config.model, kern*el_regularizer=Config.regularizer, bias_regularizer=Config.regularizer)
# Config.model.load_weights("/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Deeplabv3plus/weights48-0.517574.hdf5")
Config.model.compile(
        optimizer=Config.optimizer,
        # loss=weighted_categorical_crossentropy(Config.class_weights),
        # loss=weighted_jaccard_loss,
        loss=mixed_loss_gen(),
        metrics=[
            # weighted_iou_score,
            # weighted_f_score,
            iou_metric_all,
            iou_metric_fronts,
            iou_metric_hot,
            iou_metric_cold,
            iou_metric_stationary,
            iou_metric_occlusion,
            # categorical_accuracy,
            # iou_metric_binary
        ]
    )

models = [
    ("deeplab", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Deeplabv3plus/weights101-0.504555.hdf5"),
    ("deeplab_cce", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_cce/weights153-0.138927.hdf5"),
    ("pspnet", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/PSPNet50/weights99-0.514912.hdf5"),
    ("pspnet_cce", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/PSPNet50_cce/weights97-0.181940.hdf5"),
    ("fpn", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_Fat_new_bounds_no_weights_correct/weights80-0.505150.hdf5"),
    ("fpn_cce", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/FPN@resnet34_cce/weights131-0.190181.hdf5"),
    ("simple", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/simple_4/weights186-0.641334.hdf5"),
    ("simple_cce", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/simple_cce/weights74-0.049018.hdf5")
]


def load_model(model):
    return keras.models.load_model(model, custom_objects={
        "weighted_jaccard_loss": weighted_jaccard_loss,
        "iou_metric_all": iou_metric_all,
        "iou_metric_fronts": iou_metric_fronts,
        "iou_metric_hot": iou_metric_hot,
        "iou_metric_cold": iou_metric_cold,
        "iou_metric_stationary": iou_metric_stationary,
        "iou_metric_occlusion": iou_metric_occlusion,
        "iou_metric_binary": iou_metric_binary,
        "weighted_iou_score": weighted_iou_score,
        "weighted_f_score": weighted_f_score,
        "loss": weighted_categorical_crossentropy(Config.class_weights),
        "wcce": weighted_categorical_crossentropy(Config.class_weights),
        "mixed_loss": mixed_loss_gen()
    })


Config.model = keras.models.load_model(
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_mask/weights55-0.277438.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab/weights175-0.811108.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_binary/Unet@inceptionv3_hot/weights56-0.744803.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_Fat_new_bounds_no_weights_correct/weights80-0.505150.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/FPN@resnet34/weights87-0.811470.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_cce/weights153-0.138927.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Deeplabv3plus/weights101-0.504555.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/PSPNet50/weights99-0.514912.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/simple_cce/weights74-0.049018.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/simple_4/weights186-0.641334.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@mobilenetv2_3/weights99-0.529462.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_binary/Unet@inceptionv3/weights133-0.388710.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_binary/Unet@resnet34/weights156-0.375369.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_crop_np/last.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/Linknet@Resnet18_cce/weights166-0.199727.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_crop_np_192/weights66-0.507229.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/FPN@resnet34_cce/weights131-0.190181.hdf5",
    # "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/PSPNet50_cce/weights97-0.181940.hdf5",
    "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_mixed_loss/weights73-1.337293.hdf5",
    custom_objects={
        "weighted_jaccard_loss": weighted_jaccard_loss,
        "iou_metric_all": iou_metric_all,
        "iou_metric_fronts": iou_metric_fronts,
        "iou_metric_hot": iou_metric_hot,
        "iou_metric_cold": iou_metric_cold,
        "iou_metric_stationary": iou_metric_stationary,
        "iou_metric_occlusion": iou_metric_occlusion,
        "iou_metric_binary": iou_metric_binary,
        "weighted_iou_score": weighted_iou_score,
        "weighted_f_score": weighted_f_score,
        "loss": weighted_categorical_crossentropy(Config.class_weights),
        "wcce": weighted_categorical_crossentropy(Config.class_weights),
        "mixed_loss": mixed_loss_gen()
    })
# Config.model.compile(
#         optimizer=Config.optimizer,
#         loss=weighted_jaccard_loss,
#         metrics=[
#             iou_metric_all,
#             iou_metric_fronts,
#             iou_metric_hot,
#             iou_metric_cold,
#             iou_metric_stationary,
#             iou_metric_occlusion,
#             categorical_accuracy,
#         ]
#     )
K.set_value(Config.model.optimizer.lr, 1e-4)

# Config.mask_model._make_predict_function()

indexing = np.load("indexing.npz")
train, val, test = indexing["train"], indexing["val"], indexing["test"]

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=Config.logdir + "weights{epoch:02d}-{val_loss:.6f}.hdf5", save_best_only=True, monitor="val_iou_metric_all", mode="max"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_iou_metric_all", factor=0.5, mode='max'),
    # keras.callbacks.EarlyStopping(patience=20),
    keras.callbacks.TensorBoard(log_dir=Config.logdir),
    keras.callbacks.ModelCheckpoint(filepath=Config.logdir + "last.hdf5"),
    # TQDMCallback(leave_inner=False)
]
with Dataset(train, Config.filename, Config.varnames, Config.truth_filename, Config.batch_size,
            Config.in_size, onehot=Config.onehot, add_mask=Config.add_mask, mask_model=Config.mask_model, augment=Config.augment) as train_dataset, \
    Dataset(val, Config.filename, Config.varnames, Config.truth_filename, Config.batch_size,
            Config.in_size, onehot=Config.onehot, add_mask=Config.add_mask, mask_model=Config.mask_model, augment=False) as val_dataset, \
    Dataset(test, Config.filename, Config.varnames, Config.truth_filename, Config.batch_size,
            Config.in_size, onehot=Config.onehot, add_mask=Config.add_mask, mask_model=Config.mask_model, augment=False) as test_dataset:
    # cm_list = []
    # for i in tqdm(range(len(test_dataset))):
    #     x, y_true = test_dataset[i]
    #     y_pred = Config.model.predict(x)
    #     cm = confusion_matrix(np.argmax(y_true[0], axis=-1).flatten(), np.argmax(y_pred[0], axis=-1).flatten(), labels=[0, 1, 2, 3, 4]).astype('float')
    #     sums = cm.sum(axis=1)[:, np.newaxis]
    #     for j in range(cm.shape[0]):
    #         if sums[j].sum() == 0:
    #             cm[j] = np.zeros(5)
    #             cm[j, j] = 1
    #         else:
    #             cm[j] /= sums[j]
    #     cm_list.append(cm)
    # cm_list = np.array(cm_list)
    # np.save("cm_list", cm_list)
    # cm_list = np.load("cm_list.npy")
    # np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    # print(np.percentile(cm_list, 25, axis=0), np.percentile(cm_list, 50, axis=0), np.percentile(cm_list, 75, axis=0), sep='\n')
    # n = cm_list.shape[0]
    # m, se = cm_list.mean(axis=0), st.sem(cm_list, axis=0)
    # h = se * st.t.ppf((1 + 0.99) / 2, n - 1)
    # print(m, h, sep='\n')
    # np.savetxt("confu.csv", m, delimiter=',')
    # np.savetxt("confi.csv", h, delimiter=',')

    # for m, i in zip(tqdm(test_dataset), range(len(test_dataset))):
    #     x, y = m
    #     r = Config.model.evaluate(x, y, verbose=0)
    #     d = test_dataset.get_dates(i)[0]
    #     if r[1] < 0.3:
    #         pred = Config.model.predict(x)
    #         plot_results(x[0], y[0], pred[0], "plots/plots_deeplab_bad_drom/{0}_{1:.2f}.png".format(i, r[1]), Config.onehot, Config.in_size, d)
    # d = [[] for _ in Config.model.metrics_names]
    # for (x, y) in tqdm(train_dataset):
    #     r = Config.model.evaluate(x, y, verbose=0)
    #     for i, j in zip(d, r):
    #         i.append(j)
    # import matplotlib.pyplot as plt
    # for n, i in zip(Config.model.metrics_names, d):
    #     plt.hist(i, bins=100)
    #     plt.title(n)
    #     plt.savefig("metric_hist_deeplab_train/{}".format(n))
    #     plt.show()
    temp_x = np.array([test_dataset[i][0][0] for i in range(len(test_dataset))])
    temp_y = np.array([test_dataset[i][1][0] for i in range(len(test_dataset))])
    # temp_x, temp_y = test_dataset[0]
    # dates = test_dataset.get_dates(0)
    # Config.model.fit_generator(
    #     generator=train_dataset,
    #     validation_data=val_dataset,
    #     use_multiprocessing=False,
    #     verbose=1,
    #
    #     callbacks=callbacks,
    #     shuffle=False,
    #     workers=0,
    #     epochs=1000,
    #     initial_epoch=70
    # )
    # r = Config.model.evaluate_generator(train_dataset, workers=0, use_multiprocessing=False, verbose=1)
    # print("train", *zip(Config.model.metrics_names, r))
    # r = Config.model.evaluate_generator(val_dataset, workers=0, use_multiprocessing=False, verbose=1)
    # print("val", *zip(Config.model.metrics_names, r))
    # r = Config.model.evaluate_generator(test_dataset, workers=0, use_multiprocessing=False, verbose=1)
    # print("test", *zip(Config.model.metrics_names, r))

# y_pred = Config.model.predict(temp_x)[..., 0] > 0.5
# temp_y = temp_y[..., 0]
# mask = mask_model.predict(temp_x)
# temp_x = np.append(temp_x, mask, axis=-1)
# for n, m in models[6:]:
#     Config.model = load_model(m)
#     y_pred = Config.model.predict(temp_x, batch_size=1)
#     for i in range(temp_x.shape[0]):
#         plot_results(temp_x[i], temp_y[i], y_pred[i], "plots/plots_drom_all/{}{}".format(n, i), Config.onehot, Config.in_size, dates[i])
# y_pred = Config.model.predict(temp_x, batch_size=1)
# for i in range(temp_x.shape[0]):
#     plot_results(temp_x[i], temp_y[i], y_pred[i], "plots/plots_deeplab_mixed/{}".format(i), Config.onehot, Config.in_size, dates[i])

# y_pred = Config.model.predict(temp_x, batch_size=1).flatten()
# temp_y = temp_y[..., 0].flatten()
y_pred = np.argmax(Config.model.predict(temp_x, batch_size=1), axis=-1).flatten()
temp_y = np.argmax(temp_y, axis=-1).flatten()
del temp_x
import gc
gc.collect()
print("passed")
# print("{},{},{}".format(temp_y.size, np.count_nonzero(temp_y), np.count_nonzero(y_pred)))
# from confusion_matrix import confusion_matrix
# cm = confusion_matrix(temp_y, y_pred)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# # Balanced accuracy
# BA = (TPR + TNR) / 2
# # results = {"TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV, "FPR": FPR, "FNR": FNR, "FDR": FDR, "ACC": ACC, "BA": BA}
# results = {"FPR": FPR, "FNR": FNR}
# for k, v in results.items():
#     print("{}: {}".format(k, v))
plot_conf_matrix(temp_y, y_pred, normalize=True)

# from sklearn.metrics import precision_recall_curve, roc_curve
# import matplotlib.pyplot as plt
# from sklearn.utils.fixes import signature
# precision, recall, _ = precision_recall_curve(temp_y, y_pred)
#
# step_kwargs = ({'step': 'post'}
#                if 'step' in signature(plt.fill_between).parameters
#                else {})
# plt.step(recall, precision, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve')
# plt.show()


# # Far East
# fe_filename = '/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/era5_lcc.nc'
# import xarray as xr
# with xr.open_dataset(fe_filename) as f:
#     dates = f.time.values
# with Dataset(dates, fe_filename, Config.varnames, Config.truth_filename, Config.batch_size,
#              Config.in_size, onehot=Config.onehot, add_mask=Config.add_mask,
#              mask_model=Config.mask_model, augment=Config.augment, return_truth=False) as d:
#     x = d[0]
#     dates = d.get_dates(0)
#     result = Config.model.predict(x)
#     for i in range(result.shape[0]):
#         plot_fronts_far_east(x[i], result[i], "plots/plots_far_east_drom/deeplab_w{}.png".format(i), Config.onehot,
#                              Config.in_size, dates[i])
#         # plot_fronts_far_east(x[i], result[i], "plots/plots_far_east_2019_deeplab/{}".format(i), Config.onehot, Config.in_size, dates[i])
#         # plot_results(x[i], result[i], result[i], "plots/plots_far_east/{}".format(i), Config.onehot, Config.in_size, dates[i])
#         # import matplotlib.pyplot as plt
#         # plt.imshow(np.argmax(result[i], axis=-1))
#         # plt.show()
