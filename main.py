from keras.optimizers import *
import keras.utils

from dataset import Dataset
import numpy as np

from metrics import iou_metric_all, iou_metric_fronts, iou_metric_hot, iou_metric_cold, \
    iou_metric_stationary, iou_metric_occlusion, mixed_loss_gen
from plot import plot_conf_matrix
from deeplabv3plus import Deeplabv3

from utils import load_indexing, class_weights
from config import Config

config = Config(
    # model = FPN(backbone_name="resnet34", input_shape=(None, None, 5), classes=5, encoder_weights=None)
    # model = Unet(backbone_name="resnet34", encoder_weights=None, input_shape=(256, 256, 5))
    # model = Linknet(backbone_name="resnet18", input_shape=(256, 256, 5), classes=5, encoder_weights=None, activation="softmax")
    # model = PSPNet50(input_shape=(256, 256, 5), n_labels=5)
    model=Deeplabv3(weights=None, input_shape=(256, 256, 5), classes=5),
    optimizer=Adam(lr=5e-4),
    logdir="/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_mixed_loss/",
    class_weights=class_weights,
    in_shape=(256, 256),
    n_classes=5,
    varnames=["air", "mslet", "shum", "uwnd", "vwnd"],
    filename="/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/narr.nc",
    truth_filename="/mnt/d4dca524-e11f-4923-8fbe-6066e6efd2fd/NARR/plotted_fronts_fat.nc",
    batch_size=16,
    binary=True,
    regularizer=None,
    augment=False,
    metrics=[
        iou_metric_all,
        iou_metric_fronts,
        iou_metric_hot,
        iou_metric_cold,
        iou_metric_stationary,
        iou_metric_occlusion,
        # iou_metric_binary
    ],
    # loss=weighted_categorical_crossentropy(Config.class_weights),
    # loss=weighted_jaccard_loss,
    loss=mixed_loss_gen()
)

train, val, test = load_indexing("indexing.npz")


with Dataset(train, config) as train_dataset, \
        Dataset(val, config) as val_dataset, \
        Dataset(test, config) as test_dataset:
    pass
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
    # temp_x = np.array([test_dataset[i][0][0] for i in range(len(test_dataset))])
    # temp_y = np.array([test_dataset[i][1][0] for i in range(len(test_dataset))])
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
# y_pred = np.argmax(Config.model.predict(temp_x, batch_size=1), axis=-1).flatten()
# temp_y = np.argmax(temp_y, axis=-1).flatten()
# del temp_x
# import gc
#
# gc.collect()
# print("passed")
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
# plot_conf_matrix(temp_y, y_pred, normalize=True)
