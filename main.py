from keras.optimizers import *

from dataset import Dataset

from metrics import iou_metric_all, iou_metric_fronts, iou_metric_hot, iou_metric_cold, \
    iou_metric_stationary, iou_metric_occlusion, mixed_loss_gen
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
