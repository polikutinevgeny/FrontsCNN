import keras.models
import numpy as np
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import f_score, jaccard_score

from metrics import iou_metric_all, iou_metric_fronts, iou_metric_hot, iou_metric_cold, iou_metric_stationary, \
    iou_metric_occlusion, iou_metric_binary, weighted_categorical_crossentropy, mixed_loss_gen


def load_indexing(path):
    indexing = np.load(path)
    return indexing["train"], indexing["val"], indexing["test"]


models = [
    ("deeplab", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/Deeplabv3plus/weights101-0.504555.hdf5"),
    ("deeplab_cce",
     "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_cce/weights153-0.138927.hdf5"),
    ("deeplab_mixed",
     "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/deeplab_mixed_loss/weights73-1.337293.hdf5"),
    ("pspnet", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/PSPNet50/weights99-0.514912.hdf5"),
    ("pspnet_cce",
     "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/PSPNet50_cce/weights97-0.181940.hdf5"),
    ("fpn",
     "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/FPN@resnet34_Fat_new_bounds_no_weights_correct/weights80-0.505150.hdf5"),
    ("fpn_cce",
     "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/FPN@resnet34_cce/weights131-0.190181.hdf5"),
    ("simple", "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs/simple_4/weights186-0.641334.hdf5"),
    ("simple_cce",
     "/mnt/ldm_vol_DESKTOP-DSIGH25-Dg0_Volume1/DiplomLogs/logs_weighted/simple_cce/weights74-0.049018.hdf5")
]

trained_models = dict(models)


def load_model(model, class_weights=1):
    return keras.models.load_model(model, custom_objects={
        "weighted_jaccard_loss": jaccard_loss,
        "iou_metric_all": iou_metric_all,
        "iou_metric_fronts": iou_metric_fronts,
        "iou_metric_hot": iou_metric_hot,
        "iou_metric_cold": iou_metric_cold,
        "iou_metric_stationary": iou_metric_stationary,
        "iou_metric_occlusion": iou_metric_occlusion,
        "iou_metric_binary": iou_metric_binary,
        "weighted_iou_score": jaccard_score,
        "weighted_f_score": f_score,
        "loss": weighted_categorical_crossentropy(class_weights),
        "wcce": weighted_categorical_crossentropy(class_weights),
        "mixed_loss": mixed_loss_gen(class_weights)
    })


binary_weights = 1
class_weights = [
    0.008277938265164113,
    0.9239984235226252,
    0.3345056364872783,
    0.3148883509015734,
    1.0
]