from keras import backend as K
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score

from main import Config


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