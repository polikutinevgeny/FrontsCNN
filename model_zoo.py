from segmentation_models import FPN, Unet
from pspnet import PSPNet50
from deeplabv3plus import Deeplabv3
from biard_net import biard_net


def get_model(name, in_shape, n_classes, backend='resnet34'):
    if name == 'fpn':
        return FPN(backbone_name=backend, input_shape=in_shape, classes=n_classes, encoder_weights=None)
    if name == 'unet':
        return Unet(backbone_name=backend, input_shape=in_shape, classes=n_classes, encoder_weights=None)
    if name == 'pspnet':
        return PSPNet50(input_shape=in_shape, n_labels=n_classes)
    if name == 'deeplab':
        return Deeplabv3(input_shape=in_shape, classes=n_classes, weights=None)
    if name == 'biard':
        return biard_net(in_shape=in_shape, n_classes=n_classes)
    raise ValueError("Unknown model name")
