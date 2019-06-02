from model import Model
import os.path
from model_zoo import get_model


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if getattr(self, 'model', False):
            self.model = self.init_model()

    def init_model(self):
        if isinstance(self.model, str):
            if os.path.isfile(self.model):
                return Model(
                    file=self.model,
                    binary=self.binary,
                    regularizers=self.regularizers,
                    class_weights=self.class_weights,
                    recompile=getattr(self, 'recompile', False),
                    optimizer=getattr(self, 'optimizer', None),
                    loss=getattr(self, 'loss', None),
                    metrics=getattr(self, 'metrics', None)
                )
            else:
                self.model = get_model(
                    name=self.model,
                    in_shape=self.in_shape,
                    n_classes=self.n_classes,
                    backend=self.backend
                )
        return Model(
            keras_model=self.model,
            binary=self.binary,
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            regularizers=self.regularizers,
            class_weights=self.class_weights
        )