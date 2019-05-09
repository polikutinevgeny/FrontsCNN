from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.engine.saving import model_from_json
from utils import load_model


class Model:
    def __init__(self, file=None, keras_model=None, optimizer=None, loss=None, metrics=None, binary=False, regularizers=None, class_weights=1, recompile=False):
        if file is not None:
            self.keras_model = load_model(file, class_weights)
            if recompile:
                self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif keras_model is not None:
            self.keras_model = keras_model
            if self.keras_model.optimizer is None:
                self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        else:
            raise ValueError("No keras model supplied")
        self.binary = binary
        if regularizers:
            self.set_regularization(*regularizers)

    def set_regularization(self,
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None):
        for layer in self.keras_model.layers:
            if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = kernel_regularizer
            if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
                layer.bias_regularizer = bias_regularizer
            if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
                layer.activity_regularizer = activity_regularizer
        out = model_from_json(model.to_json())
        out.set_weights(self.keras_model.get_weights())
        self.keras_model = out

    def train(self, log_dir, train, val, monitor, initial_epoch=0, epochs=300):
        callbacks = [
            ModelCheckpoint(filepath=log_dir + "weights{epoch:02d}.hdf5",
                            save_best_only=True, monitor=monitor, mode="max"),
            ReduceLROnPlateau(monitor=monitor, factor=0.5, mode='max'),
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(filepath=log_dir + "last.hdf5"),
        ]
        self.keras_model.fit_generator(
            generator=train,
            validation_data=val,
            use_multiprocessing=False,
            verbose=1,
            callbacks=callbacks,
            shuffle=False,
            workers=0,
            epochs=epochs,
            initial_epoch=initial_epoch
        )

    def evaluate(self, data, **kwargs):
        return self.keras_model.evaluate_generator(data, workers=0, use_multiprocessing=False, **kwargs)

    def predict(self, data, **kwargs):
        result = self.keras_model.predict(data, workers=0, use_multiprocessing=False, **kwargs)
        if self.binary:
            return result[..., 0] > 0.5
        else:
            return result.argmax(axis=-1)
