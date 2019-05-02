import keras.models


def biard_net(in_shape, n_classes):
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=in_shape, activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=in_shape, activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding="same", input_shape=in_shape, activation='relu'),
        keras.layers.Conv2D(filters=n_classes, kernel_size=(5, 5), padding="same", input_shape=in_shape, activation='softmax')
    ])
