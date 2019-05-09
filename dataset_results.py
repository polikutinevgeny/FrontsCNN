import gc
import numpy as np


def dataset_results(dataset, model, binary=False):
    x = np.array([dataset[i][0][0] for i in range(len(dataset))])
    y_true = np.array([dataset[i][1][0] for i in range(len(dataset))])
    if binary:
        y_pred = model.predict(x, batch_size=1).flatten() > 0.5
        y_true = y_true[..., 0].flatten()
    else:
        y_pred = np.argmax(model.predict(x, batch_size=1), axis=-1).flatten()
        y_true = np.argmax(y_true, axis=-1).flatten()
    del x
    gc.collect()
    return y_true, y_pred
