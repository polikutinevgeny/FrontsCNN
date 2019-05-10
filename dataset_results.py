import gc
import numpy as np


def dataset_results(dataset, model, binary=False):
    x = np.array([dataset[i][0][0] for i in range(len(dataset))])
    y_true = np.array([dataset[i][1][0] for i in range(len(dataset))])
    y_pred = model.predict(x, batch_size=1, verbose=0).flatten()
    if binary:
        y_true = y_true[..., 0].flatten()
    else:
        y_true = np.argmax(y_true, axis=-1).flatten()
    del x
    gc.collect()
    return y_true, y_pred
