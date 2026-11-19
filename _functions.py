import numpy as np

def pooled_std(x : np.ndarray, y : np.ndarray) -> float:
    var_x = np.var(x)
    var_y = np.var(y)
    n_x = len(x)
    n_y = len(y)

    return np.sqrt(
        ((n_x - 1) * var_x + (n_y - 1) * var_y)
        /
        (n_x + n_y - 2)
    )

def cohen_d(x : np.ndarray, y : np.ndarray) -> float:
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    s = pooled_std(x, y)

    return (mean_x - mean_y) / s