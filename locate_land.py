import numpy as np

def find_land(x, y):
    """
    Returns:
    1 if point is over land
    0 if point is over sea
    """
    bool_arr = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            bool_arr[i][j] = m.is_land(x[i], y[j])
    return bool_arr