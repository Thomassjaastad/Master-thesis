import numpy as np
from mpl_toolkits.basemap import Basemap


def arr(x, y, BaseMapobj):
    """
    Args: 
    x: array typically from meshgrid 
    y: array typically from meshgrid 
     
    Returns:
    1 if point is over land
    0 if point is over sea
    """
    bool_arr = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            bool_arr[i][j] = BaseMapobj.is_land(x[i], y[j])
    return bool_arr

def points(x, y, BaseMapobj):
    """
    Args: 
    x: array typically from meshgrid 
    y: array typically from meshgrid 
     
    Returns:
    1 if point is over land
    0 if point is over sea
    """
    bool_arr = np.zeros(len(x))
    for i in range(len(x)):
        bool_arr[i] = BaseMapobj.is_land(x[i], y[i])
    return bool_arr