import numpy as np

def Convert_Cart(lon, lat):
    """
    Converts longitude and latitude values to a cartesian 
    coordinate system:

    Remember to check if lon and lat values are in radians and not degrees
    """
    R = 6371000 # earth radius in [m]
    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    z = R*np.sin(lat)
    return x, y, z

def Convert_Lon_Lat(x, y, z):
    """
    Convert back to longitude and latitude values

    lon and lat vals in radians now
    """
    R = 6371000
    lon = np.arctan2(y, x)
    lat = np.arcsin(z, R)
    return lon, lat
