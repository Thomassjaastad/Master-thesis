import numpy as np

def Cartesian(lon, lat):
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

def Lonlat(x, y, z):
    """
    Convert back to longitude and latitude values

    lon and lat vals in radians now
    """
    R = 6371000
    lon = np.arctan2(y, x)
    lat = np.arcsin(z/R)
    return lon, lat


def WGS84(lat):
    """
    On the WGS84 spheroid, the length in meters of a degree of latitude 

    """
    x = 111412.84*np.cos(lat) - 93.5*np.cos(3*lat) + 0.118*np.cos(5*lat) 
    y = 111132.92 - 559.82*np.cos(2*lat) + 1.175*np.cos(4*lat) - 0.0023*np.cos(6*lat)
    return x, y 

def InverseWorldGeogeticsystem(x, y):
    """
    Not working properly
    """
    lat = y/111320
    lon = 360/(40075000*np.cos(lat))
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return lon, lat