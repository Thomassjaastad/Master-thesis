import pandas as pd
import geopandas as gpd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import Grid        


# REMEMBER: x coord = latitude and y coord = longitude 

# specify shiptype 

dfTo = pd.read_csv('AISData/aisToTrondheim.csv')
ais_data = dfTo[dfTo['ShipType'].isin(['tugboats'])]

# retrieve heading and sog vals 
# and make v_lon and v_lat

heading = [(np.pi/180)*row for row in ais_data['Heading']] # in radians 
sog = [sog for sog in ais_data['SpeedOverGround']]

v_lon = sog*np.sin(heading) 
v_lat = sog*np.cos(heading)

"""--------------------------- Creating map boundaries and grid specs ------------------------------""" 

minlon = max(-180, min(ais_data['Longitude']))
minlat = max(-90, min(ais_data['Latitude']))
maxlon = min(180, max(ais_data['Longitude']))
maxlat = min(90, max(ais_data['Latitude']))

# specified for general grid 
nx, ny = 5, 5

lat0 = (maxlat+minlat)/2 
lon0 = (maxlon+minlon)/2
lat1 = (maxlat+minlat)/2-20

lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)

density, _, _ = np.histogram2d(ais_data['Latitude'], ais_data['Longitude'], bins=[lat_bins, lon_bins])

lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)

# convert to array from pandas dataframe

lon = np.asarray(ais_data['Longitude'])
lat = np.asarray(ais_data['Latitude'])

def grid(lat, lon):
    #####################################################
    # locate each data point in every gridcell          #
    # Counts each data points within each grid cell     #
    # returns:                                          #     
    # list with lat and lon values of each grid cell    #
    #####################################################     
    index_adder = []
    index_arr = []
    for j in range(nx-1):     #lon bins
        for k in range(ny-1): #lat bins
            count = 0
            for i in range(len(lon)):
                if lon[i] >= lon_bins[j] and lon[i] <= lon_bins[j + 1]:
                    if lat[i] >= lat_bins[k] and lat[i] <= lat_bins[k + 1]:
                        index_adder.append([i, j, k])
                        index_arr.append([i])
            #            count += 1
            #print(count)
    
    return np.asarray(index_adder), np.asarray(index_arr)

def cell_ind(density_mat):
    ######################################
    # getting indices as array to put    #
    # into lon and lat array for correct #
    # values in each grid cell           #
    # Returns:                           #
    # array with start and stop index    #
    # cell0 -> [start:stop]              # 
    ######################################

    density_mat = np.ravel(density_mat, order='F')
    added = 0
    cell_sorted = np.zeros(len(density_mat))
    for i in range(len(density_mat)):
        added += density_mat[i]
        cell_sorted[i] = added
    return cell_sorted

sorted_ind = cell_ind(density)
#exit()
#print(density)

def filter(heading, n):
    #################################################
    # filter function for removing boats            # 
    # going in opposite direction of wanted path    # 
    # Parameters:                                   #  
    # heading -> data from file. Check unit         #
    # nx number of longitude gridcells              # 
    # ny number of latitude gridcells               # 
    #                                               #
    # Returns:                                      #   
    # array with boats within a certain of heading  #
    #################################################
    heading = np.asarray(heading)
    filter_headingbins = np.linspace(np.min(heading), np.max(heading), n + 1)
    plt.hist(heading, bins=filter_headingbins, alpha=0.7, rwidth=0.85)
    plt.xlabel('Heading', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    return None
#filter(ais_data['Heading'], 8)


"""------------------------------------------------------------------------------------------------------"""
"""-------------------------------------------Creating map class-----------------------------------------"""



fig,ax=plt.subplots(figsize=(15,15))
m = Basemap(llcrnrlon=minlon, llcrnrlat=minlat, urcrnrlon=maxlon, urcrnrlat=maxlat,
            resolution='i', projection='cyl',lat_0=lat0, lon_0=lon0,lat_ts=lat1)


indices = grid(lat, lon)[1]
#first_cell = int(density[0,0])

m.quiver(lon[indices[:int(sorted_ind[0])]], lat[indices[:int(sorted_ind[0])]], 
         v_lon[indices[:int(sorted_ind[0])]], v_lat[indices[:int(sorted_ind[0])]])
#plt.show()
#exit()
# convert the xs and ys to map coordinates

xs, ys = m(lon_bins_2d, lat_bins_2d)

#plt.pcolormesh(xs, ys, density)
#plt.colorbar(orientation='vertical', label='Boat density')


# labels = [left,right,top,bottom]
m.drawparallels(lat_bins, labels=[False,True,True,False])
m.drawmeridians(lon_bins, labels=[True,False,True,False])

"""------------------------------------------------------------------------------------------------------"""

# formatting plot
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='lightgrey', lake_color='white')  

# Trondheim coords lat = y, lon = x
trond_lat = [63.43049]
trond_lon = [10.39506]
trond_x,trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

m.plot(trond_x, trond_y, color='black', marker='s', markersize=15)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label)
# ais data coords
x, y  = m(ais_data['Longitude'], ais_data['Latitude'])   

m.scatter(x, y, 0.7, marker='o', color='red', label= 'Tug boats')
#m.quiver(x[::2], y[::2], v_lon[::2], v_lat[::2], color='blue', label='Velocity')

plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.legend()
plt.show()
