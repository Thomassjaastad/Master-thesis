import pandas as pd
import geopandas as gpd
import numpy as np 
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import Grid  
import itertools      


"""
open csv file and use TimestampPosition column to make new columns. This may come in handy when wanting to get out certain data.
Seems as though the data and routes are structured in a timeseries. Try to extract routes from these new set up columns.
"""

# create dataframe
df = pd.read_csv('AISData/aisToTrondheim.csv')
#df = df.set_index('TimestampPosition')
df['TimestampPosition'] = pd.to_datetime(df['TimestampPosition'])

minlon = max(-180, min(df['Longitude']))
minlat = max(-90, min(df['Latitude']))
maxlon = min(180, max(df['Longitude']))
maxlat = min(90, max(df['Latitude']))

heading = [(np.pi/180)*row for row in df['Heading']] # convert to radians 
#sog = [sog for sog in df_subset['SpeedOverGround']]

# add columns on more specific dates: year, month, day, time  
df['Year'] = df['TimestampPosition'].dt.strftime('%Y')
df['Month'] = df['TimestampPosition'].dt.strftime('%m')
df['Day'] = df['TimestampPosition'].dt.strftime('%d')
#df['Hour'] = df['TimestampPosition'].dt.strftime('%H')
#df['Minute'] = df['TimestampPosition'].dt.strftime('%M')

# add columns with corresponding velocity components: longitude and latitude component
df['VeloLongitude'] = df['SpeedOverGround']*np.sin(heading)
df['VeloLatitude'] = df['SpeedOverGround']*np.cos(heading)

#specify ship type i.e. tugboats, cargoships for a subset of the data
df_subset = df[df['ShipType'].isin(['tugboats'])] 

"""----------------------------Create conditions on dataset------------------------------------"""

max_year = int(df['Year'].max())
min_year = int(df['Year'].min())

diff = max_year - min_year

num_years = np.linspace(min_year, max_year, diff + 1, dtype=int)
num_years = list(num_years)
num_mnths = ['01', '02', '03', '04', '05', '06', 
            '07', '08', '09', '10', '11', '12']

num_days = ['01', '02', '03', '04', '05', '06', 
            '07', '08', '09', '10', '11', '12',
            '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23', '24',
            '25', '26', '27', '28', '29', '30', '31']


def CreateRouteNumber(df_sorted, year, month, day):
    
    # df must be a sorted dataframe. Sorted by date 

    route_numb = 0
    route_lst = []    

    for years in year:
        for months in month:
            for days in day:
                routes = df_sorted.loc[(df_sorted['Year'] == str(years)) & 
                          (df_sorted['Month'] == months) & 
                          (df_sorted['Day'] == days)]
                
                if routes.empty == True:
                    continue
                else:
                    # create column ['Route'] in routes df_subset with route number
                    route_numb += 1
                    shpe = routes.shape
                    route_numb_col = [route_numb]*shpe[0]
                    routes['Route'] = route_numb_col
                    route_lst.append(route_numb_col)
                    #print(years, months, days, shpe[0])

        mergedRoutes = list(itertools.chain.from_iterable(route_lst))
    return mergedRoutes 


# sort values by date, needed for CreateRouteNumber func. Must be sorted 
df_subset['TimestampPosition'] = pd.to_datetime(df_subset['TimestampPosition'])
df_subsetsorted = df_subset.sort_values(by='TimestampPosition')
df_subsetsorted['Route'] = CreateRouteNumber(df_subsetsorted, num_years, num_mnths, num_days)

trond_lat = [63.43049]
trond_lon = [10.39506]


def CalcDistAbs(TrondLon, TrondLat, lon, lat):
    ########################################################
    # Parameters:                                          #
    # lon, lat vals Trondheim and lon, lat array           #
    # Returns:                                             #
    # Bool statement: True if route travels to Trondheim   #
    ########################################################

    # TODO: Do not need to find distance for all values, only first and last
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    
    lonT = lon - TrondLon
    latT = lat - TrondLat
    lonT = np.reshape(lonT, (len(lonT), 1))
    latT = np.reshape(latT, (len(latT), 1))
    DistArr = np.append(lonT, latT, axis = 1)
    #check = np.sqrt(lonT[0]**2 + latT[0]**2)
    Absolute_Dist = np.linalg.norm(DistArr, axis = 1)
    if Absolute_Dist[0] > Absolute_Dist[-1]:    
        print(Absolute_Dist[-1])
        return True
    else:
        #print('Travelling from Trondheim')
        return False



"""--------------------------- Creating map boundaries and grid specs ------------------------------""" 

# specified for general grid 
nx, ny = 3, 3

lat0 = (maxlat+minlat)/2 
lon0 = (maxlon+minlon)/2
lat1 = (maxlat+minlat)/2-20

lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)

density, _, _ = np.histogram2d(df['Latitude'], df['Longitude'], bins=[lat_bins, lon_bins])

lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)

"""------------------------------------------------------------------------------------------------------"""
"""-------------------------------------------Creating map class-----------------------------------------"""
    

fig,ax=plt.subplots(figsize=(15,15))
m = Basemap(llcrnrlon=minlon, llcrnrlat=minlat, urcrnrlon=maxlon, urcrnrlat=maxlat,
            resolution='i', projection='cyl',lat_0=lat0, lon_0=lon0,lat_ts=lat1)


# convert the xs and ys to map coordinates

#xs, ys = m(lon_bins_2d, lat_bins_2d)
#
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

#xTo, yTo = m(lon(ToTrond), lat[ToTrond])
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

# plotting Trondheim
m.plot(trond_x, trond_y, color='black', marker='s', markersize=15)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label)

# Find travel direction, start indexing from 1
cmap = mpl.cm.autumn

for i in range(df_subsetsorted['Route'].max()):
    route = df_subsetsorted.loc[(df_subsetsorted['Route']) == i + 1, 
                            ['Heading', 'Longitude', 'Latitude', 'VeloLongitude', 'VeloLatitude', 'TimestampPosition', 'Route']]
    dist = CalcDistAbs(trond_lon, trond_lat, 
                    route.iloc[:, 1], route.iloc[:, 2]) 
    if dist == True:
        # create vector field of all boats travelling to Trondheim
        if route['Route'].iloc[0] == 4:
            print(route)
            x, y = m(route.iloc[:, 1], route.iloc[:, 2])   
            u, v = m(route.iloc[:, 3], route.iloc[:, 4])
            m.quiver(x, y, u, v, color=cmap(i / df_subsetsorted['Route'].max()), label='Route %d' % (route['Route'].iloc[0]))

plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.legend()
plt.show()
