import pandas as pd
#import geopandas as gpd
import numpy as np 
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
#import Grid  
import itertools      

"""
open csv file and use TimestampPosition column to make new columns. This may come in handy when wanting to get out certain data.
Seems as though the data and routes are structured in a timeseries. Try to extract routes from these new set up columns.
"""

# create dataframe
df = pd.read_csv('aisToTrondheim.csv')
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

def create_routes(df_sorted, years, months, days, IMONumbers):
    ###############################################################
    # Create routes from Timestamp column by checking time log    # 
    # Parameters:                                                 #    
    # sorted dataframe w.r.t Timestamp, number of years,          #    
    # months and days in Timestamp data                           #   
    # Returns:                                                    #     
    # pandas.series with route numbers for each timestamp data    # 
    ###############################################################
    route_numb = 0
    route_lst = []    
    for IMOS in IMONumbers:
        for year in years:
            for month in months:
                for day in days:
                    routes = df_sorted.loc[(df_sorted['IMONumber'] == IMOS) &
                               (df_sorted['Year'] == str(year)) & 
                              (df_sorted['Month'] == month) & 
                              (df_sorted['Day'] == day)]

                    if routes.empty == True:
                        continue
                    else:
                        # create column ['Route'] in routes df_subset with route number
                        route_numb += 1
                        shpe = routes.shape
                        route_numb_col = [route_numb]*shpe[0]
                        routes['Route'] = route_numb_col
                        route_lst.append(route_numb_col)
                        #print(year, months, days, shpe[0])

        mergedRoutes = list(itertools.chain.from_iterable(route_lst))
    return mergedRoutes 

def calc_dist(TrondLon, TrondLat, lon, lat):
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
    return Absolute_Dist[0], Absolute_Dist[-1]



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
    

fig,ax=plt.subplots(figsize=(15,30))
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
trond_lat = [63.43049]
trond_lon = [10.39506]

#xTo, yTo = m(lon(ToTrond), lat[ToTrond])
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

# plotting Trondheim
m.plot(trond_x, trond_y, color='black', marker='s', markersize=15)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label)

# Find travel direction, start indexing from 1
cmap = mpl.cm.autumn

# minimun distance to Trondheim
eps = 0.1

# specify ship type i.e. tugboats, cargoships, passenger_ships, other, high_speed_crafts, tankships for a subset of the data
df_subset = df[df['ShipType'].isin(['tugboats'])] 

# sort values by date: TimestampPosition. Must be sorted 
df_subset['TimestampPosition'] = pd.to_datetime(df_subset['TimestampPosition'])
df_subsetsorted = df_subset.sort_values(by='TimestampPosition')
IMOS = df_subsetsorted['IMONumber'].drop_duplicates() 
df_subsetsorted['Route'] = create_routes(df_subsetsorted, num_years, num_mnths, num_days, IMOS)

for i in range(df_subsetsorted['Route'].max()):
    route_info = df_subsetsorted.loc[(df_subsetsorted['Route']) == i + 1, 
            ['IMONumber', 'Heading', 'Longitude', 'Latitude', 'VeloLongitude', 'VeloLatitude', 'TimestampPosition', 'Route']]
    #print(route_info.head())
    start_to = calc_dist(trond_lon, trond_lat, route_info.iloc[:, 2], route_info.iloc[:, 3])[0]
    end_to = calc_dist(trond_lon, trond_lat, route_info.iloc[:, 2], route_info.iloc[:, 3])[1]

    if start_to > end_to and end_to < eps:
        # travelling to Trondheim
        print("final distance of route_info %d is: %.3f with data points %s"  % (i+1, end_to, route_info['Route'].shape))
        Q = m.quiver(route_info.iloc[:, 2], route_info.iloc[:, 3], 
                     route_info.iloc[:, 4], route_info.iloc[:, 5], 
                     color=cmap(i / df_subsetsorted['Route'].max()), 
                     label='Route %d' % (route_info['Route'].iloc[0]))
            
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.legend()
plt.show()
