import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import sourcepanel
import sourcepoint
import FlowBoundary
import Flow
#import find_land
import sys
import convert
#from scipy import linalg

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('Routing/aisToTrondheim.csv', index_col = 'TimestampPosition', parse_dates = True)
df = df.drop(columns=['dummy'])

trond_lat = [63.43049]
trond_lon = [10.39506]

# add columns to df
df['Heading'] = np.radians(df['Heading'])
df['VelocityLongitude'] = df['SpeedOverGround']*np.sin(df['Heading'])
df['VelocityLatitude'] = df['SpeedOverGround']*np.cos(df['Heading'])
df['DistTrondheimHaversine'] = convert.distsphere(df['Longitude'], trond_lon, df['Latitude'], trond_lat)

print('Ship types in data set:')
print(df['ShipType'].drop_duplicates())

# extract subset of original df to analyze. Dataset to big to analyze all together
df_subset = df[df['ShipType'].isin(['tugboats'])].copy()

# get rid of hours, minutes and seconds in Timestamp col/idx
date = df_subset.index.to_period('d')
unique_dates = date.drop_duplicates()

date_freq = date.value_counts()
# find routes sorting by unique dates. Add new column to df_subset determining travelling direction
for i in range(unique_dates.shape[0]):
    route = df_subset.loc[str(unique_dates[i])]
    last_dist = route['DistTrondheimHaversine'][-1]
    first_dist = route['DistTrondheimHaversine'][0]
    df_subset.loc[route.index, 'Route number'] = i
    #df_subset.loc[i, 'route number']
    if last_dist < first_dist:
        df_subset.loc[route.index, 'Direction'] = 'To'  
        count = 0

        # check route going to if starting to deviate from dest.
        for j in range(route.shape[0] - 1):
            next_dist = route['DistTrondheimHaversine'][j + 1]
            prev_dist = route['DistTrondheimHaversine'][j]
            if next_dist < prev_dist:
                #dist decreasing
                count = 0
            else:
                #dist increasing
                count += 1
            if count == 5:
                #Ship could be travelling through
                df_subset.loc[route.index, 'Direction'] = 'Through'
                break
    else:
        df_subset.loc[route.index, 'Direction'] = 'From'

df_ToTrondheim = df_subset.loc[df_subset['Direction'] == 'To', 
['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheimHaversine', 'Route number']]


unique_tugroutes =df_subset['Route number'].drop_duplicates()
unique_routes = df_ToTrondheim['Route number'].drop_duplicates()
print(unique_routes)

df_route = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == 7.0, 
        ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheimHaversine', 'Route number']]


# grid boundaries
minlon = max(-180, min(df_ToTrondheim['Longitude']))
minlat = max(-90, min(df_ToTrondheim['Latitude']))
maxlon = min(180, max(df_ToTrondheim['Longitude']))
maxlat = min(90, max(df_ToTrondheim['Latitude']))

nx, ny = 3, 3

lat0 = (maxlat + minlat)/2 
lon0 = (maxlon + minlon)/2
lat1 = (maxlat + minlat)/2-20
lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)
fig, ax = plt.subplots(figsize = (30, 30))

lat_trond = 63.43049
lon_trond = 10.39506

# insert area threshold for more detailed map plot
eps = 0.01

# use merc or cyl projection types!
# create map instance
m = Basemap(llcrnrlon = minlon, llcrnrlat = minlat, urcrnrlon = maxlon, urcrnrlat = maxlat,
            resolution = 'i', projection = 'cyl', lon_0 = lon0, lat_0 = lat0)

# formatting plot
m.drawmapboundary(fill_color = 'lightblue')
m.fillcontinents(color = 'green', lake_color = 'lightblue', zorder=1)  

# create color plot for unique routes 
for i in range(len(unique_tugroutes.values)):
    routes = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == i, 
        ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude']]

    m.quiver(routes['Longitude'], routes['Latitude'], 
             routes['VelocityLongitude'], routes['VelocityLatitude'], 
             scale = 700, color='C%d' % i)


#  labels = [left,right,top,bottom]
m.drawparallels(lat_bins, labels = [False, True, True, False])
m.drawmeridians(lon_bins, labels = [True, False, True, False])

# Trondheim
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

m.plot(trond_x, trond_y, color = 'orange', marker = 's', markersize = 4)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label, color='orange')

plt.xlabel('Longitude', fontsize = 18)
plt.ylabel('Latitude', fontsize = 18)
plt.legend()
plt.show()
