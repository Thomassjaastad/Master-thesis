import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import sourcepanel
import sourcepoint
import FlowBoundary
import Flow
import sys
import convert
from scipy import linalg

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('Routing/aisToTrondheim.csv', index_col = 'TimestampPosition', parse_dates = True)
df = df.drop(columns=['dummy'])

trond_lat = 63.43049
trond_lon = 10.39506

# add columns to df
df['Heading'] = np.radians(df['Heading'])
df['VelocityLongitude'] = df['SpeedOverGround']*np.sin(df['Heading'])
df['VelocityLatitude'] = df['SpeedOverGround']*np.cos(df['Heading'])
df['DistTrondheimHaversine'] = convert.distsphere(df['Longitude'], trond_lon, df['Latitude'], trond_lat)

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


# Find all vessels travelling toward Trondheim. 
df_ToTrondheim = df_subset.loc[df_subset['Direction'] == 'To', 
['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheimHaversine', 'Route number']]

# single route from tugboat df

unique_routes = df_ToTrondheim['Route number'].drop_duplicates()
print(unique_routes)

df_route = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == 13.0, 
        ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheimHaversine', 'Route number']]


# insert area threshold for more detailed map plot

eps = 0.01

# grid boundaries
minlon = trond_lon - eps
minlat = trond_lat - eps
#minlon = max(-180, min(df_route['Longitude']))
#minlat = max(-90, min(df_route['Latitude']))
maxlon = min(180, max(df_route['Longitude']))
maxlat = min(90, max(df_route['Latitude']))

nx, ny = 3, 3
 
lat0 = (maxlat + minlat)/2 
lon0 = (maxlon + minlon)/2
lat1 = (maxlat + minlat)/2-20
lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)
fig, ax = plt.subplots(figsize = (30, 30))

lat_trond = 63.43049
lon_trond = 10.39506

# use merc or cyl projection types!
# create map instance
m = Basemap(llcrnrlon = minlon, llcrnrlat = minlat, urcrnrlon = maxlon, urcrnrlat = maxlat,
            resolution = 'i', projection = 'cyl', lon_0 = lon0, lat_0 = lat0)
 
# formatting plot
m.drawmapboundary(fill_color = 'lightblue')
m.fillcontinents(color = 'green', lake_color = 'lightblue', zorder=1)  
# coastline and panels values XB and YB

# maybe change coordinate orientation?!?!
coast = m.drawcoastlines()
coordinates = coast.get_segments() 
coordinates = np.vstack(coordinates)

# boundary points in lon and lat degree values!
XB = coordinates[:, 0]
YB = coordinates[:, 1]

XC, YC = FlowBoundary.control_points(XB,YB)
#TODO: check delta angle. Not used in program! Necessary?
# tweakable parameters!

Vinf = -1
AoA = np.arctan((trond_lat -  df_route['Latitude'][0])/(trond_lon - df_route['Longitude'][0]))
panels, phi = FlowBoundary.panels(XB, YB)
delta, beta = FlowBoundary.angles(phi, AoA)

# solving system of equations
I = sourcepanel.solveGeometricIntegrals(XC, YC, panels, phi, beta, XB, YB)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta)
lamda = linalg.solve(I, b)

# grid for flow computations
nxx, nyy = 100, 100

# lon and lat rep
x_flow = np.linspace(minlon, maxlon, nxx)
y_flow = np.linspace(minlat, maxlat, nyy)
X, Y = np.meshgrid(x_flow, y_flow)

# sink located at Trondheim
lamda_trond = 0.1        # sink strength
trond_flow = Flow.Velocity(lamda_trond, trond_lon, trond_lat)
vx_sink, vy_sink = trond_flow.sink(X, Y)

vx = np.zeros((nxx, nyy))
vy = np.zeros((nxx, nyy))

# compute velocites at every grid cell (XP, YP)
for i in range(nxx):
    for j in range(nyy):
        XGeom, YGeom = sourcepoint.solvePointGeometry(X[i,j], Y[i,j], panels, phi, XB, YB)
        if m.is_land(X[i,j], Y[i,j]) == True:
            vx[i, j] = 0 
            vy[i, j] = 0
        else:
            vx[i, j] = Vinf*np.cos(AoA) + np.dot(lamda, XGeom.T/(2*np.pi)) + vx_sink[i, j]  #+ np.sum(lamda*XGeom/(2*np.pi)) #np.dot(lamda, XGeom.T)/(2*np.pi) + vx_sink[i, j] #+ vx_source2[i,j] + vx_source1[i, j]  
            vy[i, j] = Vinf*np.sin(AoA) + np.dot(lamda, YGeom.T/(2*np.pi)) + vy_sink[i, j]  #+ np.sum(lamda*YGeom/(2*np.pi)) #  #+ vy_source2[i,j] + vy_source1[i, j]  

#m.quiver(df['Longitude'], df['Latitude'], 
#         df['VelocityLongitude'], df['VelocityLatitude'], color = 'blue', scale = 2000, label = 'AIS-data')

# all tug boats
#m.quiver(df_subset['Longitude'], df_subset['Latitude'], 
#         df_subset['VelocityLongitude'], df_subset['VelocityLatitude'], 
#         scale=1100, color='C0')

# single route
m.quiver(df_route['Longitude'], df_route['Latitude'],
         df_route['VelocityLongitude'], df_route['VelocityLatitude'], scale = 800)


# starting points for streamlines 
# plt function not good at plotting points 
stream = plt.streamplot(X[1:-1, 1:-1], Y[1:-1, 1:-1], vx[1:-1, 1:-1], vy[1:-1, 1:-1],
                linewidth = 0.75, density = 4, color = 'C0', arrowstyle = '->')

#  labels = [left,right,top,bottom]
m.drawparallels(lat_bins, labels = [False, True, True, False])
m.drawmeridians(lon_bins, labels = [True, False, True, False])

trond_lat = [63.43049]
trond_lon = [10.39506]


# Trondheim
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

# get streamline points
path = stream.lines.get_paths()
path = np.asarray(path)
segment = stream.lines.get_segments()
segment = np.asarray(segment)
val = 10

# finding starting point of streamline
for i in range(segment.shape[0]):    
    dist = np.sqrt((df_route['Longitude'][0] - segment[i, 0, 0])**2 + (df_route['Latitude'][0] - segment[i, 1, 1])**2)
    if dist < val:
        val = dist
        arg = i

#stream_len = 180
#m.plot(segment[arg + 18: arg + stream_len, 0, 0], segment[arg + 18:arg + stream_len, 1, 1], color='red')

m.plot(trond_x, trond_y, color = 'orange', marker = 's', markersize = 4)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label, color='orange')

plt.xlabel('Longitude', fontsize = 18)
plt.ylabel('Latitude', fontsize = 18)
plt.show()
