import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import sourcepanel
import sourcepoint
import FlowBoundary
import Flow
import find_land
import sys
import convert
from scipy import linalg

np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('Routing/aisToTrondheim.csv', index_col = 'TimestampPosition', parse_dates = True)
df = df.drop(columns=['dummy'])

trond_lat = [63.43049]
trond_lon = [10.39506]

trond_x, trond_y = convert.tocartesian(trond_lat)

df['x'], df['y'] = convert.tocartesian(df['Latitude'])

lonconv, latconv = convert.tolonlat(df['x'], df['y'], df['Latitude'])


#print(df['Longitude'].head(), df['Latitude'].head(), df['x'].head(), df['y'].head())

lats = df['Latitude'].values
x, y = convert.tocartesian(lats)
lon, lat = convert.tolonlat(x, y, lats) 

# add columns to df
df['Heading'] = np.radians(df['Heading'])
df['VelocityLongitude'] = df['SpeedOverGround']*np.sin(df['Heading'])
df['VelocityLatitude'] = df['SpeedOverGround']*np.cos(df['Heading'])
df['DistTrondheimCart'] = np.sqrt((df['x'].values - trond_x)**2 + (df['y'].values - trond_y)**2)
#df['DistTrondheimWGS'] = np.sqrt((df['Longitude'].values - trond_lon)**2 + (df['Latitude'].values - trond_lat)**2)
df['DistTrondheimHaversine'] = convert.distsphere(df['Longitude'], trond_lon, df['Latitude'], trond_lat)

# extract subset of original df to analyze. Dataset to big to analyze all together
df_subset = df[df['ShipType'].isin(['tugboats'])].copy()

# get rid of hours, minutes and seconds in Timestamp col/idx
date = df_subset.index.to_period('d')
unique_dates = date.drop_duplicates()

#route0 = df_subset.loc[str(unique_dates[0])]
#route1 = df_subset.loc[str(unique_dates[1])]

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

#print(df_ToTrondheim['DistTrondheimHaversine'].head())
# single route from tugboat df
unique_routes = df_ToTrondheim['Route number'].drop_duplicates()
df_route = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == 9.0, 
        ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheimHaversine', 'Route number']]

# grid boundaries
minlon = max(-180, min(df_ToTrondheim['Longitude']))
minlat = max(-90, min(df_ToTrondheim['Latitude']))
maxlon = min(180, max(df_ToTrondheim['Longitude']))
maxlat = min(90, max(df_ToTrondheim['Latitude']))

nx, ny = 3, 3
#minx, miny = convert.tocartesian(minlat)
#maxx, maxy = convert.tocartesian(maxlat)

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

#m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
# coastline and panels values XB and YB

# maybe change coordinate orientation?!?!
coast = m.drawcoastlines()
coordinates = coast.get_segments() 
coordinates = np.vstack(coordinates)

# boundary points in lon and lat degree values!
XB = coordinates[:, 0]
YB = coordinates[:, 1]
#y_argmax = np.argmax(YB)

# 228 idx is the top left boundary point
XB = np.insert(XB, 228, minlon)
YB = np.insert(YB, 228, maxlat)

XB = np.insert(XB, 229, XB[249])
YB = np.insert(YB, 229, YB[249])

XC, YC = FlowBoundary.control_points(XB,YB)
#TODO: check delta angle. Not used in program! Necessary?
# tweakable parameters!

Vinf = 0.01
AoA = -np.pi #np.arctan((trond_lat -  df_ToTrondheim['Latitude'][0])/(trond_lon - df_ToTrondheim['Longitude'][0]))
panels, phi = FlowBoundary.panels(XB, YB)
delta, beta = FlowBoundary.angles(phi, AoA)

# convert to cartesian for avoiding numerical errors?
#XB, YB = convert.tocartesian(YB)
#XC, YC = convert.tocartesian(YC)

# solving system of equations
I, J = sourcepanel.solveGeometricIntegrals(XC, YC, panels, phi, beta, XB, YB)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta)
lamda = linalg.solve(I, b)


# grid for flow computations
nxx, nyy = 40, 40

# cart rep
#x_flow = np.linspace(minx, maxx, nxx)
#y_flow = np.linspace(miny, maxy, nyy)
#X, Y = np.meshgrid(x_flow, y_flow)

# lon and lat rep
x_flow = np.linspace(minlon, maxlon, nxx)
y_flow = np.linspace(minlat, maxlat, nyy)
X, Y = np.meshgrid(x_flow, y_flow)

# sink located at Trondheim
lamda_trond = 0.1        # sink strength
trond_flow = Flow.Velocity(lamda_trond, trond_lon, trond_lat)
vx_sink, vy_sink = trond_flow.sink(X, Y)

#source located at edge of map upper left
#source_bound = 2
#source_flow = Flow.Velocity(source_bound, minlon, maxlat)
#vx_source, vy_source = source_flow.source(X, Y)

#source_bound1 = 2
#source_flow1 = Flow.Velocity(source_bound1, maxlon, maxlat)
#vx_source1, vy_source1 = source_flow1.source(X, Y)
#
#source_bound2 = 2
#source_flow2 = Flow.Velocity(source_bound2, 8.88112, 63.4482)
#vx_source2, vy_source2 = source_flow2.source(X, Y)

#source_bound3 = 10
#source_flow3 = Flow.Velocity(source_bound3, 9.43876, 64.0045)
#vx_source3, vy_source3 = source_flow3.source(X, Y)

# exclude land areas
#land_vals = find_land.arr(x_flow, y_flow, m)
#land_vals = np.rot90(land_vals)
#sea_vals = 1 - land_vals 
#Y = np.flip(Y, axis = 0)

# XP and YP
# TODO: Try to only calculate in for loops. minimize if checks and decrease computation time. 
#X_sea = X*sea_vals
#Y_sea = Y*sea_vals

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
            #if X[i,j] > trond_lon:
            #    Vinf = -1
            #else: 
            #    Vinf = 2
            vx[i, j] = Vinf*np.cos(AoA) + np.dot(lamda, XGeom.T/(2*np.pi)) + vx_sink[i, j]  #+ np.sum(lamda*XGeom/(2*np.pi)) #np.dot(lamda, XGeom.T)/(2*np.pi) + vx_sink[i, j] #+ vx_source2[i,j] + vx_source1[i, j]  
            vy[i, j] = Vinf*np.sin(AoA) + np.dot(lamda, YGeom.T/(2*np.pi)) + vy_sink[i, j]  #+ np.sum(lamda*YGeom/(2*np.pi)) #  #+ vy_source2[i,j] + vy_source1[i, j]  

#m.quiver(df['Longitude'], df['Latitude'], 
#         df['VelocityLongitude'], df['VelocityLatitude'], color = 'blue', scale = 2000, label = 'AIS-data')

# all tug boats
#m.quiver(df_subset['Longitude'], df_subset['Latitude'], 
#         df_subset['VelocityLongitude'], df_subset['VelocityLatitude'], 
#         scale=1100, color='C0')

# single route
#m.quiver(df_route['Longitude'], df_route['Latitude'],
#         df_route['VelocityLongitude'], df_route['VelocityLatitude'], scale = 800)

#plt.plot(df_route['Longitude'], df_route['Latitude'])
# plot tugboats     
#m.quiver(df_ToTrondheim['Longitude'], df_ToTrondheim['Latitude'],
#         df_ToTrondheim['VelocityLongitude'], df_ToTrondheim['VelocityLatitude'], 
#         scale=800, color = 'C0', label='Tugboats')
#[1: -1, 1: -1]

#plot velocities
#m.quiver(X[1: -1, 1: -1], Y[1: -1, 1: -1], vx[1: -1, 1: -1], vy[1: -1, 1: -1]
#         , color = 'C0', zorder = 3, scale = 200, label = "Fluid velocity")

# plot velocities
#m.quiver(X[1: -1, 1: -1], Y[1: -1, 1: -1], vx[1: -1, 1: -1], vy[1: -1, 1: -1]
#         , color = 'green', zorder = 3, scale = 100, label = "Fluid velocity")

# starting points for streamlines 
# plt function not good at plotting points 
#X_initpoints = np.linspace(x_flow[0], x_flow[-1], nxx)
#Y_initpoints = y_flow[-1]*np.ones(len(X_initpoints))                            
#XY_init = np.vstack((X_initpoints.T, Y_initpoints.T)).T  
plt.streamplot(X, Y, vx, vy, linewidth = 0.75, density = 4, color = 'C0', 
                   arrowstyle = '->')

#  labels = [left,right,top,bottom]
m.drawparallels(lat_bins, labels = [False, True, True, False])
m.drawmeridians(lon_bins, labels = [True, False, True, False])

# Trondheim
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

#plt.scatter(9.43876, 64.0045, label = 'source')
#plt.scatter(minlon, maxlat, label = 'source')
#plt.scatter(maxlon, maxlat, label = 'source')
#plt.scatter(8.88112, 63.4482,label = 'source')
m.plot(trond_x, trond_y, color = 'orange', marker = 's', markersize = 4)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label, color='orange')

#plt.title("AIS-data in fjord of Trondheim", loc='left', fontsize= 16)
plt.xlabel('Longitude', fontsize = 18)
plt.ylabel('Latitude', fontsize = 18)
plt.legend()
plt.show()
