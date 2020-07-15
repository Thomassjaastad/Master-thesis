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


np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv('Routing/aisToTrondheim.csv', index_col = 'TimestampPosition', parse_dates = True)
df = df.drop(columns=['dummy'])

trond_lat = [63.43049]
trond_lon = [10.39506]

trond_x, trond_y = convert.WGS84(np.rad2deg(trond_lat))

LonRad = np.deg2rad(df['Longitude'])
LatRad = np.deg2rad(df['Latitude'])
df['x'], df['y'] = convert.WGS84(LatRad)

# add columns to df
df['Heading'] = np.radians(df['Heading'])
df['VelocityLongitude'] = df['SpeedOverGround']*np.sin(df['Heading'])
df['VelocityLatitude'] = df['SpeedOverGround']*np.cos(df['Heading'])
df['DistTrondheim'] = np.sqrt((df['x'].values - trond_x)**2 + (df['y'].values - trond_y)**2)

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
    last_dist = route['DistTrondheim'][-1]
    first_dist = route['DistTrondheim'][0]
    df_subset.loc[route.index, 'Route number'] = i
    #df_subset.loc[i, 'route number']
    if last_dist < first_dist:
        df_subset.loc[route.index, 'Direction'] = 'To'  
        count = 0

        # check route going to if starting to deviate from dest.
        for j in range(route.shape[0] - 1):
            next_dist = route['DistTrondheim'][j + 1]
            prev_dist = route['DistTrondheim'][j]
            if next_dist < prev_dist:
                #dist decreasing
                count = 0
            else:
                #dist increasing
                count += 1
            if count == 3:
                #Ship could be travelling through
                df_subset.loc[route.index, 'Direction'] = 'Through'
                break
    else:
        df_subset.loc[route.index, 'Direction'] = 'From'

df_ToTrondheim = df_subset.loc[df_subset['Direction'] == 'To', 
                              ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheim', 'Route number']]


# single route from tugboat df
#df_route = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == 2.0, 
#                                         ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheim', 'Route number']]

# grid boundaries
minlon = max(-180, min(df_ToTrondheim['Longitude']))
minlat = max(-90, min(df_ToTrondheim['Latitude']))
maxlon = min(180, max(df_ToTrondheim['Longitude']))
maxlat = min(90, max(df_ToTrondheim['Latitude']))

nx, ny = 3, 3

minx, miny = convert.WGS84(np.deg2rad(minlat))
maxx, maxy = convert.WGS84(np.deg2rad(maxlat))

lat0 = (maxlat + minlat)/2 
lon0 = (maxlon + minlon)/2
lat1 = (maxlat + minlat)/2-20
lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)

fig, ax = plt.subplots(figsize = (30, 30))

# insert area threshold for more detailed map plot
eps = 0.01

# create map instance
m = Basemap(llcrnrlon = minlon, llcrnrlat = minlat, urcrnrlon = maxlon , urcrnrlat = maxlat,
            resolution = 'i', projection = 'cyl', lat_0 = lat0, lon_0 = lon0)

# formatting plot
#m.drawmapboundary(fill_color = 'lightblue')
#m.fillcontinents(color = 'brown', lake_color = 'lightblue', zorder=1)  

#m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
# coastline and panels values XB and YB

coast = m.drawcoastlines()
coordinates = coast.get_segments() 

# maybe change coordinate orientation?!?!
coordinates = np.vstack(coordinates)

# tweakable parameters!
Vinf = 1
AoA = 0 #np.arctan((trond_lat -  df_ToTrondheim['Latitude'][0])/(trond_lon - df_ToTrondheim['Longitude'][0]))

# boundary points in lon and lat degree values!
XB = coordinates[:, 0]
YB = coordinates[:, 1]
#XB, YB = convert.WGS84(np.deg2rad(latB))
plt.plot(XB,YB)
plt.show()
exit()
# rolling values for non lines over sea area
roll_idx = XB.size - 255
XB = np.roll(XB, roll_idx)
YB = np.roll(YB, roll_idx)

XC, YC = FlowBoundary.control_points(XB, YB)

#TODO:
# check delta angle. Not used in program! Necessary?
panels, phi = FlowBoundary.panels(XB, YB)
delta, beta = FlowBoundary.angles(phi, AoA)

# solving system of equations
I, J = sourcepanel.solveGeometricIntegrals(XC, YC, panels, phi, beta, XB, YB)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta)
lamda = np.linalg.solve(I, b)

# grid for flow computations
nxx, nyy = 100, 100

minx, miny = convert.WGS84(np.deg2rad(minlat))
maxx, maxy = convert.WGS84(np.deg2rad(maxlat))

# cart rep
#x_flow = np.linspace(minx, maxx, nxx)
#y_flow = np.linspace(miny, maxy, nyy)
#X, Y = np.meshgrid(x_flow, y_flow)

# lon and lat rep
x_flow = np.linspace(minlon, maxlon, nxx)
y_flow = np.linspace(minlat, maxlat, nyy)
X, Y = np.meshgrid(x_flow, y_flow)

# sink located at Trondheim
lamda_trond = 3 # sink strength
trond_flow = Flow.Velocity(lamda_trond, trond_lon, trond_lat)
vx_sink, vy_sink = trond_flow.sink(X, Y)

# exclude land areas
land_vals = find_land.arr(x_flow, y_flow, m)
land_vals = np.rot90(land_vals)
sea_vals = 1 - land_vals 
Y = np.flip(Y, axis = 0)

# XP and YP
# TODO: Try to only calculate in for loops. minimize if checks and decrease computation time. 
X_sea = X*sea_vals
Y_sea = Y*sea_vals

vx = np.zeros((nxx, nyy))
vy = np.zeros((nxx, nyy))

# compute velocites at every grid cell (XP, YP)
for i in range(nxx):
    for j in range(nyy):
        XP = X[i, j]
        YP = Y[i, j]
        XGeom, YGeom = sourcepoint.solvePointGeometry(XP, YP, panels, phi, XB, YB)
        if m.is_land(XP, YP) == True:
            vx[i, j] = 0 
            vy[i, j] = 0
        else:
            vx[i, j] = Vinf*np.cos(AoA) + np.dot(lamda, XGeom.T)/(2*np.pi) + vx_sink[i, j]  
            vy[i, j] = Vinf*np.sin(AoA) + np.dot(lamda, YGeom.T)/(2*np.pi) + vy_sink[i, j]  

#m.quiver(df['Longitude'], df['Latitude'], 
#         df['VelocityLongitude'], df['VelocityLatitude'], color = 'blue', scale = 2000, label = 'AIS-data')

# all tug boats
#m.quiver(df_subset['Longitude'], df_subset['Latitude'], 
#         df_subset['VelocityLongitude'], df_subset['VelocityLatitude'], 
#         scale=1200, color='C0', label = 'Tugboats')

# single route
#m.quiver(df_route['Longitude'], df_route['Latitude'],
#         df_route['VelocityLongitude'], df_route['VelocityLatitude'])


# plot tugboats     
#m.quiver(df_ToTrondheim['Longitude'], df_ToTrondheim['Latitude'],
#         df_ToTrondheim['VelocityLongitude'], df_ToTrondheim['VelocityLatitude'], 
#         scale=800, color = 'C0', label='Tugboats')
#[1: -1, 1: -1]

#plot velocities
m.quiver(X[1: -1, 1: -1], Y[1: -1, 1: -1], vx[1: -1, 1: -1], vy[1: -1, 1: -1]
         , color = 'green', zorder = 3, scale = 100, label = "Fluid velocity")

# plot velocities
#m.quiver(X[1: -1, 1: -1], Y[1: -1, 1: -1], vx[1: -1, 1: -1], vy[1: -1, 1: -1]
#         , color = 'green', zorder = 3, scale = 100, label = "Fluid velocity")

# starting points for streamlines 
# plt function not good at plotting points 
#Y_initpoints = np.linspace(y_flow[0], y_flow[-1], nyy)
#X_initpoints = x_flow[0]*np.ones(len(Y_initpoints))                            
#XY_init = np.vstack((X_initpoints.T, Y_initpoints.T)).T  
#plt.streamplot(X, Y, vx, vy, linewidth = 0.5, density = 10, color = 'r', 
#                   arrowstyle = '-')

#  labels = [left,right,top,bottom]
m.drawparallels(lat_bins, labels = [False, True, True, False])
m.drawmeridians(lon_bins, labels = [True, False, True, False])

# Trondheim
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

m.plot(trond_x, trond_y, color = 'orange', marker = 's', markersize = 10)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label, color='orange')

#plt.title("AIS-data in fjord of Trondheim", loc='left', fontsize= 16)
plt.xlabel('Longitude', fontsize = 18)
plt.ylabel('Latitude', fontsize = 18)
plt.legend()
plt.show()
