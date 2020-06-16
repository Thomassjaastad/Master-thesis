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
np.set_printoptions(threshold=sys.maxsize)


df = pd.read_csv('Routing/aisToTrondheim.csv', index_col = 'TimestampPosition', parse_dates = True)
df = df.drop(columns=['dummy'])

trond_lat = [63.43049]
trond_lon = [10.39506]

# add columns to df
df['Heading'] = np.radians(df['Heading'])
df['VelocityLongitude'] = df['SpeedOverGround']*np.sin(df['Heading'])
df['VelocityLatitude'] = df['SpeedOverGround']*np.cos(df['Heading'])
df['DistTrondheim'] = np.sqrt((df['Longitude'].values - trond_lon)**2 + (df['Latitude'].values - trond_lat)**2)

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
                              ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheim']]

# grid boundaries
minlon = max(-180, min(df['Longitude']))
minlat = max(-90, min(df['Latitude']))
maxlon = min(180, max(df['Longitude']))
maxlat = min(90, max(df['Latitude']))

nx, ny = 3, 3

lat0 = (maxlat + minlat)/2 
lon0 = (maxlon + minlon)/2
lat1 = (maxlat + minlat)/2-20
lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)

fig, ax = plt.subplots(figsize = (30, 30))

# insert area threshold for more detailed map plot
eps = 0.0
m = Basemap(llcrnrlon = minlon + eps, llcrnrlat = minlat + eps, urcrnrlon = maxlon + eps, urcrnrlat = maxlat + eps,
            resolution = 'i', projection = 'cyl', lat_0 = lat0, lon_0 = lon0, lat_ts = lat1)

# formatting plot
m.drawmapboundary(fill_color = 'white')
m.fillcontinents(color = 'lightgrey', lake_color = 'white', zorder=1)  

# coastline and panels values XB and YB
coast = m.drawcoastlines() 
coordinates = coast.get_segments()
coords = np.vstack(coordinates)
XB = coords[:, 0]
YB = coords[:, 1]
roll_idx = XB.size - 255
XB = np.roll(XB, roll_idx)
YB = np.roll(YB, roll_idx)
#for i in range(XB.shape[0]):
#    if i < 50:
#        m.plot(XB[i], YB[i], color='red',marker = 'o')
#    if i > 50 and i < 100:
#        m.plot(XB[i], YB[i], color='green',marker = 'o')
#    if i > 100 and i < 150:
#        m.plot(XB[i], YB[i], color='yellow',marker = 'o')
#    if i > 150 and i < 200:
#        m.plot(XB[i], YB[i], color='blue',marker = 'o')
#    if i > 200 and i < 255:
#        m.plot(XB[i], YB[i], color='orange',marker = 'o')
#    if i > 254:
#        m.plot(XB[i], YB[i], color='black',marker = 'o')
#testarr = np.arange(10)
#print(np.roll(testarr, -2))
# need to change coords[254:] to start of coords
#plt.legend()
m.plot(XB, YB)
plt.show()
exit()
#coordinates = np.vstack(coast.get_segments())

# tweekable parameters!
Vinf = 1
AoA = np.radians(0)

XB = coordinates[:, 0]
YB = coordinates[:, 1]
XC, YC = FlowBoundary.control_points(XB, YB)

bools = find_land.points(XC, YC, m)
bools = 1 - bools 
testX = bools*XC
testY = bools*YC

panels, phi = FlowBoundary.panels(XB, YB)
delta, beta = FlowBoundary.angles(phi, AoA)

#m.scatter(testX, testY, color = 'red')
for i in range(XB.shape[0]):
    if i < 50:
        m.plot(XB[i], YB[i], color = 'red', marker = 'o')
    if i > 50 and i < 100:
        m.plot(XB[i], YB[i], color = 'green', marker = 'o')
    if i > 100 and i < 150:
        m.plot(XB[i], YB[i], color = 'yellow', marker = 'o')
    if i > 150 and i < 200:
        m.plot(XB[i], YB[i], color = 'blue', marker = 'o')
    if i > 200 and i < 250:
        m.plot(XB[i], YB[i], color = 'orange', marker = 'o')
    if i > 250:
        m.plot(XB[i], YB[i], color = 'black', marker = 'o')
        
m.plot(XB,YB)
#plt.legend()
plt.show()

exit()


I, J = sourcepanel.solveGeometricIntegrals(XC, YC, panels, phi, beta, XB, YB)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta)
lamda = np.linalg.solve(I, b)

# find land and sea areas     
# grid for flow computations
nxx = 10
nyy = 10
x_flow = np.linspace(minlon, maxlon, nxx)
y_flow = np.linspace(minlat, maxlat, nyy)
X, Y = np.meshgrid(x_flow, y_flow)

# sink located at Trondheim
lamda_trond = 3 # sink strength
trond_flow = Flow.VelocityPotential(lamda_trond, trond_lon, trond_lat)

vx_sink, vy_sink = trond_flow.sink(X, Y)

#land_vals = find_land(x_flow, y_flow)
#land_vals = np.rot90(land_vals)
#sea_vals = 1 - land_vals 
#Y = np.flip(Y)

# XP and YP
# TODO: Try to only calculate in for loops. minmize if checks and decrease computation time. 
#X_sea = X*sea_vals
#Y_sea = Y*sea_vals

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

# plot ships    
m.quiver(df_ToTrondheim['Longitude'], df_ToTrondheim['Latitude'],
         df_ToTrondheim['VelocityLongitude'], df_ToTrondheim['VelocityLatitude'], scale=1000)

# plot velocities
m.quiver(X[1: -1, 1: -1], Y[1: -1, 1: -1], vx[1: -1,1: -1], vy[1: -1,1: -1],
         scale = 500, color='green', zorder=3)

# Trondheim
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

m.plot(trond_x, trond_y, color = 'black', marker = 's', markersize = 15)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label)
#m.scatter(X[1:-1,1:-1], Y[1:-1, 1:-1], marker='o', color='red', zorder=2, label='Land')
#m.scatter(X_sea[1:-1,1:-1], Y_sea[1:-1, 1:-1], color='teal', zorder=3, label='Sea')
#plt.legend()
plt.show()
