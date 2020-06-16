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
#print(minlon, min(df['Longitude']))
#exit()
nx, ny = 3, 3

lat0 = (maxlat + minlat)/2 
lon0 = (maxlon + minlon)/2
lat1 = (maxlat + minlat)/2-20
lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)

fig, ax = plt.subplots(figsize = (30, 30))
# insert area threshold for more detailed map plot
eps = 0.01
m = Basemap(llcrnrlon = minlon + eps, llcrnrlat = minlat + eps, urcrnrlon = maxlon , urcrnrlat = maxlat,
            resolution = 'i', projection = 'cyl', lat_0 = lat0, lon_0 = lon0)

# formatting plot
m.drawmapboundary(fill_color = 'white')
m.fillcontinents(color = 'lightgrey', lake_color = 'white', zorder=1)  
# coastline and panels values XB and YB
coast = m.drawcoastlines()
print(type(coast), coast)
coordinates = coast.get_segments() 
print(type(coordinates), len(coordinates), coordinates)
exit()
coordinates = np.vstack(coordinates)

# tweakable parameters!
Vinf = -1
AoA = np.radians(45)

XB = coordinates[:, 0]
YB = coordinates[:, 1]

# rolling values for non lines over sea area
roll_idx = XB.size - 255
XB = np.roll(XB, roll_idx)
YB = np.roll(YB, roll_idx)

# creating boundaries for fluid flow 
#XB = np.insert(XB, 0, minlon)
#YB = np.insert(YB, 0, maxlat)
#XB = np.append(XB, minlon)
#YB = np.append(YB, maxlat)

#print(XB[0], YB[0])

idx = [50, 100, 150, 200, 250]

#for i in range(XB.size):
#    if i < idx[0]:
#        m.scatter(XB[i], YB[i], color = 'red')
#    if i > idx[0] and i < idx[1]:
#        m.scatter(XB[i], YB[i], color = 'blue')
#    if i > idx[1] and i < idx[2]:
#        m.scatter(XB[i], YB[i], color = 'yellow')
#    if i > idx[2] and i < idx[3]:
#        m.scatter(XB[i], YB[i], color = 'orange')
#    if i > idx[3] and i < idx[4]:
#        m.scatter(XB[i], YB[i], color = 'green')
#    if i > idx[4]:
#        m.scatter(XB[i], YB[i], color = 'teal')

#XB = np.insert(XB, 0, minlon)
#YB = np.insert(YB, 0, maxlat)

#m.plot(XB, YB, color = 'red')
XC, YC = FlowBoundary.control_points(XB, YB)

# remove excess segments
#XC = np.delete(XC, 27)
#YC = np.delete(YC, 27)
#XB = np.delete(XB, 27)
#YB = np.delete(YB, 27)

#print(XC.shape, XB.shape)
#m.scatter(XC, YC, color = 'blue')
#plt.show()

#TODO:
# Check delta angle. Not used in program! Necessary?
panels, phi = FlowBoundary.panels(XB, YB)
delta, beta = FlowBoundary.angles(phi, AoA)

I, J = sourcepanel.solveGeometricIntegrals(XC, YC, panels, phi, beta, XB, YB)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta)
lamda = np.linalg.solve(I, b)

#for i in range(lamda.size):
#    if abs(lamda[i]) > 900:
#        lamda[i] = 0
            
# grid for flow computations
nxx, nyy = 20, 20

x_flow = np.linspace(minlon, maxlon, nxx)
y_flow = np.linspace(minlat, maxlat, nyy)
X, Y = np.meshgrid(x_flow, y_flow)

# sink located at Trondheim
lamda_trond = 3 # sink strength
trond_flow = Flow.VelocityPotential(lamda_trond, trond_lon, trond_lat)
vx_sink, vy_sink = trond_flow.sink(X, Y)

# create boundary
push_pot_x = np.zeros((nxx, nyy))
push_pot_y = np.zeros((nxx, nyy))

for i in range(nxx):
    for j in range(nyy):
        if i < nxx/2 and j < nyy/2:
            push_pot_x[i, j] = 1
            push_pot_y[i, j] = -1
            #m.scatter(push_pot_x[i,j], push_pot_y[i, j],color='red')
        if i >= nxx/2 and j < nyy/2:
            push_pot_x[i, j] = -1
            push_pot_y[i, j] = 1
            #m.scatter(push_pot_x[i,j], push_pot_y[i, j],color='blue')
        if i < nxx/2 and j >= nyy/2:
            push_pot_x[i, j] = 1
            push_pot_y[i, j] = -1
            #m.scatter(push_pot_x[i,j], push_pot_y[i, j],color='green')
        if i >= nxx/2 and j >= nyy/2:
            push_pot_x[i, j] = -1
            push_pot_y[i, j] = -1
            #m.scatter(push_pot_x[i,j], push_pot_y[i, j],color='yellow')

#push_pot_x = np.rot90(push_pot_x)
#push_pot_y = np.rot90(push_pot_y)

# exclude land areas
land_vals = find_land.arr(x_flow, y_flow, m)
land_vals = np.rot90(land_vals)
sea_vals = 1 - land_vals 
Y = np.flip(Y)

# Trondheim
trond_x, trond_y = m(trond_lon, trond_lat)
labels = ['Trondheim']

# XP and YP
# TODO: Try to only calculate in for loops. minmize if checks and decrease computation time. 
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
            vx[i, j] = Vinf*np.cos(AoA) + np.dot(lamda, XGeom.T)/(2*np.pi) + vx_sink[i, j] #+ push_pot_x[i, j]
            vy[i, j] = Vinf*np.sin(AoA) + np.dot(lamda, YGeom.T)/(2*np.pi) + vy_sink[i, j] #+ push_pot_y[i, j]

#m.quiver(df['Longitude'], df['Latitude'], 
#         df['VelocityLongitude'], df['VelocityLatitude'], scale = 1000)
#plt.show()
#exit()
m.quiver(df_subset['Longitude'], df_subset['Latitude'], 
         df_subset['VelocityLongitude'], df_subset['VelocityLatitude'])

m.plot(trond_x, trond_y, color = 'black', marker = 's', markersize = 15)

plt.title('Tugboats travelling Trondheimsfjorden', fontsize = 20)
plt.xlabel('Longitude', fontsize = 18)
plt.ylabel('Latitude', fontsize = 18)
plt.show()

exit()
# plot ships    
m.quiver(df_ToTrondheim['Longitude'], df_ToTrondheim['Latitude'],
         df_ToTrondheim['VelocityLongitude'], df_ToTrondheim['VelocityLatitude'], scale=1000)
#[1: -1, 1: -1]
# plot velocities

m.quiver(X, Y, vx, vy, color='green', zorder=3)

# starting points for streamlines 
# plt function not good at plotting points 
#Y_initpoints = np.linspace(y_flow[0], y_flow[-1], nyy)
#X_initpoints = x_flow[0]*np.ones(len(Y_initpoints))                            
#XY_init = np.vstack((X_initpoints.T, Y_initpoints.T)).T  
#plt.streamplot(X, Y, vx, vy, linewidth = 0.5, density = 10, color = 'r', 
#                   arrowstyle = '-')

m.plot(trond_x, trond_y, color = 'black', marker = 's', markersize = 15)
for label, xpt, ypt in zip(labels, trond_x, trond_y):
    plt.text(xpt, ypt, label)
#m.scatter(X[1:-1,1:-1], Y[1:-1, 1:-1], marker='o', color='red', zorder=2, label='Land')
#m.scatter(X_sea[1:-1,1:-1], Y_sea[1:-1, 1:-1], color='teal', zorder=3, label='Sea')
#plt.legend()
plt.show()
