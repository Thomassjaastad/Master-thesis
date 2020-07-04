import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
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
df_subset = df[df['ShipType'].isin(['cargo_ships'])].copy()

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

print(df_ToTrondheim['Route number'].drop_duplicates())
#exit()
# single route from tugboat df
df_route = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == 79.0, 
                                         ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheim', 'Route number']]

# grid boundaries
minlon = max(-180, min(df_route['Longitude']))
minlat = max(-90, min(df_route['Latitude']))
maxlon = min(180, max(df_route['Longitude']))
maxlat = min(90, max(df_route['Latitude']))



plt.show()

# Create streamline plot for vessel planning
# Set out:
# Source -> start point
# Sink -> end point 
# doublet -> island (none - traversable area)

# Set up grid
nx, ny = 30, 30
x = np.linspace(minlon, maxlon, nx)
y = np.linspace(minlat, maxlat, ny)
X, Y = np.meshgrid(x, y)

# Create flow types
source_str = 40
x_source = df_route['Longitude'][0]
y_source = df_route['Latitude'][0]
flow0 = Flow.Velocity(source_str, x_source, y_source)
u_source, v_source = flow0.source(X, Y)

sink_str = 150
x_sink = df_route['Longitude'][-1]
y_sink = df_route['Latitude'][-1]
flow1 = Flow.Velocity(sink_str, x_sink, y_sink)
u_sink, v_sink = flow1.sink(X, Y)

#doublet_str0 = 0.05
#x_doublet0 = 10.8
#y_doublet0 = 63.5
#flow2 = Flow.Velocity(doublet_str0, x_doublet0, y_doublet0)
#u_doublet0, v_doublet0 = flow2.doublet(X, Y)
#
#doublet_str1 = 0.02
#x_doublet1 = 10.9
#y_doublet1 = 63.77
#flow3 = Flow.Velocity(doublet_str1, x_doublet1, y_doublet1)
#u_doublet1, v_doublet1 = flow3.doublet(X, Y)


# Superposition
u_tot = u_sink + u_source# + u_doublet0 + u_doublet1 
v_tot = v_sink + v_source# + v_doublet0 + v_doublet1

#rad = 0.1
#for i in range(nx):
#    for j in range(ny):
#        insiderad0 = np.sqrt((x_doublet0 - x[i])**2 + (y_doublet0 - y[j])**2)
#        insiderad1 = np.sqrt((x_doublet1 - x[i])**2 + (y_doublet1 - y[j])**2)
#        
#        if insiderad0 <= rad and insiderad1 <= rad:
#            u_tot[i, j] = 0
#            v_tot[i, j] = 0

#plt.quiver(X, Y, u_tot, v_tot)

fig, ax = plt.subplots()

stream = ax.streamplot(X, Y, u_tot, v_tot, linewidth = 0.5, density = 3, color = 'b', 
                   arrowstyle = '->')

plt.quiver(df_route['Longitude'], df_route['Latitude'], 
            df_route['VelocityLongitude'], df_route['VelocityLatitude'], scale = 400, label='AIS-data')

"""------path coordinates------"""
path = stream.lines.get_paths()
path = np.asarray(path)

segment = stream.lines.get_segments()
segment = np.asarray(segment)

line_segments = LineCollection(segment[10:50], color = 'red', label='streamline route')
ax.add_collection(line_segments)

#plt.scatter(x_doublet1, y_doublet1, color = 'orange')
#plt.scatter(x_doublet0, y_doublet0, color = 'orange')
plt.scatter(x_sink, y_sink, color = 'purple', label = 'End')
plt.scatter(x_source, y_source, color = 'green', label= 'Start')
plt.title('Vessel motion and streamline', fontsize = 18)
plt.xlabel('X', fontsize = 16)
plt.ylabel('Y', fontsize = 16)
plt.legend(loc='upper left')
plt.show()