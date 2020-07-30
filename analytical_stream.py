import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import Flow
import find_land
import sys
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap

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

#print(df_ToTrondheim['Route number'].drop_duplicates())
#exit()
# single route from tugboat df
df_route = df_ToTrondheim.loc[df_ToTrondheim['Route number'] == 79.0, 
                                         ['Longitude', 'Latitude', 'VelocityLongitude', 'VelocityLatitude', 'DistTrondheim', 'Route number']]

# grid boundaries
minlon = 9.22149 
minlat = 63.6005
maxlon = 9.49229
maxlat = 63.767

lat0 = (maxlat + minlat)/2 
lon0 = (maxlon + minlon)/2
lat1 = (maxlat + minlat)/2-20

m = Basemap(llcrnrlon = minlon, llcrnrlat = minlat, urcrnrlon = maxlon, urcrnrlat = maxlat,
            resolution = 'f', projection = 'cyl', lon_0 = lon0, lat_0 = lat0)

#m.drawmapboundary(fill_color = 'lightblue')
#m.fillcontinents(color = 'green', lake_color = 'lightblue', zorder = 1) 
#
#plt.show()
#exit()

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
source_str = 200
x_source = 9.31318
y_source = 63.604
flow0 = Flow.Velocity(source_str, x_source, y_source)
u_source, v_source = flow0.source(X, Y)

sink_str = 1e3
x_sink = 9.48137
y_sink = 63.7375
flow1 = Flow.Velocity(sink_str, x_sink, y_sink)
u_sink, v_sink = flow1.sink(X, Y)


# Trying to represent source and doublets as islands. Use correct strengths in for loop
source_strs = [100, 200, 300, 400, 500, 750]
#doublet_strs = [0.01, 0.1, 0.2, 0.3, 0.5, 1]

gs = gridspec.GridSpec(nrows = 3, ncols = 2)
fig, ax = plt.subplots(3, 2, figsize = (10, 10), sharex=True, sharey=True)

ax = ax.ravel()

# source islands placement
x_island0 = 9.41911
y_island0 = 63.6656 

strength_island1 = source_strs[0]    
x_island1 = 9.32632
y_island1 = 63.6425

# remember to change list doublet or source calc
for count, strengths in enumerate(source_strs, 0):
    m = Basemap(llcrnrlon = minlon, llcrnrlat = minlat, urcrnrlon = maxlon, urcrnrlat = maxlat,
        resolution = 'f', projection = 'cyl', lon_0 = lon0, lat_0 = lat0, ax=ax[count])

    m.drawmapboundary(fill_color = 'lightblue')
    m.fillcontinents(color = 'green', lake_color = 'lightblue', zorder = 3)
  

    flow2 = Flow.Velocity(strengths, x_island0, y_island0)
    u_source0, v_source0 = flow2.source(X, Y)

    flow3 = Flow.Velocity(strength_island1, x_island1, y_island1)
    u_source1, v_source1 = flow3.source(X, Y)

    # Superposition source island rep
    u_tot = u_sink + u_source + u_source0 + u_source1  
    v_tot = v_sink + v_source + v_source0 + v_source1 
    
    
    """
    flow2 = Flow.Velocity(strengths, x_island0, y_island0)
    u_doublet0, v_doublet0 = flow2.doublet(X, Y)
    
    flow3 = Flow.Velocity(strengths, x_island1, y_island1)
    u_doublet1, v_doublet1 = flow3.doublet(X, Y)
    
    # Superposition doublet island rep
    u_tot = u_source + u_doublet1 + u_doublet0 
    v_tot = v_source + v_doublet1 + v_doublet0
    """

    #rad = 0.01
    for i in range(nx):
        for j in range(ny):
            if m.is_land(X[i,j], Y[i, j]) == True:
                u_tot[i,j] = 0
                v_tot[i,j] = 0    
            #insiderad0 = np.sqrt((X[i, j] - x_island0)**2 + (Y[i,j] - y_island0)**2)
            #insiderad1 = np.sqrt((X[i, j] - x_island1)**2 + (Y[i,j] - y_island1)**2)
            #if insiderad0 <= rad or insiderad1 <= rad:
            #    u_tot[i, j] = 0
            #    v_tot[i, j] = 0
    
    #ax[count] = fig.add_subplot(gs[0, 0])
    stream = m.streamplot(X, Y, u_tot, v_tot, linewidth = 0.5, density = 2, color = 'b', 
                       arrowstyle = '->')
    #plt.savefig('islands_stream%d' % count)
    #plt.quiver(df_route['Longitude'], df_route['Latitude'], 
    #            df_route['VelocityLongitude'], df_route['VelocityLatitude'], scale = 400, label='AIS-data')

    """------path coordinates------"""
    path = stream.lines.get_paths()
    path = np.asarray(path)
    segment = stream.lines.get_segments()
    segment = np.asarray(segment)



    #line_segments = LineCollection(segment[10:50], color = 'red', label='streamline route')
    #ax.add_collection(line_segments)

    #plt.title('Vessel motion and streamline', fontsize = 18)

    #circle0 = plt.Circle((x_island0, y_island0), radius = rad, color = 'black', zorder=2)
    #circle1 = plt.Circle((x_island1, y_island1), radius = rad, color = 'black', zorder=2)
    #ax[count].add_artist(circle0)
    #ax[count].add_artist(circle1)
    m.scatter(x_island1, y_island1, color = 'yellow' ,zorder = 4, label= 'source str = %.3f' % strength_island1)
    m.scatter(x_island0, y_island0, color = 'orange' ,zorder = 4, label = 'source str = %.3f' % strengths)
    m.scatter(x_sink, y_sink, color = 'purple', label = 'End point')
    m.scatter(x_source, y_source, color = 'red', label= 'Starting point' )
    ax[count].legend(loc='upper left',fontsize=6)
plt.setp(ax[:], xlabel='lon')
plt.setp(ax[:], ylabel='lat')
plt.show()
