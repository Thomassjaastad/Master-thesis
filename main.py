import numpy as np
import os
import matplotlib.pyplot as plt
import Flow
import pandas as pd
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
import Grid

"""
Program for visualizing Potential theory in Fluid mechanics 
Defining several stream functions, using superposition to plot flow field
Dipol, sink and constant fields are used. 
"""

"""map specs"""
x0 = 0
y0 = 0

xn = 10
yn = 10

"""Grid init quiver specs"""
nx = 9
ny = 9

Grid_quiver = Grid.Grid(x0, y0, xn, yn, nx, ny) # maybe change
X, Y = Grid_quiver.map()                        # maybe change

#"""stream specs for plot"""
nx_strm = 50
ny_strm = 50

Grid_stream = Grid.Grid(x0, y0, xn, yn, nx_strm, ny_strm)
X_strm, Y_strm = Grid_stream.map()

"""Creating environment"""

# uniform flow
flow = Flow.VelocityPotential(1, X_strm, Y_strm)
u_uni, v_uni = flow.uniform(1, 0) #strengths parameters as input here

# sink flow
x0_sink = 8
y0_sink = 8

sinkflow = Flow.VelocityPotential(20, x0_sink, y0_sink)
u_sink, v_sink = sinkflow.sink(X_strm, Y_strm)

# source flow
x0_source = 2
y0_source = 2

sourceflow = Flow.VelocityPotential(20, x0_source, y0_source)
u_source, v_source = sourceflow.source(X_strm, Y_strm)

# doublet flow
x0_doublet = 5
y0_doublet = 5

doubletflow = Flow.VelocityPotential(10, x0_doublet, y0_doublet)
u_doublet, v_doublet = doubletflow.doublet(X_strm, Y_strm)



"""Finalizing environment with superposition principle for quiver plot"""
u_tot = u_source + u_sink + u_doublet  
v_tot = v_source + v_sink + v_doublet 

#u_tot = u_doublet
#v_tot = v_doublet


"""----------------------------Plotting------------------------------------"""
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#


fig, ax = plt.subplots()
eps = 1e-3
seed_points = np.array([[4], [1]])
end_points = np.array([[x0_sink], [y0_sink]])

#whole_lines = np.array([np.arange(0,9,0.1), np.arange(0,9,0.1)])
#seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

#ax.set_title("A doublet flow located at (%.1f, %.1f) "% (x0_source, y0_source))


"""-----quiver specs------"""
#ax.scatter(X, Y, color='r', s=15) #velocity component position
#speed = np.hypot(u_tot, v_tot)
#q1 = ax.quiver(X, Y, u_tot, v_tot, speed, units = 'xy', pivot = 'middle', width=0.1, scale=4, headwidth=2)
#qk = ax.quiverkey(q1, 0.9, 0.9, 1, r'$2 \frac{m}{s}$', labelpos='W',coordinates = 'figure')

"""------streamline specs------"""
strm = ax.streamplot(X_strm, Y_strm, u_tot, v_tot, linewidth=0.5)
plt.plot(x0_sink, y0_sink, 'o')
plt.plot(x0_doublet, y0_doublet, 'o')
plt.plot(x0_source, y0_source, 'o')
#plt.plot(seed_points[0], seed_points[1], 'bo')    #start pos for ship
#plt.plot(end_points[0], end_points[1], 'go')      #end pos for ship
#
#"""------path coordinates------"""
#path = strm.lines.get_paths()
#path = np.asarray(path)
#segment = strm.lines.get_segments()
#segment = np.asarray(segment)
#
#"""------Plotting circle/boundaries for dipol with  center values------"""
#dipol_center_0 = plt.Circle((x0_dipol_0, y0_dipol_0), 2, fill = False)
#ax = plt.gca()
#ax.add_artist(dipol_center_0)
#plt.plot(x0_dipol_0, y0_dipol_0,'o')
#
#dipol_center_1 = plt.Circle((x0_dipol_1, y0_dipol_1), 2, fill = False)
#ax = plt.gca()
#ax.add_artist(dipol_center_1)
#plt.plot(x0_dipol_1, y0_dipol_1,'o')
#
#dipol_center_2 = plt.Circle((x0_dipol_2, y0_dipol_2), 2, fill = False)
#ax = plt.gca()
#ax.add_artist(dipol_center_2)
#plt.plot(x0_dipol_2, y0_dipol_2,'o')

#"""------Colorbars and add-ons------"""
#
#cbar = plt.colorbar(q1)
#cbar.set_label('Speed magnitude')
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.show()


"""real ship motion program with use of AIS-data"""


# specify shiptype 
dfTo = pd.read_csv('AISData/ais2Trondheim.csv')
ais_data = dfTo[dfTo['ShipType'].isin(['tugboats'])]
lonVals = ais_data['Longitude'].to_numpy()

"""--------------------------- Creating map boundaries and grid specs ------------------------------""" 

minlon = max(-180, min(ais_data['Longitude']))
minlat = max(-90, min(ais_data['Latitude']))
maxlon = min(180, max(ais_data['Longitude']))
maxlat = min(90, max(ais_data['Latitude']))

nx, ny = 5, 5

lat0 = (maxlat+minlat)/2 
lon0 = (maxlon+minlon)/2
lat1 = (maxlat+minlat)/2-20

lon_bins = np.linspace(minlon, maxlon, nx)
lat_bins = np.linspace(minlat, maxlat, ny)

density, _, _ = np.histogram2d(ais_data['Latitude'], ais_data['Longitude'], [lat_bins, lon_bins])


lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)

"""------------------------------------------------------------------------------------------------------"""
"""-------------------------------------------Creating map class-----------------------------------------"""

fig,ax=plt.subplots(figsize=(15,15))
m = Basemap(llcrnrlon=minlon, llcrnrlat=minlat, urcrnrlon=maxlon, urcrnrlat=maxlat,
            resolution='i', projection='cyl',lat_0=lat0, lon_0=lon0,lat_ts=lat1)

# convert the xs and ys to map coordinates
xs, ys = m(lon_bins_2d, lat_bins_2d)
print('------------- Density matrix flipped compared to grid plot -----------\n')
print(np.flipud(density), '\n')
print('------------------------- longitude bins ----------------------------------\n')
print(xs[0,0], xs.shape, '\n')
print('------------------------  latitude bins  ----------------------------------\n')
print(ys[0,0], ys.shape)

plt.pcolormesh(xs, ys, density)
plt.colorbar(orientation='vertical', label='Density')

#xsize = (maxlat - minlat)/(xs[0].size - 1)
#ysize = (maxlon - minlon)/(ys[0].size - 1)
#
#parallels = np.arange(minlat, maxlat, xsize) #latitude   y-axis
#meridians = np.arange(minlon, maxlon, ysize) #longitudes x-axis

# labels = [left,right,top,bottom]
m.drawparallels(lat_bins, labels=[False,True,True,False])
m.drawmeridians(lon_bins, labels=[True,False,True,False])
"""------------------------------------------------------------------------------------------------------"""


# formatting plot
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='lightgrey', lake_color='white')  

# ais data coords
x, y  = m(ais_data['Longitude'], ais_data['Latitude'])   
#m.scatter(x, y, 0.5, marker='o', color='red')
#plt.title('Cargo ships travelling to Trondheim', fontsize=18)
plt.xlabel('Longitude',fontsize=16)
plt.ylabel('Latitude',fontsize=16)
plt.show()
