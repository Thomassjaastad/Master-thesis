import numpy as np 
import matplotlib.pyplot as plt 
import Flow

# Create streamline plot for vessel planning
# Set out:
# Source -> start point
# Sink -> end point 
# doublet -> island (none - traversable area)

# Set up grid
nx, ny = 30, 30
x = np.linspace(-20, 20, nx)
y = np.linspace(-20, 20, ny)
X, Y = np.meshgrid(x, y)

# Create flow types
source_str = 1
x_source = -14
y_source = -14
flow0 = Flow.VelocityPotential(source_str, x_source, y_source)
u_source, v_source = flow0.source(X, Y)

sink_str = 4
x_sink = 14
y_sink = 13
flow1 = Flow.VelocityPotential(sink_str, x_sink, y_sink)
u_sink, v_sink = flow1.sink(X, Y)

doublet_str0 = 0.5
x_doublet0 = -5
y_doublet0 = 5
flow2 = Flow.VelocityPotential(doublet_str0, x_doublet0, y_doublet0)
u_doublet0, v_doublet0 = flow2.doublet(X, Y)


doublet_str1 = 0.5
x_doublet1 = 1
y_doublet1 = -2
flow3 = Flow.VelocityPotential(doublet_str1, x_doublet1, y_doublet1)
u_doublet1, v_doublet1 = flow3.doublet(X, Y)

# Superposition
u_tot = u_sink + u_source + u_doublet0 + u_doublet1
v_tot = v_sink + v_source + v_doublet0 + v_doublet1


rad = 3
for i in range(nx):
    for j in range(ny):
        insiderad0 = np.sqrt((x_doublet0 - x[i])**2 + (y_doublet0 - y[j])**2)
        insiderad1 = np.sqrt((x_doublet1 - x[i])**2 + (y_doublet1 - y[j])**2)
        
        if insiderad0 <= rad and insiderad1 <= rad:
            u_tot[i, j] = 0
            v_tot[i, j] = 0

#plt.quiver(X, Y, u_tot, v_tot)

plt.streamplot(X, Y, u_tot, v_tot, linewidth = 0.5, density = 2, color = 'b', 
                   arrowstyle = '->')

circle0 = plt.Circle((x_doublet0, y_doublet0), rad, color = 'black', fill = True,zorder=2)
circle1 = plt.Circle((x_doublet1, y_doublet1), rad, color = 'black', fill = True,zorder=2)

ax = plt.gca()
ax.add_artist(circle0)
ax.add_artist(circle1)
#plt.scatter(x_doublet0, y_doublet0, color = 'orange')
plt.scatter(x_sink, y_sink, color = 'red')
plt.scatter(x_source, y_source, color = 'green')
plt.title('Flow around two islands analytic', fontsize = 18)
plt.xlabel('X', fontsize = 16)
plt.ylabel('Y', fontsize = 16)
plt.show()
