import numpy as np
import matplotlib.pyplot as plt 
import createobject
import sourcepanel
import sourcepoint

# create circle to simulate flow around object
"""
# Parameters:
######################################################################################################################
# ------------------------------------------------------------------------------------------------------------------ #
# XB, YB: array(size: n_boundary_points) of boundary points which define start and end points of panel in space      #
# XC, YC: array(size: n_boundary_points) of control points, located at center of each panel                          # 
# panels: array(size: n_boundary_points - 1) type of panel lengths                                                   #   
# alpha: Angle of Attack (AoA) of freestream velocity                                                                #   
# delta: array(size: n_boundary_points - 1) of angles from x-axis to normal array                                    #    
# beta: array(size: n_boundary_points - 1) of angles from AoA to normal array                                        #
# ------------------------------------------------------------------------------------------------------------------ #   
######################################################################################################################
"""

# Init class Object variables
radius = np.array([1, 1])
x_center = np.array([0, 3])
y_center = np.array([0, 3])
n_boundary_points = np.array([300, 9])

numb_objs = len(radius)

# scalar values
Vinf = 10        
alpha = np.radians(0)
circle_points = 100

obj = createobject.Object(radius[0], x_center[0], y_center[0], 
                          n_boundary_points[0])

# create grid
nx, ny  = 30, 30
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
XX, YY = np.meshgrid(x, y)

# plotting circle
x_coords, y_coords = obj.create_circle(circle_points)

# create circle object with corresponding boundary and control points. 
XB, YB = obj.boundary_points()
XC, YC = obj.control_points(XB, YB)
panels, phi = obj.panels(XB, YB)
delta, beta = obj.angles(phi, alpha)
# solve Ax = b for source strengths lamda: Integral computation 
I, J = sourcepanel.solveGeometricIntegrals(XC, YC, panels, phi, beta, XB, YB)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta)
print(np.linalg.matrix_rank(I))
lamda = np.linalg.solve(I, b)
print(lamda)
vx = np.zeros((nx, ny))
vy = np.zeros((nx, ny))
# compute velocites at every grid cell (XP, YP)
for i in range(nx):
    for j in range(ny):
        XP = XX[i, j]
        YP = YY[i, j]
        X, Y = sourcepoint.solvePointGeometry(XP, YP, panels, phi, XB, YB)
        dist = np.sqrt((XP - x_center[0])**2 + (YP - y_center[0])**2) 
        if dist < radius[0]:
            vx[i, j] = 0 
            vy[i, j] = 0    
        else:
            vx[i, j] = Vinf*np.cos(alpha) + np.dot(lamda, X.T)/(2*np.pi)
            vy[i, j] = Vinf*np.sin(alpha) + np.dot(lamda, Y.T)/(2*np.pi)

# starting points for streamlines 
# plt function not good at plotting points 
Y_initpoints = np.linspace(y[0], y[-1], ny)
X_initpoints = x[0]*np.ones(len(Y_initpoints))                            
XY_init   = np.vstack((X_initpoints.T, Y_initpoints.T)).T  

plt.streamplot(XX, YY, vx, vy, linewidth=0.5, density=10, color='r', 
                   arrowstyle='-', start_points = XY_init)
plt.title(r'Source Panel method around circle with (AoA = %.1f) ' % (alpha*180/np.pi), fontsize = 17)
#plt.plot(x_coords, y_coords, '-', color = 'blue')
plt.fill(XB,YB, color='black')
plt.xlabel(r'x', fontsize = 16)
plt.ylabel(r'y', fontsize = 16)
plt.axis('Equal')
#plt.legend()
plt.show()
