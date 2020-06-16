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
radius = np.array([0.5, 0.5])
x_center = np.array([0, 0])
y_center = np.array([2, -2])
n_boundary_points = np.array([30, 30])

numb_objs = len(radius)

# scalar values
Vinf = 1      
alpha = np.radians(0)
#circle_points = 100

# creating 2 circle object located at (0,0) and (3,3), respectively
obj0 = createobject.Object(radius[0], x_center[0], y_center[0], 
                          n_boundary_points[0])

obj1 = createobject.Object(radius[1], x_center[1], y_center[1], 
                          n_boundary_points[1])

# create grid
nx, ny  = 50, 50
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
XX, YY = np.meshgrid(x, y)

# plotting circle
#x_coords, y_coords = obj0.create_circle(circle_points)
#x_coords1, y_coords1 = obj1.create_circle(circle_points)

# create circle object with corresponding boundary and control points. 



XB0, YB0 = obj0.boundary_points()
XC0, YC0 = obj0.control_points(XB0, YB0)
panels0, phi0 = obj0.panels(XB0, YB0)
delta0, beta0 = obj0.angles(phi0, alpha)

XB1, YB1 = obj1.boundary_points()
XC1, YC1 = obj1.control_points(XB1, YB1)
panels1, phi1 = obj1.panels(XB1, YB1)
delta1, beta1 = obj1.angles(phi1, alpha)

#panels_tot = np.append(panels0, panels1)
#phi_tot = np.append(phi0, phi1)
#XB_tot =  np.append(XB0, XB1)
#YB_tot =  np.append(YB0, YB1)
#XC_tot = np.append(XC0, XC1)
#YC_tot = np.append(YC0, YC1)
#beta_tot = np.append(beta0, beta1)

# solve Ax = b for source strengths lamda: Integral computation 
I0, J0 = sourcepanel.solveGeometricIntegrals(XC0, YC0, panels0, phi0, beta0, XB0, YB0)
np.fill_diagonal(I0, np.pi)
b0 = -2*np.pi*Vinf*np.cos(beta0)
lamda0 = np.linalg.solve(I0, b0)
#print(np.dot(lamda0, panels0))

I1, J1 = sourcepanel.solveGeometricIntegrals(XC1, YC1, panels1, phi1, beta1, XB1, YB1)
np.fill_diagonal(I1, np.pi)
b1 = -2*np.pi*Vinf*np.cos(beta1)
lamda1 = np.linalg.solve(I1, b1)
#print(np.dot(lamda1, panels1))
#print(lamda0)
#print(lamda1)
panels_tot = np.append(panels1, panels0)
phi_tot = np.append(phi1, phi0)
XB_tot =  np.append(XB1, XB0)
YB_tot =  np.append(YB1, YB0)
XC_tot = np.append(XC1, XC0)
YC_tot = np.append(YC1, YC0)
beta_tot = np.append(beta1, beta0)

# solve Ax = b for source strengths lamda: Integral computation
I, J = sourcepanel.solveGeometricIntegrals(XC_tot, YC_tot, panels_tot, 
                                           phi_tot, beta_tot, XB_tot, YB_tot)
np.fill_diagonal(I, np.pi)
b = -2*np.pi*Vinf*np.cos(beta_tot)
lamda = np.linalg.solve(I, b)

lamda_tot = np.stack((lamda0, lamda1), axis = 1)

#panels_tot = np.stack((panels0, panels1), axis=1)
#phi_tot = np.stack((phi0, phi1), axis=1)
#XB_tot = np.stack((XB0, XB1), axis=1)
#YB_tot = np.stack((YB0, YB1), axis=1)

vx = np.zeros((nx, ny))
vy = np.zeros((nx, ny))

# compute velocites at every grid cell (XP, YP)
for i in range(nx):
    for j in range(ny):
        for k in range(numb_objs):
            XP = XX[i, j]
            YP = YY[i, j]
            X, Y = sourcepoint.solvePointGeometry(XP, YP, panels_tot, phi_tot, XB_tot, YB_tot) #Should be sum of all panels, phi, XB and YB?
            dist0 = np.sqrt((x_center[0] - XP)**2 + (y_center[0] - YP)**2)
            dist1 = np.sqrt((x_center[1] - XP)**2 + (y_center[1] - YP)**2)

            if dist0 < radius[0]:
                vx[i, j] = 0 
                vy[i, j] = 0 
                plt.scatter(XP, YP, color = 'blue', s = 2, zorder = 2)  

            #elif dist0 < radius[0]:
            #    vx[i, j] = 0 
            #    vy[i, j] = 0    
            #    plt.scatter(XP, YP, color = 'red', s = 2, zorder = 2)
            else:
                vx[i, j] = Vinf*np.cos(alpha) + np.dot(lamda, X.T)/(2*np.pi)
                vy[i, j] = Vinf*np.sin(alpha) + np.dot(lamda, Y.T)/(2*np.pi)

                
# starting points for streamlines 
# plt function not good at plotting points 

Y_initpoints = np.linspace(y[0], y[-1], ny)
X_initpoints = x[0]*np.ones(len(Y_initpoints))                            
XY_init   = np.vstack((X_initpoints.T, Y_initpoints.T)).T  
plt.streamplot(XX, YY, vx, vy, linewidth = 0.5, density = 10, color = 'r', 
                   arrowstyle = '-', start_points = XY_init)

#plt.scatter(XX, YY, color = 'green')

plt.title(r'Source Panel method around circle with (AoA = %.1f) ' % (alpha*180/np.pi), fontsize = 17)
#plt.plot(x_coords, y_coords, '-', color = 'blue')
plt.fill(XB0, YB0, color='black')
plt.fill(XB1, YB1, color='black')
plt.xlabel(r'x', fontsize = 16)
plt.ylabel(r'y', fontsize = 16)
plt.axis('Equal')
#plt.legend()
plt.show()
