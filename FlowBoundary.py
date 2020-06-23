import numpy as np

"""
Args:
Boundary points collected from get_segments(): XB[n_panels + 1, 1], YB[n_panels + 1, 1]

Args calculating angles:
(angle of attack of freestream velocity): alpha[n_panels, 1] 
angle from x-axis to panel: phi[n_panels, 1] 

Returns: 
Control points XC,YC at each panel [n panels, 1]
Each panel length with corresponding angle phi [n panels, 1]
Angles delta [n panels, 1] and beta[n panels, 1]   
"""

def control_points(XB, YB):
    n_panels = len(XB) - 1
    XC = np.zeros(n_panels)
    YC = np.zeros(n_panels)
    for i in range(n_panels):
        XC[i] = (XB[i] + XB[i + 1])/2   
        YC[i] = (YB[i] + YB[i + 1])/2  
    return XC, YC 

def panels(XB, YB):
    panel = np.zeros(len(XB) - 1)
    phi = np.zeros(len(XB) - 1)
    for i in range(len(panel)):
        dx = XB[i+1] - XB[i]                  # change in x direction from boundary points 
        dy = YB[i+1] - YB[i]                  # change in y direction from boundary points
        panel[i] = np.sqrt(dx**2 + dy**2)     # panel length (should be equal from circle and points definition)
        phi[i] = np.arctan2(dy, dx)
        if phi[i] < 0:                        # angles in fourth quadrant. Must point outwards
            phi[i] = phi[i] + 2*np.pi                                       
    return panel, phi

def angles(phi, alpha):
    delta = phi + np.pi/2
    beta = delta - alpha
    return delta, beta
