import numpy as np

def control_points(XB, YB):
    n_panels = len(XB) - 1
    XC = np.zeros(n_panels)
    YC = np.zeros(n_panels)
    for i in range(n_panels):
        XC[i] = (XB[i] + XB[i + 1])/2   
        YC[i] = (YB[i] + YB[i + 1])/2  
    #XC = np.flipud(XC)
    #YC = np.flipud(YC)
    return XC, YC 

def panels(XB, YB):
    panel = np.zeros(len(XB) - 1)
    phi = np.zeros(len(XB) - 1)
    for i in range(len(panel)):
        dx = XB[i+1] - XB[i]                  # change in x direction from boundary points 
        dy = YB[i+1] - YB[i]                  # change in y direction from boundary points
        panel[i] = np.sqrt(dx**2 + dy**2)     # panel length (should be equal from circle and points definition)
        phi[i] = np.arctan2(dy, dx)
        if phi[i] < 0:
            phi[i] = phi[i] + 2*np.pi                                       
    return panel, phi

def angles(phi, alpha):
    delta = phi + np.pi/2
    beta = delta - alpha
    return delta, beta