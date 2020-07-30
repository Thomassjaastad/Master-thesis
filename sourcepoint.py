import numpy as np
import matplotlib.pyplot as plt

# Function for computing geometric integrals for normal and tangential velcity components

def solvePointGeometry(XP, YP, S, phi, XB, YB):
    """
    Args:
    
    XP: x coordinates at point p. Size: scalar value
    YP: y coordinates at point p. Size: scalar value
    S: vector panel length. Size: [n_panels].
    phi: vector of angles from x-axis to point i + 1. Size: [n_panels].
    XB: x boundary point i of panel i. Size: [n_points = n_panels + 1].
    YB: y boundary point i of panel i. Size: [n_points = n_panels + 1].
    
    Returns:
    Arrays X and Y containing source strength contributions at a point p.
    Used to compute velocities at every grid point. 
    """

    n_panels = len(S)
    X = np.zeros(n_panels)
    Y = np.zeros(n_panels)

    eps = 1e-6
    # normal and tangential velocity calculation at control point i and panel i
    for j in range(n_panels):    
        
        Ax = np.cos(phi[j])  
        Ay = np.sin(phi[j])
        B = (XB[j] - XP)*np.cos(phi[j]) + (YB[j] - YP)*np.sin(phi[j])
        C = (XP - XB[j])**2 + (YP - YB[j])**2
        Dx = XP - XB[j]
        Dy = YP - YB[j]
        E = np.sqrt(C - B**2) + eps
        
        # watch out for angle ambiguity. May not matter as calcluating strengths            
        
        Xterm1 = -0.5*Ax*np.log((S[j]**2 + 2*S[j]*B + C)/C)
        Xterm2 = (Dx+Ax*B)/E*(np.arctan2((S[j] + B), E) - np.arctan2(B, E))
        X[j] = Xterm1 + Xterm2
        if np.isnan(X[j]) == True:
            X[j] = 0
        Yterm1 = -0.5*Ay*np.log((S[j]**2 + 2*S[j]*B + C)/C)
        Yterm2 = (Dy+Ay*B)/E*(np.arctan2((S[j] + B), E) - np.arctan2(B, E))
        Y[j] = Yterm1 + Yterm2        
        if np.isnan(Y[j]) == True:
            Y[j] = 0
        
    return X, Y
