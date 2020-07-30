import numpy as np

# Function for computing geometric integrals for normal and tangential velcity components


#TODO:
# Check if angles are computed correctly. Not using beta in program. Is necessary?

def solveGeometricIntegrals(xc, yc, S, phi, beta, XB, YB):
    """
    Args:
    
    xc: vector of x coordinates at control points. Size: [n_panels]
    yc: vector of y coordinates at control points. Size: [n_panels]
    S: vector panel length. Size: [n_panels] 
    phi: vector of angles from x-axis to point i + 1. Size: [n_panels]
    beta: vector of angles from free stream vector to normal vector. Size: [n_panels]
    XB: x boundary point i of panel i. Size: [n_points = n_panels + 1]
    YB: y boundary point i of panel i. Size: [n_points = n_panels + 1]
    
    Returns:
    Matrix I and J containing each contribution from source panel 
    I is used in normal velocity calc
    J is used in tangential velocity calc
    
    Note:
    I and J should only have values on off-diagonals
    I and J must have full rank
    """

    n_panels = len(xc)
    I = np.zeros((n_panels, n_panels))
    J = np.zeros((n_panels, n_panels))
    eps = 1e-10
    # normal and tangential velocity calculation at control point i and panel i
    for i in range(n_panels):
        for j in range(n_panels):
            if j != i:
                An = np.sin(phi[i] - phi[j])  
                
                At = -np.cos(phi[i] - phi[j])

                B = (XB[j]*np.cos(phi[j]) + YB[j]*np.sin(phi[j])
                    - xc[i]*np.cos(phi[j]) - yc[i]*np.sin(phi[j]))
                
                C = (xc[i] - XB[j])**2 + (yc[i] - YB[j])**2
                
                Dn = (XB[j] - xc[i])*np.sin(phi[i]) + (yc[i] - YB[j])*np.cos(phi[i])
                
                Dt = (xc[i] - XB[j])*np.cos(phi[i]) + (yc[i] - YB[j])*np.sin(phi[i])
                arg = C - B**2
                E = np.sqrt(arg) + eps
                if np.isnan(E) == True:
                    print('E is 0')
                
                # watch out for angle ambiguity. May not matter as calculating strengths            
                Iterm1 = 0.5*An*np.log((S[j]**2 + 2*S[j]*B + C)/C)
                Iterm2 = ((Dn-An*B)/E)*(np.arctan2((S[j] + B), E) - np.arctan2(B, E))
                I[i][j] = Iterm1 + Iterm2

                Jterm1 = 0.5*At*np.log((S[j]**2 + 2*S[j]*B + C)/C)
                Jterm2 = ((Dt-At*B)/E)*(np.arctan2((S[j] + B), E) - np.arctan2(B, E))
                J[i][j] = Jterm1 + Jterm2    
            
            if np.isnan(J[i][j]) == True:
                J[i][j] = 0.0
            elif np.isnan(I[i][j]) == True:
                I[i][j] = 0.0
            elif np.isinf(I[i][j]) == True:
                I[i][j] = 0.0
            elif np.isinf(J[i][j]) == True:
                J[i][j] = 0.0
    return I, J
