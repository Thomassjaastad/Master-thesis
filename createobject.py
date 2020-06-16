import numpy as np

class Object():
    """
    Creates object with attributes to approximates objects with line segments.
    Used for panel method fluid mechanics.   
    """
    def __init__(self, radius, x_center, y_center, n_boundary_points):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center
        self.n_boundary_points = n_boundary_points

    def create_circle(self, n_points):
        angles =  np.linspace(0, 2*np.pi, n_points)
        x_comp = self.radius*np.cos(angles) + self.x_center
        y_comp = self.radius*np.sin(angles) + self.y_center
        return x_comp, y_comp

    def boundary_points(self):
        boundary_angles = np.linspace(0, 2*np.pi, self.n_boundary_points)
        XB = self.radius*np.cos(boundary_angles) + self.x_center
        YB = self.radius*np.sin(boundary_angles) + self.y_center

        # fix orientation
        XB = np.flipud(XB)
        YB = np.flipud(YB)
        return XB, YB

    def control_points(self, XB, YB):
        n_panels = self.n_boundary_points - 1
        XC = np.zeros(n_panels)
        YC = np.zeros(n_panels)
        for i in range(n_panels):
            XC[i] = (XB[i] + XB[i + 1])/2   
            YC[i] = (YB[i] + YB[i + 1])/2  
        #XC = np.flipud(XC)
        #YC = np.flipud(YC)
        return XC, YC 
    
    def panels(self, XB, YB):
        panel = np.zeros(self.n_boundary_points - 1)
        phi = np.zeros(self.n_boundary_points - 1)
        for i in range(len(panel)):
            dx = XB[i+1] - XB[i]                  # change in x direction from boundary points 
            dy = YB[i+1] - YB[i]                  # change in y direction from boundary points
            panel[i] = np.sqrt(dx**2 + dy**2)     # panel length (should be equal from circle and points definition)
            phi[i] = np.arctan2(dy,dx)
            if phi[i] < 0:
                phi[i] = phi[i] + 2*np.pi                                       
        return panel, phi
    
    def angles(self, phi, alpha):
        delta = phi + np.pi/2
        beta = delta - alpha
        return delta, beta