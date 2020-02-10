import numpy as np

"""

Velocitypotential class sets up a flow environment with different flow charateristics
Representation of flows are: 

Methods:
--------
sink    -> end point/destination 
source  -> starting point 
dipol   -> objects in vicinity
uniform -> stream direction 
"""

class VelocityPotential():
    # Instance attributes
    # Flow: dipol, uniform, source, sink
    # Potentials at center x0, y0 
    #------------------##### To do: ######----------------------
    #---------------------- errors: ----------------------------
    #-----------------------------------------------------------
    # 1. Problem in areas where equations are divided by zero 
    # 2. How to calculate strengths for a producing a good model
    #-----------------------------------------------------------

    def __init__(self, flow_strength, x0, y0):
        self.flow_strength = flow_strength
        self.x0 = x0
        self.y0 = y0
        

    def uniform(self, x_Str, y_Str):
        """
        creates a 2D uniform flow
        returns velocity components u, v  
        """
        u_uniform = x_Str
        v_uniform = y_Str
        return u_uniform, v_uniform

    def sink(self, x, y):
        """
        creates a sink stream given a environment of x and y vectors 
        returns velocity components u, v 
        """
        u_sink = -self.flow_strength*(x - self.x0)/(2*np.pi*((x - self.x0)**2 + (y - self.y0)**2))
        v_sink = -self.flow_strength*(y - self.y0)/(2*np.pi*((x - self.x0)**2 + (y - self.y0)**2))
        return u_sink, v_sink
        
    def source(self, x, y):
        """
        create source with initial values for center
        returns velocity components u, v 
        """
        u_source = self.flow_strength*(x - self.x0)/(2*np.pi*((x - self.x0)**2 + (y - self.y0)**2))
        v_source = self.flow_strength*(y - self.y0)/(2*np.pi*((x - self.x0)**2 + (y - self.y0)**2))
        return u_source, v_source

    def doublet(self, x, y):
        """
        create doublet with initial values for center
        returns velocity components u, v 
        """
        u_dipol = self.flow_strength*((y - self.y0)**2 - (x - self.x0)**2)/((x - self.x0)**2 + (y - self.y0)**2)**2 
        v_dipol = -(2*self.flow_strength*(y - self.y0)*(x - self.x0))/((x - self.x0)**2 + (y - self.y0)**2)**2 
        return u_dipol, v_dipol


    #Not used yet
    def superposition(u, v):
        """
        u and v are vectors 
        finialize environment by adding velocity components/superposition principle 
        returns final u, v to be plotted 
        """
        u_tot = np.sum(u, axis=0)
        v_tot = np.sum(v, axis=0)
        return u_tot, v_tot
