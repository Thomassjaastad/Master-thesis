import numpy as np

"""
Velocitypotential class sets up a flow environment with different flow charateristics
Representation of flows are: 

Methods:
----------------------------------
Potential  Physical representation
sink    -> end point/destination 
source  -> starting point/ objects 
dipol   -> objects in vicinity
----------------------------------
"""

class Velocity():
    # Instance attributes
    # Flow: sink, dipol, source
    # Potentials at center x0, y0 

    def __init__(self, flow_strength, x0, y0):
        self.flow_strength = flow_strength
        self.x0 = x0
        self.y0 = y0
        
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
