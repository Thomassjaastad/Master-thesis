import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(-10,10,1000)

Cs = [1,2,3,4]

def y(x, k, psi):
    param = k/(2*psi)
    return np.sqrt(param**2 - x**2) - param

for stream in Cs:
    plt.plot(x, y(x, 1, stream))

plt.show()
