"""
Constants and whatnot

"""

import numpy as np
from numpy import pi

n = 4 # number of states 
m = 2
p = 3
rE = 6378 # km - Earth radius
r0 = rE + 300 # km - orbit radius
delta_t = 10 # s
mu = 398600 # km**3/s**2
omegaE = 2*pi / 86400
I = np.eye(n)
