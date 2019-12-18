"""
Constants and whatnot

"""

import numpy as np
from numpy import pi
from scipy.stats import multivariate_normal as mvn

n = 4 # number of states 
m = 2
p = 3
rE = 6378 # km - Earth radius
r0 = rE + 300 # km - orbit radius
delta_t = 10 # s
mu = 398600 # km**3/s**2
omegaE = 2*pi / 86400
I = np.eye(n)
T = 2*pi * np.sqrt(r0**3 / mu)


# Intiial state
t_0 = 0
X0 = r0                                         # [km]
Y0 = 0                                          # [km]
X0dot = 0                                       # [km/s]
Y0dot = r0 * np.sqrt(mu/r0**3)                  # [km/s]
x_nom_0 = np.array([X0, X0dot, Y0, Y0dot])      # Nominal full n x 1 nominal state at t_0
P_0 = np.diag([0.01, 0.001, 0.01, 0.001])
dx_0 = mvn(mean=None, cov=P_0).rvs(random_state=13131)
dx_est_0 = dx_0 


# Not constants but useful/helpful
data_dir = './Data/'
fig_dir = './Report/Figures/'

def matprint(mat, fmt="g"): 
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        # https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
