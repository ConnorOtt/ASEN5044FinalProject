"""
********************************************************************************

ASEN 5044 Statistical Estimation - Final Project: OD

This main script performs the following:
    - Initializes the system and and its initial conditions
    - Implements and tunes a linearized Kalman filter
    - Implements and tunes an extended Kalman filter

Authors:        Keith Covington, Connor Ott
Created:        09 Dec 2019
Last Modified:  09 Dec 2019

********************************************************************************
"""

# The usual imports
import numpy as np
from math import sqrt
from constants import *
from scipy.io import loadmat

# Import kalman filter classes
from lkf import LKF
from ekf import EKF
from ukf import UKF

# Local imports
from system_def import dt_jac_eval_funcs, ct_nl_funcs



####################### Verify that LKF works correctly ########################

# Initial state
X0 = 6678                                       # [km]
Y0 = 0                                          # [km]
r0 = sqrt(X0**2 + Y0**2)                        # [km]
X0dot = 0                                       # [km/s]
Y0dot = r0 * sqrt(mu/r0**3)                     # [km/s]
x_nom_0 = np.array([X0, Y0, X0dot, Y0dot])      # Build full n x 1 intial state vector
dx0 = np.array([0, 0.075, 0, -0.021])           # Initial deviation in Ofer's writeup 

# Read in measurements from .mat file
data = loadmat('Assignment/orbitdeterm_finalproj_KFdata.mat')

# Extract measurements (y) and their corresponding station IDs
yraw = data["ydata"].T
yraw = np.concatenate(yraw, axis=0)
ydata = []
stationIDs = []
for yk in yraw:
    ypacket = {
            "meas": yk[0:3].T,
            "stationID": np.squeeze(yk[-1]).tolist() 
            }
    ydata.append(ypacket)

tdata = data["tvec"].T
Qtrue = data["Qtrue"]
Rtrue = data["Rtrue"]


# Initialize system
# NOTE: ~A lot~ Less of this is random values right now
# NOTE: I fucked with some variable names in some spot and compatibility may be off
system = {
        # Required by KF algo
        "t0": tdata[0], 
        "x0":xnom_0 + dx0,
        "P0": 1000 * np.ones((n,n)),
        "Q": Qtrue, 
        "R": Rtrue, 
        **dt_jac_eval_funcs, 
        **ct_nl_funcs,

        # LKF specific
        "x_nom_0":x_nom_0,
        "dx0": dx0, 
        "dt": 10,
        }

# Instantiate filter for system
lkf = LKF(system)

# Simulate measurements coming in and continuously update the estimate with KF
#for k in range(len(ydata)):
for k in range(1,3):
    print(k)
    tk = tdata[k]
    yk = ydata[k]
    lkf.update(tk, yk)



################################## Tune LKF ####################################
# TODO





####################### Verify that EKF works correctly ########################
# TODO





################################## Tune EKF ####################################
# TODO





####################### Verify that UKF works correctly ########################
# TODO





################################## Tune UKF ####################################
# TODO




