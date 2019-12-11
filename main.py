"""
ASEN 5044 Statistical Estimation - Final Project: OD

This main script performs the following:
    - Initializes the system and and its initial conditions
    - Implements and tunes a linearized Kalman filter
    - Implements and tunes an extended Kalman filter

Authors:        Keith Covington, Connor Ott
Created:        09 Dec 2019
Last Modified:  09 Dec 2019

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
x_nom_0 = np.array([X0, Y0, X0dot, Y0dot]).T      # Build full n x 1 intial state vector
dx_0 = np.array([0, 0.075, 0, -0.021]).T           # Initial deviation in Ofer's writeup 

# Read in measurements from .mat file

data = loadmat('Assignment/orbitdeterm_finalproj_KFdata.mat')

# Extract measurements (y) and their corresponding station IDs
ydata = []
yraw = data['ydata'][0]
for y_k in yraw[1:]: # Ignoring measurement 0
    if y_k.size is 0:
        meas, ids = (None, None)
    else:   
        # Make long measurements
        meas, ids = zip(*[(y_k[:3, i], y_k[3, i]) for i in range(int(y_k.shape[1]))])
        meas = np.concatenate(meas)

    y_packet = {
        'meas':meas,
        'stationID':ids
    }
    ydata.append(y_packet)

tdata = data["tvec"].T
Qtrue = data["Qtrue"]
Rtrue = data["Rtrue"]

# Initialize system
# NOTE: ~A lot~ Less of this is random values right now
system = {
    # Required by KF algo
    "t_0": tdata[0], 
    "x_0":x_nom_0 + dx_0,
    "P_0": 1 * np.eye(n),
    "Q": Qtrue, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # LKF specific
    "x_nom_0":x_nom_0,
    "dx_0": dx_0, 
    "dt": 10,
}   

# Instantiate filter for system
lkf = LKF(system)

# Simulate measurements coming in and continuously update the estimate with KF
# for k in range(len(ydata)):
num_meas = 75
for t_k, y_k in zip(tdata[:num_meas], ydata[:num_meas]):
    # print(t_k, y_k)
    lkf.update(t_k, y_k)

out = lkf.report_hist(['x_post_kp1', 'P_post_kp1'])

lkf.plot_hist()

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




