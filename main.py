"""
ASEN 5044 Statistical Estimation - Final Project: OD

This main script performs the following:
    - Initializes the system and and its initial conditions
    - Implements and tunes a linearized Kalman filter
    - Implements and tunes an extended Kalman filter

Authors:        Keith Covington, Connor Ott
Created:        09 Dec 2019
Last Modified:  11 Dec 2019

"""

# The usual imports
import numpy as np
from math import sqrt
from constants import *
from scipy.io import loadmat
from pprint import pprint

# Import kalman filter classes
from lkf import LKF
from ekf import EKF
from ukf import UKF

# Local imports
from system_def import dt_jac_eval_funcs, ct_nl_funcs




############################### Set up system  #################################

# Initial state
X0 = 6678                                       # [km]
Y0 = 0                                          # [km]
r0 = sqrt(X0**2 + Y0**2)                        # [km]
X0dot = 0                                       # [km/s]
Y0dot = r0 * sqrt(mu/r0**3)                     # [km/s]
x_nom_0 = np.array([X0, X0dot, Y0, Y0dot])      # Build full n x 1 intial state vector
dx_0 = np.array([0, 0.075, 0, -0.021])          # Initial deviation in Ofer's writeup 

# Read in measurements from .mat file
data = loadmat('Assignment/orbitdeterm_finalproj_KFdata.mat')

# Extract measurements (y) and their corresponding station IDs
ydata = []
yraw = data['ydata'][0]
tvec = data['tvec'].reshape((-1, 1))
for y_k, t_k in zip(yraw[1:], tvec[1:]): # Ignoring measurement 0
    if y_k.size == 0:
        meas, ids = (None, None)
    else:   
        # Make long measurements
        meas, ids = zip(*[(y_k[:3, i], y_k[3, i]) for i in range(int(y_k.shape[1]))])
        meas = np.concatenate(meas)

    y_packet = {
        'meas':meas,
        'stationID':ids,
        't':t_k[0]
    }
    ydata.append(y_packet)

tdata = data["tvec"].T
Qtrue = data["Qtrue"]
Rtrue = data["Rtrue"]


# Create "truth" data





####################### Verify that LKF works correctly ########################

# Initialize system
# NOTE: Less of this is random values right now
system = {
    # Required by KF algo
    "t_0": tdata[0], 
    "x_0": dx_0,
    "P_0": np.diag([10, 1, 10, 1]),
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
num_meas = -1   # NOTE: -1 => ONLY RUNS TO SECOND TO LAST ELEMENT APPARENTLY wtf python make up your mind 
#for t_k, y_k in zip(tdata[:num_meas], ydata[:num_meas]):
for y_k in ydata:
    t_k = y_k["t"]
    lkf.update(t_k, y_k) 
    # NOTE: could just pass in yk only since it has the tk in it...idk makes the KF class slightly more specific though. Feel free to change if you want



"""From here to next docstring is just 'splorin junk"""
# for y in ydata[19:22]:
#     pprint(y)

"""
out = lkf.report_hist(['y_kp1', 
                        'y_nom_kp1',
                        'dy_update', 
                        'dy_est_kp1',
                        'x_post_kp1', 
                        'dy_nom_kp1', 
                        'pre_fit_residual', 
                        'y_pre_est_kp1',
                        'y_post_est_kp1',
                        'x_update', 
                        'x_full_kp1'])

#pprint(lkf.t_hist)

# for pfr in out['x_post_kp1'][200:300]:
#     print(pfr)

# Plot measurement predictions and nominal
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12, 4*p
fig, ax = plt.subplots(p, 1, sharex=True)
start = 0
end = -1
for i in range(p):
    meas_quant1 = [dy[i] for dy in out['y_pre_est_kp1'][start:end]]
    meas_quant2 = [dy[i] for dy in out['y_post_est_kp1'][start:end]]
    meas_quant3 = [dy[i] for dy in out['y_kp1'][start:end]]

    ax[i]. plot(lkf.t_hist[start+1:end], meas_quant3, '-', color='green', label='y')
    ax[i]. plot(lkf.t_hist[start+1:end], meas_quant1, '-', color='orangered', label='y prefit est')
    #ax[i]. plot(lkf.t_hist[start+1:end], meas_quant2, '-', color='dodgerblue', label='y postfit est')
    
ax[0].legend()
# # plt.show()


plt.rcParams['figure.figsize'] = 12, 4*n
# fig, ax = plt.subplots(n, 1, sharex=True)
# start = 0
# end = -1
# for i in range(n):
#     state_quant1 = [x[i] for x in out['x_post_kp1'][start:end]]
#     # state_quant2 = [x[i] for dy in out['x_post_kp1'][start:end]]

#     ax[i]. plot(lkf.t_hist[start+1:end], state_quant1, '-', color='orangered')
#     # ax[i]. plot(lkf.t_hist[start:end], post_state, '-', color='dodgerblue')


# Plot measurement predictions and nominal
fig, ax = plt.subplots(n, 1, sharex=True)
start = 0
end = -1
ax[0].set_title('$x_{nom} + \delta x$')
for i in range(n):
    state_quant1 = [x[i] for x in out['x_full_kp1'][start:end]]
    # state_quant2 = [x[i] for dy in out['x_post_kp1'][start:end]]

    ax[i]. plot(lkf.t_hist[start+1:end], state_quant1, '-', color='orangered')
    # ax[i]. plot(lkf.t_hist[start:end], post_state, '-', color='dodgerblue')

# plt.show()

lkf.plot_hist()
"""

"""end LKF 'splorin"""


################################## Tune LKF ####################################
# TODO





####################### Verify that EKF works correctly ########################

# Initialize system
system = {
    # Required by KF algo
    "t_0": tdata[0], 
    "x_0": x_nom_0,
    "P_0": np.diag([10, 1, 10, 1]),
    "Q": Qtrue, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # EKF specific
    "x_nom_0":x_nom_0,
    "dx_0": dx_0, 
    "dt": 10,
}   
# Instantiate filter for system
ekf = EKF(system)

# Simulate measurements coming in and continuously update the estimate with KF
for y_k in ydata:
    t_k = y_k["t"]
    ekf.update(t_k, y_k) 



"""
out = ekf.report_hist(['y_kp1', 
                        'x_post_kp1', 
                        'pre_fit_residual', 
                        'y_pre_est_kp1',
                        'y_post_est_kp1',
                        'x_full_kp1'])

# Plot measurement predictions and nominal
fig, ax = plt.subplots(n, 1, sharex=True)
start = 0
end = -1
ax[0].set_title('$x_{nom} + \delta x$')
for i in range(n):
    state_quant1 = [x[i] for x in out['x_full_kp1'][start:end]]
    state_quant2 = [x[i] for dy in out['x_post_kp1'][start:end]]

    ax[i]. plot(ekf.t_hist[start+1:end], state_quant1, '-', color='orangered')
    ax[i]. plot(ekf.t_hist[start:end], post_state, '-', color='dodgerblue')

# plt.show()

"""

ekf.plot_hist()



################################## Tune EKF ####################################
# TODO





####################### Verify that UKF works correctly ########################
# TODO





################################## Tune UKF ####################################
# TODO




