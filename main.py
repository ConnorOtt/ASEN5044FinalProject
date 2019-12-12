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
from scipy.stats import multivariate_normal as mvn
from pprint import pprint
import matplotlib.pyplot as plt

# Import kalman filter classes
from lkf import LKF
from ekf import EKF
from ukf import UKF

# Local imports
from system_def import dt_jac_eval_funcs, ct_nl_funcs, nl_orbit_prop


# -----------------------// Set up system //-----------------------------------

# Initial state
X0 = 6678                                       # [km]
Y0 = 0                                          # [km]
r0 = sqrt(X0**2 + Y0**2)                        # [km]
X0dot = 0                                       # [km/s]
Y0dot = r0 * sqrt(mu/r0**3)                     # [km/s]
x_nom_0 = np.array([X0, X0dot, Y0, Y0dot])      # Nominal full n x 1 nominal state at t_0
P_0 = np.diag([0.001, 0.0001, 0.001, 0.0001])
# dx_0 = mvn(mean=None, cov=P_0).rvs(random_state=13131)
dx_0 = np.array([1e-4, 1e-5, 1e-4, 1e-5])

# Read in measurements from .mat file
data = loadmat('Assignment/orbitdeterm_finalproj_KFdata.mat')

# tdata = data["tvec"].T
Qtrue = data["Qtrue"]
Rtrue = data["Rtrue"]

# Creating Monte-Carlo trajectories
t_0_sim = 0
t_f_sim = 2*pi * sqrt(r0**3 / mu) #* 0.1  # NOTE: The 0.1 of an orbit is for testing
dt_sim = 10  
tvec_sim = np.arange(t_0_sim, t_f_sim, dt_sim)
N = 1 # int(10**(n) / 2) # 5000 total trajectories sounds good I guess
num_step = tvec_sim.shape[0]
h_func = ct_nl_funcs['h']

# Generate N-by-num_steps noise vectors from Q_true (N-by-num_step-by-2 total numbers) 
Q_sim = Qtrue
R_sim = Rtrue
p_noise_dist = mvn(mean=[0, 0], cov=Q_sim)
truth_trajectories = []
truth_measurements = []
# k = 0
for _ in range(N): 
    # Sample some noise for this trajectory
    p_noise = p_noise_dist.rvs(size=num_step)

    # restart the sim
    noisey_traj = [x_nom_0 + dx_0]
    noisey_meas = []
    nl_orbit_prop.set_initial_value(x_nom_0 + dx_0, t_0_sim).set_f_params(None, p_noise)
    for k in range(1, num_step):
        nl_orbit_prop.integrate(tvec_sim[k])
        noisey_traj.append(nl_orbit_prop.y)

        meas, _ = h_func(nl_orbit_prop.y, tvec_sim[k], noise_cov=R_sim)
        noisey_meas.append(meas)

    truth_trajectories.append(noisey_traj)
    truth_measurements.append(noisey_meas)


plt.rcParams['figure.figsize'] = 12, 4*n
fig, ax = plt.subplots(n, 1, sharex=True)
start = 0
end = -1
ax[0].set_title('noisey truth trajectories')
for i in range(n):
    for traj in truth_trajectories:
        state_quant1 = [x[i] for x in traj]
        ax[i]. plot(tvec_sim, state_quant1, '-')


plt.rcParams['figure.figsize'] = 12, 4*p
fig, ax = plt.subplots(p, 1, sharex=True)
start = 0
end = -1
for i in range(p):
    for traj_meas in truth_measurements:
        meas_quant1 = [dy[i] for dy in traj_meas]
        ax[i]. plot(tvec_sim[1:], meas_quant1, '-')
    
plt.show()

# exit(0)  # NOTE: this is here

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



# --------------  //  Verify that EKF works correctly // ----------------------

# Initialize system
system = {
    # Required by KF algo
    "t_0": tvec[0], 
    "x_0": dx_0,
    "P_0": np.diag([10, 1, 10, 1]),
    "Q": Qtrue, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # LKF specific
    "x_nom_0":x_nom_0,
    "dt": 10,
}   

# Instantiate filter for system
lkf = LKF(system)

# Simulate measurements coming in and continuously update the estimate with KF
num_meas = -1   # NOTE: -1 => ONLY RUNS TO SECOND TO LAST ELEMENT APPARENTLY wtf python make up your mind 
for y_k in ydata:
    t_k = y_k["t"]
    lkf.update(t_k, y_k) 

lkf.plot_hist()

# -------------------------- // Tune LKF // -----------------------------------






# ------------- //  Verify that EKF works correctly // ------------------------


# Initialize system
system = {
    # Required by KF algo
    "t_0": tvec[0], 
    "x_0": x_nom_0 + dx_0,
    "P_0": np.diag([10, 1, 10, 1]),
    "Q": Qtrue, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # EKF specific
    "dt": 10,
}   
# Instantiate filter for system
ekf = EKF(system)

# Simulate measurements coming in and continuously update the estimate with KF
for y_k in ydata:
    t_k = y_k["t"]
    ekf.update(t_k, y_k) 

 
ekf.plot_hist()



# -------------------------- // Tune EKF // -----------------------------------
# TODO





# ------------- //  Verify that UKF works correctly // ------------------------
# TODO





# -------------------------- // Tune UKF // -----------------------------------
# TODO




