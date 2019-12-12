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
from constants import *
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import pickle

# Import kalman filter classes
from lkf import LKF
from ekf import EKF
from ukf import UKF

# Local imports
from system_def import dt_jac_eval_funcs, ct_nl_funcs, nl_orbit_prop


# -----------------------// Set up system //-----------------------------------

# Initial state

t_0 = 0
X0 = 6678                                       # [km]
Y0 = 0                                          # [km]
X0dot = 0                                       # [km/s]
Y0dot = r0 * np.sqrt(mu/r0**3)                     # [km/s]
x_nom_0 = np.array([r0, X0dot, Y0, Y0dot])      # Nominal full n x 1 nominal state at t_0
P_0 = np.diag([0.001, 0.0001, 0.001, 0.0001])
# dx_0 = mvn(mean=None, cov=P_0).rvs(random_state=13131)
dx_0 = np.array([1e-4, 1e-5, 1e-4, 1e-5])

data = loadmat('Assignment/orbitdeterm_finalproj_KFdata.mat')
Qtrue = data["Qtrue"]
Rtrue = data["Rtrue"]

pickle_in = open(data_dir + "5044data.pickle","rb")
ydata = pickle.load(pickle_in)

pickle_in = open(data_dir + "truth_traj.pickle","rb")
truth_trajectories = pickle.load(pickle_in)

pickle_in = open(data_dir + "truth_meas.pickle","rb")
truth_measurements = pickle.load(pickle_in)



# --------  // Perform NEES/NIS tests on LKF // -----------------

# Initialize system
system = {
    # Required by KF algo
    "t_0": t_0, 
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
    "t_0": t_0, 
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




