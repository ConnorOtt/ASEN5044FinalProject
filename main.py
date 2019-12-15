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
from numpy.linalg import inv
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
import pickle


# Import kalman filter classes
from lkf import LKF
from ekf import EKF
from ukf import UKF

# Local imports
from system_def import dt_jac_eval_funcs, ct_nl_funcs


# -----------------------// Set up system //-----------------------------------


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
    "x_0": dx_est_0,
    "P_0": P_0,
    "Q": Qtrue*100, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # LKF specific
    "x_nom_0":x_nom_0,
    "dt": 10,
}   

report_fields = ['x_full_kp1', 'P_post_kp1', 'y_kp1', 
                'y_pre_est_kp1', 'innov_cov', 'y_post_est_kp1']
num_traj = len(truth_trajectories)

# Get 95% confidence bounds - NIS/NEES
df_NEES = num_traj * n
conf = [0.975, 0.025]
bounds_NEES = chi2.ppf(conf, df_NEES) / num_traj
df_NIS = num_traj * p # this is not p! because measurement size changes from 
bounds_NIS = chi2.ppf(conf, df_NIS) / num_traj

all_NEES = []
all_NIS = [] 
for i in range(num_traj):

    # Instantiate filter for this sim
    lkf = LKF(system)
    truth_meas_i = truth_measurements[i] # from trajectory i
    truth_traj_i = truth_trajectories[i]

    for y_k in truth_meas_i:
        t_k = y_k["t"]
        lkf.update(t_k, y_k) 

    report = lkf.report_hist(report_fields)

    y_vecs = report['y_kp1']
    #y_est_pre = report['y_pre_est_kp1']
    y_est_pre = report['y_post_est_kp1']
    innov_cov = report['innov_cov']

    full_est = report['x_full_kp1']
    state_cov = report['P_post_kp1']

    state_resid_i = [xt for xt, xe in zip(truth_traj_i, full_est)]
    meas_resid_i = [y - y_pre for y, y_pre in zip(y_vecs, y_est_pre)]

    nees_vec_i = [ex.T @ inv(P) @ ex for ex, P in zip(state_resid_i, state_cov)]
    nis_vec_i = [ey.T @ inv(S) @ ey for ey, S in zip(meas_resid_i, innov_cov)]

    all_NEES.append(nees_vec_i)
    all_NIS.append(nis_vec_i)


lkf.plot_hist()

# print(bounds_NEES)
# print(bounds_NIS)


# Average across the simulations
avg_NEES = np.mean(np.array(all_NEES), 0)
avg_NIS = np.mean(np.array(all_NIS), 0)

plt.rcParams['figure.figsize'] = 12, 6
fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].set_title('NEES and NIS tests for LKF')
#ax[0].plot([nees for nees in np.array(all_NEES).T], '.', color='orangered', label='NEES Results')
ax[0].plot(np.array(state_resid_i).T[0], '.')
ax[0].plot(np.array(state_resid_i).T[1], '.')
ax[0].plot(np.array(state_resid_i).T[2], '.')
ax[0].plot(np.array(state_resid_i).T[3], '.')
#ax[0].axhline(bounds_NEES[0], linestyle='--', color='black')
#ax[0].axhline(bounds_NEES[1])#, linestyle='--', color='black')
# ax[0].legend()


#ax[1].plot([nis for nis in np.array(all_NIS).T], '.', color='dodgerblue', label='NIS Results')
ax[1].plot(avg_NIS, '.', color='dodgerblue', label='NIS Results')
ax[1].set_xlabel('time step k')
ax[1].axhline(bounds_NIS[0])#, linestyle='--', color='black')
ax[1].axhline(bounds_NIS[1])#, linestyle='--', color='black')
# ax[1].legend()

plt.show()

exit(0)


# -------------------------- // Tune LKF // ---------------------------------






# ------------- //  Perform NEES/NIS tests on EKF // ------------------------


# Initialize system
system = {
    # Required by KF algo
    "t_0": t_0, 
    "x_0": x_nom_0,
    "P_0": P_0,
    "Q": Qtrue*10, 
    "R": Rtrue*10, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # EKF specific
    "dt": 10,
}

# Instantiate filter for system
report_fields = ['x_post_kp1', 'P_post_kp1', 'y_kp1', 
                'y_pre_kp1', 'innov_cov']
all_NEES = []
all_NIS = [] 
for i in range(num_traj):

    ekf = EKF(system)
    truth_meas_i = truth_measurements[i]
    truth_traj_i = truth_trajectories[i]

    for y_k in truth_meas_i:
        t_k = y_k["t"]
        ekf.update(t_k, y_k) 

    report = ekf.report_hist(report_fields)

    y_vecs = report['y_kp1']
    y_est_pre = report['y_pre_kp1']
    innov_cov = report['innov_cov']

    full_est = report['x_post_kp1']
    state_cov = report['P_post_kp1']

    meas_resid_i = [y - y_pre for y, y_pre in zip(y_vecs, y_est_pre)]
    state_resid_i = [xt - xe for xt, xe in zip(truth_traj_i, full_est)]

    nees_vec_i = [ex.T @ P @ ex for ex, P in zip(state_resid_i, state_cov)]
    nis_vec_i = [ey.T @ S @ ey for ey, S in zip(meas_resid_i, innov_cov)]

    all_NEES.append(nees_vec_i)
    all_NIS.append(nis_vec_i)


# Average across the simulations
NEES_avg = np.mean(np.array(all_NEES), 0)
NIS_avg = np.mean(np.array(all_NIS), 0)

plt.rcParams['figure.figsize'] = 12, 6
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title('NEES and NIS tests for EKF')
ax[0].plot(NEES_avg, '.', color='orangered', label='NEES Results')
ax[0].axhline(bounds_NEES[0])#, '--', color='black')
ax[0].axhline(bounds_NEES[1])#, '--', color='black')
ax[0].legend()

ax[1].plot(NIS_avg, '.', color='dodgerblue', label='NIS Results')
ax[1].axhline(bounds_NIS[0])#, '--', color='black')
ax[1].axhline(bounds_NIS[1])#, '--', color='black')
ax[1].set_xlabel('time step k')
ax[1].legend()

plt.show()


# -------------------------- // Tune EKF // -----------------------------------
# TODO





# ------------- //  Verify that UKF works correctly // ------------------------
# TODO





# -------------------------- // Tune UKF // -----------------------------------
# TODO




