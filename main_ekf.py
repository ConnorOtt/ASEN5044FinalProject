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
from scipy.stats.distributions import chi2
import matplotlib.pyplot as plt
import pickle
from copy import copy

# Import kalman filter classes
from ekf import EKF

# Local imports
from system_def import dt_jac_eval_funcs, ct_nl_funcs


plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 16, 10


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

#--------// run EKF with given dataset //---------------------

# Initialize system
system = {
    # Required by KF algo
    "t_0": t_0, 
    "x_0": x_nom_0 + dx_est_0,
    "P_0": P_0,
    "Q": np.eye(2)*1e-6, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # EKF specific
    "dt": 10,
}

ekf = EKF(system)
ycp = copy(ydata)
while len(ycp) > 0:
    y = ycp.pop(0)
    ekf.update(y['t'], y)

report = ekf.report_hist(['x_post_kp1', 'P_post_kp1'])

fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Estimate and 2$\sigma$ Bounds - EKF')
for i in range(n):
    state_quant1 = [x[i] for x in report['x_post_kp1']]
    state_quant2 = [x[i] + 2*np.sqrt(P[i, i]) for x, P in zip(report['x_post_kp1'], report['P_post_kp1'])]
    state_quant3 = [x[i] - 2*np.sqrt(P[i, i]) for x, P in zip(report['x_post_kp1'], report['P_post_kp1'])]

    ax[i].plot(state_quant1, '-', color='dodgerblue', label='Estimate Error')
    ax[i].plot(state_quant2, '--', color='black')
    ax[i].plot(state_quant3, '--', color='black')

    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].plot([None], '--', label='2$\sigma$ Bounds')
ax[0].legend(loc='upper left')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'ekf_dataset_est.png')

plt.show()

# exit(0)

# ------------- //  Perform NEES/NIS tests on EKF // ------------------------

num_traj = len(truth_trajectories)

# Get 95% confidence bounds - NIS/NEES
df_NEES = num_traj * n
conf = [0.975, 0.025]
bounds_NEES = chi2.ppf(conf, df_NEES) / num_traj
df_NIS = num_traj * p # this is not p! because measurement size changes from 
bounds_NIS = chi2.ppf(conf, df_NIS) / num_traj



# Initialize system
system = {
    # Required by KF algo
    "t_0": t_0, 
    "x_0": x_nom_0 + dx_est_0,
    "P_0": P_0,
    "Q": np.eye(2)*10e-10, 
    "R": Rtrue, 
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

    state_resid_i = [xt - xe for xt, xe in zip(truth_traj_i[1:], full_est)]
    meas_resid_i = [y - y_pre for y, y_pre in zip(y_vecs, y_est_pre)]

    nees_vec_i = [ex.T @ inv(P) @ ex for ex, P in zip(state_resid_i, state_cov)]
    nis_vec_i = []
    for ey, S in zip(meas_resid_i, innov_cov):
        epsilon = ey.T @ inv(S) @ ey
        if ey.shape[0] > p:
            epsilon = epsilon / 2
        nis_vec_i.append(epsilon)
    
    all_NEES.append(nees_vec_i)
    all_NIS.append(nis_vec_i)


# Average across the simulations
NEES_avg = np.mean(np.array(all_NEES), 0)
NIS_avg = np.mean(np.array(all_NIS), 0)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title('NEES and NIS tests for EKF')
ax[0].plot(NEES_avg, '.', color='orangered', label='NEES Results')
ax[0].axhline(bounds_NEES[0], linestyle='--', color='black')
ax[0].axhline(bounds_NEES[1], linestyle='--', color='black')
ax[0].set_ylim([0, 10])
ax[0].autoscale(enable=True, axis='x', tight=True)
ax[0].legend()

ax[1].plot(NIS_avg, '.', color='dodgerblue', label='NIS Results')
ax[1].axhline(bounds_NIS[0], linestyle='--', color='black')
ax[1].axhline(bounds_NIS[1], linestyle='--', color='black')
ax[1].set_xlabel('time step k')
ax[1].set_ylim([0, 10])
ax[1].autoscale(enable=True, axis='x', tight=True)
ax[1].legend()
fig.savefig(fig_dir + 'NEESNIS_ekf_N' + str(num_traj) + '.png')


fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Errors and 2$\sigma$ Bounds - EKF')
for i in range(n):
    state_quant1 = [x[i] for x in state_resid_i]
    state_quant2 = [(2*np.sqrt(P[i, i]), -2*np.sqrt(P[i, i])) for P in report['P_post_kp1']]
    ax[i].plot(state_quant1, '-', color='dodgerblue', label='Estimate Error')
    ax[i].plot(state_quant2, '--', color='black')
    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].plot([None], '--', label='2$\sigma$ Bounds')
ax[0].legend(loc='upper right')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'ekf_estimate_th.png')


plt.show()

