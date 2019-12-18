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
from pprint import pprint

# Import kalman filter classes
from lkf import LKF

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


# ---------// run the LKF with the data provided //--------------
Q = np.eye(2) * 1e-9
system = {
    # Required by KF algo
    "t_0": t_0, 
    "x_0": dx_est_0,
    "P_0": P_0,
    "Q": Qtrue, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # LKF specific
    "x_nom_0":x_nom_0,
    "dt": 10,
}

lkf = LKF(system)
ycp = copy(ydata)
while len(ycp) > 0:
    y = ycp.pop(0)
    lkf.update(y['t'], y)

report = lkf.report_hist(['x_full_kp1', 'P_post_kp1'])

fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Errors and 2$\sigma$ Bounds - LKF')
for i in range(n):
    state_quant1 = [x[i] for x in report['x_full_kp1']]
    state_quant2 = [x[i] + 2*np.sqrt(P[i, i]) for x, P in zip(report['x_full_kp1'], report['P_post_kp1'])]
    state_quant3 = [x[i] - 2*np.sqrt(P[i, i]) for x, P in zip(report['x_full_kp1'], report['P_post_kp1'])]

    ax[i].plot(state_quant1, '-', color='dodgerblue', label='Estimate Error')
    ax[i].plot(state_quant2, '--', color='black')
    ax[i].plot(state_quant3, '--', color='black')

    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].plot([None], '--', label='2$\sigma$ Bounds')
ax[0].legend(loc='upper left')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'lkf_dataset_est.png')


# --------  // Perform NEES/NIS tests on LKF // -----------------
report_fields = ['x_full_kp1', 'P_post_kp1', 'y_kp1', 
                'y_pre_est_kp1', 'innov_cov', 'x_nom_kp1',
                'y_nom_kp1']
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
    y_est_pre = report['y_pre_est_kp1']
    innov_cov = report['innov_cov']

    full_est = report['x_full_kp1']
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
avg_NEES = np.mean(np.array(all_NEES), 0)
avg_NIS = np.mean(np.array(all_NIS), 0)


fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title('NEES and NIS tests for LKF, N = {}'.format(num_traj))
ax[0].plot(avg_NEES, '.', color='orangered', label='NEES Results')
ax[0].axhline(bounds_NEES[0], linestyle='--', color='black')
ax[0].axhline(bounds_NEES[1], linestyle='--', color='black')
ax[0].set_ylim([0, 10])
ax[0].autoscale(enable=True, axis='x', tight=True)
ax[0].legend()

ax[1].plot(avg_NIS, '.', color='dodgerblue', label='NIS Results')
ax[1].set_xlabel('time step k')
ax[1].axhline(bounds_NIS[0], linestyle='--', color='black')
ax[1].axhline(bounds_NIS[1], linestyle='--', color='black')
ax[1].set_ylim([0, 10])
ax[1].autoscale(enable=True, axis='x', tight=True)
ax[1].legend()

fig.savefig(fig_dir + 'NEESNIS_lkf_N' + str(num_traj) + 'Q{:.1E}.png'.format(Q[0, 0]))


# Plot that last one to get an idea of the state and measurement residuals
fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('Typical Noisey Truth Trajectory')
for i in range(n):
    state_quant1 = [x[i] for x in truth_traj_i]
    state_quant2 = [x[i] for x in report['x_nom_kp1']]
    ax[i].plot(state_quant1, '-', color='dodgerblue', label='Noisey')
    ax[i].plot(state_quant2, '-', color='orangered', label='Noiseless')
    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].legend(loc='upper right')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'noisey_truth.png')




fig, ax = plt.subplots(p, 1, sharex=True)
ax[0].set_title('Typical Noisey Measurements')


# Not really sure what I want to do with this but it seemed like
# a good idea like 10 minutes ago
meas_by_station = {k:[] for k in range(1, 13)}
for y in truth_meas_i:
    for k in meas_by_station:
        # Give the station None by default
        meas_by_station[k].append([None for _ in range(p)])

        # Go through station ids in y and replace those in meas_by_station if they match
        for i in range(len(y['stationID'])):
            if y['stationID'][i] == k:
                meas_by_station[k][-1] = y['meas'][p*i:p*i+p]           


for i in range(p):
    quant1 = [y['meas'][i] for y in truth_meas_i]
    quant2 = [y[i] for y in report['y_nom_kp1']]
    ax[i].plot(quant1, '-', color='dodgerblue', label='Noisey')
    ax[i].plot(quant2, '-', color='orangered', label='Noiseless')
    ax[i].set_ylabel('$y_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[-1].set_xlabel('time step')
ax[0].legend(loc='upper right')
fig.savefig(fig_dir + 'noisey_meas.png')


# typical LKF estimate (zoomed)
fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Errors and 2$\sigma$ Bounds - LKF')
for i in range(n):
    state_quant1 = [x[i] for x in state_resid_i]
    state_quant2 = [(2*np.sqrt(P[i, i]), -2*np.sqrt(P[i, i])) for P in report['P_post_kp1']]
    ax[i].plot(state_quant1[:100], '-', color='dodgerblue', label='Estimate Error')
    ax[i].plot(state_quant2[:100], '--', color='black')
    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].plot([None], '--', label='2$\sigma$ Bounds')
ax[0].legend(loc='upper right')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'lkf_estimate_th_ZOOM.png')

# typical LKF estimate
fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Errors and 2$\sigma$ Bounds - LKF')
for i in range(n):
    state_quant1 = [x[i] for x in state_resid_i]
    state_quant2 = [(2*np.sqrt(P[i, i]), -2*np.sqrt(P[i, i])) for P in report['P_post_kp1']]
    ax[i].plot(state_quant1, '-', color='dodgerblue', label='Estimate Error')
    ax[i].plot(state_quant2, '--', color='black')
    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].plot([None], '--', label='2$\sigma$ Bounds')
ax[0].legend(loc='upper left')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'lkf_estimate_th.png')


plt.show()

