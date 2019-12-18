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
from ukf import UKF

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

test_traj = truth_trajectories[0]
test_meas = truth_measurements[0]
num_traj = len(truth_trajectories)


print(dx_est_0)
matprint(P_0)
# exit(0)

#--------// run UKF with given dataset //---------------------
# Initialize system
Q = np.eye(2)*1e-9
system = {
    # Required by KF algo
    "t_0": t_0, 
    "x_0": x_nom_0 + dx_est_0,
    "P_0": P_0,
    "Q": Q, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # EKF specific
    "dt": 10,
    'a': 1.5,  # methinks about a=0.5 is 1 std
    'b': 2,
    'k': 0,
}


# First do it with a the given data
ukf = UKF(system)
ycp = copy(ydata)
while len(ycp) > 0:
    y = ycp.pop(0)
    ukf.update(y['t'], y)

report = ukf.report_hist(['x_post_kp1', 'P_post_kp1'])

fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Estimate and 2$\sigma$ Bounds - UKF')
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
fig.savefig(fig_dir + 'ukf_dataset_est.png')


# NEES and NIS Tests
NIS_all = []
NEES_all = []
count = 1
for i in range(num_traj):

    truth_meas_i = truth_measurements[i]
    truth_traj_i = truth_trajectories[i]

    print('trajectory {}'.format(count))
    ukf = UKF(system)
    for y in truth_meas_i:
        # print(y['meas'])
        ukf.update(y['t'], y)

    report = ukf.report_hist(['nis', 'P_post_kp1', 
        'x_post_kp1'])
    nis = report['nis']
    est_th = report['x_post_kp1']
    P_th = report['P_post_kp1']

    # calculate NEES values off the truth
    state_resid_i = [xt - xe for xt, xe in zip(truth_traj_i[1:], est_th)]
    nees = [ex.T @ inv(P) @ ex for ex, P in zip(state_resid_i, P_th)]

    NIS_all.append(nis)
    NEES_all.append(nees)
    count += 1

NIS_avg = np.nanmean(np.array(NIS_all), 0)
NEES_avg = np.nanmean(np.array(NEES_all), 0)

conf = [0.975, 0.025]
bounds_NIS = chi2.ppf(conf, num_traj*p) / num_traj
bounds_NEES = chi2.ppf(conf, num_traj*n) / num_traj


fig, ax = plt.subplots(2, 1)
ax[0].set_title('NEES and NIS tests for UKF, N = {}'.format(num_traj))
ax[0].plot(NEES_avg, '.', color='orangered')
ax[0].axhline(bounds_NEES[0], linestyle='--', color='black')
ax[0].axhline(bounds_NEES[1], linestyle='--', color='black')
ax[0].autoscale(enable=True, axis='x', tight=True)
ax[0].set_ylim([0, 10])
ax[0].set_ylabel('NEES Value')

ax[1].plot(NIS_avg, '.', color='dodgerblue')
ax[1].axhline(bounds_NIS[0], linestyle='--', color='black')
ax[1].axhline(bounds_NIS[1], linestyle='--', color='black')
ax[1].autoscale(enable=True, axis='x', tight=True)
ax[1].set_ylim([0, 10])
ax[1].set_xlabel('time step, k')
ax[1].set_ylabel('NIS Value')
fig.savefig(fig_dir + 'NEESNIS_ukf_N' + str(num_traj) + 'Q{:.1E}.png'.format(Q[0, 0]))


fig, ax = plt.subplots(n, 1, sharex=True)
ax[0].set_title('State Errors and 2$\sigma$ Bounds - UKF')
for i in range(n):
    state_quant1 = [x[i] for x in state_resid_i]
    state_quant2 = [(2*np.sqrt(P[i, i]), -2*np.sqrt(P[i, i])) for P in P_th]
    ax[i].plot(state_quant1, '-', color='dodgerblue', label='Estimate Error')
    ax[i].plot(state_quant2, '--', color='black')
    ax[i].set_ylabel('$x_{%d}$' % (i+1))
    ax[i].autoscale(enable=True, axis='x', tight=True)
ax[0].plot([None], '--', label='2$\sigma$ Bounds')
ax[0].legend(loc='upper right')
ax[-1].set_xlabel('time step')
fig.savefig(fig_dir + 'ukf_estimate_th.png')


plt.show()
