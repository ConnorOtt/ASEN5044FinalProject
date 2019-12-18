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

#--------// run UKF with given dataset //---------------------
# Initialize system
system = {
    # Required by KF algo
    "t_0": t_0, 
    "x_0": x_nom_0 + dx_est_0,
    "P_0": P_0,
    "Q": np.eye(2)*1e-9, 
    "R": Rtrue, 
    **dt_jac_eval_funcs, 
    **ct_nl_funcs,

    # EKF specific
    "dt": 10,
    'a': 1,  # methinks about a=0.5 is 1 std
    'b': 2,
    'k': 0,
}


NIS_all = []
count = 1
for truth_meas in truth_measurements:
    print('trajectory {}'.format(count))
    ukf = UKF(system)
    for y in ydata:
        # print(y['meas'])
        ukf.update(y['t'], y)

    # NIS test baby!
    nis = ukf.report_hist(['nis'])['nis']
    NIS_all.append(nis)
    count += 1

NIS_avg = np.nanmean(np.array(NIS_all), 0)

conf = [0.975, 0.025]
bounds_NIS = chi2.ppf(conf, num_traj*p) / num_traj

fig, ax = plt.subplots(1, 1)
ax.plot(NIS_avg, '.', color='dodgerblue')
ax.set_ylim([0, 20])
ax.axhline(bounds_NIS[0], linestyle='--', color='black')
ax.axhline(bounds_NIS[1], linestyle='--', color='black')

fig.savefig(fig_dir + 'NIS_ukf_N' + str(num_traj) + '.png')

plt.show()
