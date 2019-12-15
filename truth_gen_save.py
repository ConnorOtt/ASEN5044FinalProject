""" 
Generate and save off truth trajectories for testing in monte-carlo

"""

from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn
import numpy as np
import pickle

from system_def import nl_orbit_prop as nlop
from system_def import h_func
from constants import *


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

ass_pickle = open(data_dir + "5044data.pickle","wb")
pickle.dump(ydata, ass_pickle)
ass_pickle.close()

Qtrue = data["Qtrue"]
Rtrue = data["Rtrue"]


#------------// Creating Monte-Carlo trajectories //-------------------------
t_0_sim = 0
t_f_sim = 2*pi * np.sqrt(r0**3 / mu) * 0.3  # NOTE: The 0.1 of an orbit is for testing
dt_sim = 10  # s
tvec_sim = np.arange(t_0_sim, t_f_sim, dt_sim)
num_step = tvec_sim.shape[0]

N = 10 # 50-100 total trajectories sounds good I guess
W_sim = Qtrue
R_sim = Rtrue
p_noise_dist = mvn(mean=[0, 0], cov=W_sim)

truth_trajectories = []
truth_measurements = []
for _ in range(N): 
    # Sample some noise for this trajectory 
    # NOTE: looking into a cleaner way to get noise into the dynamics - see f_func()
    p_noise = p_noise_dist.rvs(size=num_step)

    # restart the sim
    noisey_traj = [x_nom_0 + dx_0]
    noisey_meas = []  # measurements for k >= 1
    nlop.set_initial_value(noisey_traj, t_0_sim).set_f_params(None, p_noise)
    for k in range(1, num_step):
        nlop.integrate(tvec_sim[k])
        noisey_traj.append(nlop.y)

        meas, ids = h_func(nlop.y, tvec_sim[k], noise_cov=R_sim)  # generates a noisey meas
        noisey_meas.append({
            'meas':meas,
            'stationID':ids,
            't':tvec_sim[k],
            })

    truth_trajectories.append(noisey_traj)
    truth_measurements.append(noisey_meas)


# Save it off!
traj_pickle = open(data_dir + "truth_traj.pickle","wb")
pickle.dump(truth_trajectories, traj_pickle)
traj_pickle.close()

meas_pickle = open(data_dir + "truth_meas.pickle","wb")
pickle.dump(truth_measurements, meas_pickle)
meas_pickle.close()





