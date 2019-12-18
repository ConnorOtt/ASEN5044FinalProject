""" 
Validate the linearized jacobians against nonlinear propagation and 
measurement function

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
from system_def import nl_orbit_prop as nlop

# overwriting dx_0 for linearized perturbations
dx_0 = np.array([0.01, 0.001, 0.01, 0.001])
x_pert_0 = x_nom_0 + dx_0

t0 = 0
tf = T 
h_func = ct_nl_funcs['h']

# Nominal propagation
x_nom = [x_nom_0]
y_nom = []
nlop.set_initial_value(x_nom_0, t0).set_f_params(None, None)
while nlop.t < tf:
	# calculate nonlinear measurement
	meas, ids = h_func(nlop.y, nlop.t)
	y_nom.append({
		'meas': meas,
		'ids': ids,
		't': nlop.t,
	})

	nlop.integrate(nlop.t + delta_t)
	x_nom.append(nlop.y)

	
# Perturbed propagation
x_pert = [x_pert_0]
y_pert = []
nlop.set_initial_value(x_pert_0, t0).set_f_params(None, None)
for y in y_nom:
	meas, _ = h_func(nlop.y, nlop.t, id_list=y['ids'])
	y_pert.append({
		'meas':meas,
		'ids':y['ids'],
		't': nlop.t
	})

	nlop.integrate(y['t'] + delta_t)
	x_pert.append(nlop.y)

dx_nl = [xp - xn for xp, xn in zip(x_pert, x_nom)]
dy_nl = [yp['meas'] - yn['meas'] for yp, yn in zip(y_pert, y_nom)]
for dy in dy_nl:
	if dy[2] > pi:
		dy[2] -= 2*pi
	elif dy[2] < -pi:
		dy[2] += 2*pi

# Perturbation propagation
F_eval = dt_jac_eval_funcs['F']
H_eval = dt_jac_eval_funcs['H']
dx = [dx_0]
dy = []
for x, y in zip(x_nom, y_nom):
	F_k = F_eval(x, delta_t)
	dx_k = dx[-1]
	dx_kp1 = F_k @ dx_k
	dx.append(dx_kp1)

	H_k = H_eval(x, y['t'], y['ids'])
	dy_k = H_k @ dx[-1]
	dy.append(dy_k)


# Plot the nonlinear perturbation and the linear perturbations over time

plt.rcParams['font.size'] = 24
plt.rcParams['figure.figsize'] = 16, 12
fig, ax = plt.subplots(n, 1, sharex=True)

ax[0].set_title('Comparison between nonlinear and linear perturbation propagation')

for i in range(n):
    state_quant1 = [x[i] for x in dx_nl]
    state_quant2 = [x[i] for x in dx]
    ax[i].plot(state_quant1, '-', color='dodgerblue', label=u'$\delta x_{nonlinear}$')
    ax[i].plot(state_quant2, '-', color='orangered', label=u'$\delta x_{linear}$')
    ax[i].autoscale(enable=True, axis='x', tight=True)
    ax[i].set_ylabel('$x_{%d}$' % (i+1))

ax[0].legend(loc='upper left')
ax[-1].set_xlabel('time step')

fig.savefig(fig_dir + 'nonlvl_state.png')

fig, ax = plt.subplots(p, 1, sharex=True)
ax[0].set_title('Comparison between nonlinear and linear perturbation propagation')
for i in range(p):
    quant1 = [y[i] for y in dy_nl]
    quant2 = [y[i] for y in dy]
    ax[i].plot(quant1, '-', color='dodgerblue', label=u'$\delta y_{nonlinear}$')
    ax[i].plot(quant2, '-', color='orangered', label=u'$\delta y_{linear}$')
    ax[i].autoscale(enable=True, axis='x', tight=True)		
    ax[i].set_ylabel('$y_{%d}$' % (i+1))
ax[0].legend(loc = 'upper left')
ax[-1].set_xlabel('time step')

plt.show()
fig.savefig(fig_dir + 'nonlvl_meas.png')




