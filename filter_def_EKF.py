"""
Define how the EKF performs time and measurement updates

Conventions:
	_k, _kp1, _km1		-Indicates time steps k, k (p)lus 1 and k (m)inus 1. This file does not
						use _km1, and all time updates bring state and covariance from time k to 
						time kp1.
	_pre, _post 		-Indicates pre or post measurement update for the specified time step.
						Matches ^- and ^+ notation respectively.
	_nom 				-Indicates state or measurement pulled from or evaluated on some predetermined
						nominal trajectory (not used in the EKF algorithm).
"""


# Standard imports
from numpy.linalg import inv
from scipy.integrate import ode

# Local imports
from system_def_LFK import dt_jac_eval_funcs, ct_nl_funcs
from system_def_LFK import nl_orbit_prop as nl_prop
from constants import I

H_func = dt_jac_eval_funcs['H']
F_func = dt_jac_eval_funcs['F']
G_func = dt_jac_eval_funcs['G']
Omega_func = dt_jac_eval_funcs['Omega']

f = ct_nl_funcs['f']
h = ct_nl_funcs['h']


def kalman_gain(P_pre_kp1, H_kp1, R_kp1, **kwargs):
	# Kalman gain at time t = t_{k+1}
	K_kp1 = P_pre_kp1@H_kp1 @ inv(H_kp1@P_pre_kp1@H_kp1.T + R_kp1)
	return K_kp1


def time_update(x_post_k, P_post_k, **kwargs):
	# time update to bring x and P up to k+1 from k (EKF)
	t_k = kwargs['t_k']
	t_kp1 = kwargs['t_kp1']
	u_k = kwargs['u_k']
	Q_k = kwargs['Q_k']

	# Propagate the covariance first because it's based on linearized
	# jacobians about previous time step
	F_k = F_func(x_post_k, delta_t)
	Omega_k = Omega_func(delta_t)
	P_pre_kp1 = F_k @ P_post_k @ F_k.T + Omega_k @ Q_k @ Omega_k.T

	# Nonlinear state propagation
	nl_prop.set_initial_value(x_post_k, t_k)
	x_pre_kp1 = nl_prop.integrate(t_kp1)
	
	return x_pre_kp1, P_pre_kp1


def meas_update(x_pre_kp1, P_pre_kp1, y_kp1, R_kp1, **kwargs):
	"""Apply measurement update for EKF
	

		TODO: Package some of this up into a dict as output to include pre/post-fit 
		residuals, pre/post measurement update stats and maybe some of the evaluated
		jacobians. The dict can be parsed once all of them are collected and the 
		filter finishes.
	"""

	id_list = kwargs['id_list']  # will probably be used when pulling in McMahon's pre-made measurements which include measurement ids
	t_kp1 = kwargs['t_kp1']

	y_pre_kp1 = h(x_pre_kp1, t_kp1)  # Nonlinear measurement mapping
	nl_innov = y_kp1 - y_pre_kp1

	H_kp1 = H_func(x_pre_kp1, t_kp1, id_list=id_list)
	K_kp1 = kalman_gain(P_pre_kp1, H_kp1, R_kp1)

	x_post_kp1 = x_pre_kp1 + K_kp1 @ nl_innov
	P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

	return x_post_kp1, P_post_kp1


