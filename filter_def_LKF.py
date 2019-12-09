"""
Define how the LKF performs time and measurement updates

Conventions:
	_k, _kp1, _km1		-Indicates time steps k, k (p)lus 1 and k (m)inus 1. This file does not
						use _km1, and all time updates bring state and covariance from time k to 
						kp1.
	_pre, _post 		-Indicates pre or post measurement update for the specified time step.
						Matches ^- and ^+ notation respectively.
	_nom 				-Indicates state or measurement pulled from or evaluated on some predetermined
						nominal trajectory.
"""

from .system_def_LFK import dt_jac_eval_funcs, ct_nl_funcs, n, m
from numpy.linalg import inv

H_func = dt_jac_eval_funcs['H']
F_func = dt_jac_eval_funcs['F']
G_func = dt_jac_eval_funcs['G']
Omega_func = dt_jac_eval_funcs['Omega']

f = ct_nl_funcs['f']
h = ct_nl_funcs['h']

I = np.eye(n)


def kalman_gain(P_pre_kp1, H_kp1, R_kp1, **kwargs):
	# Kalman gain at time t = t_{k+1}
	K_kp1 = P_pre_kp1@H_kp1 @ inv(H_kp1@P_pre_kp1@H_kp1.T + R_kp1)
	return K_kp1


def time_update(dx_post_k, P_post_k, **kwargs):
	# time update to bring x and P up to k+1 from k (LKF)
	x_nom_k = kwargs['x_nom_k']
	delta_t = kwargs['dt']
	u_k = kwargs['u_k']
	Q_k = kwargs['Q_k']

	F_k = F_func(x_nom_k, delta_t)
	G_k = G_func(x_nom_k, delta_t)
	Omega_k = Omega_func(x_nom_k, delta_t)

	dx_pre_kp1 = F_k @ dx_post_k + G_k @ u_k
	P_pre_kp1 = F_k @ dx_post_k @ F_k.T + Omega_k @ Q_k @ Omega_k.T

	return dx_pre_kp1, P_pre_kp1


def meas_update(dx_pre_kp1, P_pre_kp1, y_kp1, R_kp1, **kwargs):
	# Apply measurement update for LKF
	id_list = kwargs['id_list']
	t_kp1 = kwargs['t_kp1']
	x_nom_kp1 = kwargs['x_nom_kp1']

	# Evaluate jacobians and Kalman gain on nominal trajectory
	H_kp1 = H_func(x_nom_kp1, t_kp1, id_list=id_list)
	K_kp1 = kalman_gain(P_pre_kp1, H_kp1, R_kp1)

	# Generate nominal measurement and pre-fit residual
	y_nom_kp1 = h(x_nom_kp1, t_kp1, id_list=id_list) # nominal measurement
	dy_kp1 = y_kp1 - y_nom_kp
	pre_fit_residual = dy_kp1 - H_kp1 @ dx_pre_kp1;

	# Apply mu
	dx_post_kp1 = dx_pre_kp1 + K_kp1 @ pre_fit_residual
	P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

	# TODO: Package some of this up into a dict as output to include pre/post-fit 
	# residuals, pre/post measurement update stats and maybe some of the evaluated
	# jacobians. The dict can be parsed once all of them are collected and the 
	# filter finishes.

	return dx_post_kp1, P_post_kp1






