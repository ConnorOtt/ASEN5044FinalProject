"""
Define how the filter performs time and measurement updates

"""

from .system_def_LFK import dt_jac_eval_funcs, ct_nl_funcs, n, m
from numpy.linalg import inv

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
	# time update to bring x and P up to k+1 from k (LKF)
	delta_t = kwargs['dt']
	u_k = kwargs['u_k']
	Q_k = kwargs['Q_k']

	F_k = F_func(x_nom_k, delta_t)
	G_k = G_func(x_nom_k, delta_t)
	Omega_k = Omega_func(x_post_k, delta_t)

	x_pre_kp1 = F_k @ x_post_k + G_k @ u_k
	P_pre_kp1 = F_k @ P_post_k @ F_k.T + Omega_k @ Q_k @ Omega_k.T

	return x_pre_kp1, P_pre_kp1

def meas_update(x_pre_kp1, P_pre_kp1, **kwargs):
	# Apply measurement update for LKF

	id_list = kwargs['id_list']
	t_kp1 = kwargs['t_kp1']
	x_nom_kp1 = kwargs['x_nom_kp1']

	H_kp1 = H_func(x_nom_kp1, t_kp1, id_list=id_list)
	y_pre_kp1 = h(x_pre_kp1, t_kp1, id_list=id_list)


	return x_post_kp1, P_post_kp1






