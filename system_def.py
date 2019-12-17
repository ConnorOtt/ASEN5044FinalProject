"""
Define the dynamic system for ASEN 5044 - Final Project

TODO: Split up the dynamics and measurements into two separate modules
to keep things more organized. 

"""

# Standard imports
import sympy as sp
import numpy as np
import numpy.linalg as la
from math import sin, cos
from scipy.integrate import ode
from scipy.stats import multivariate_normal as mvn

# Local imports
from constants import *

# Symbolic variable definitions
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3 = sp.symbols('y1 y2 y3')
u1, u2 = sp.symbols('u1 u2')
w1, w2 = sp.symbols('w1 w2')
xs, ys, xdots, ydots = sp.symbols('xs ys xdots ydots')

x = sp.Matrix([[x1, x2, x3, x4]]).T  # spacecraft state
s = sp.Matrix([xs, xdots, ys, ydots]).T  # station state
u = sp.Matrix([[u1, u2]]).T  # control vector
w = sp.Matrix([[w1, w2]]).T  # Noise vector

"""State Dynamcs"""
# Create f(x) = xdot
f1 = x2
f2 = -mu*x1*(x1**2 + x3**2)**(-3/2) + u1 + w1
f3 = x4
f4 = -mu*x3*(x1**2 + x3**2)**(-3/2) + u2 + w2
f = sp.Matrix([[f1, f2, f3, f4]]).T
f_lam = sp.lambdify([*x, *u, *w], f, 'numpy') 

# Differentiate w/ respect to state/control/noise and convert to DT
# Differentiate f wrt x
A_tilde = f.jacobian(x)
A_tilde_func = sp.lambdify(x, A_tilde, 'numpy')

# Differentiate f wrt u and w - constant so just converting to
# ndarray right out of the gate. 
B_tilde = np.array(f.jacobian(u).tolist()).astype(np.float64)
Gam_tilde = np.array(f.jacobian(w).tolist()).astype(np.float64)


def f_func(t_k, x_k, u_k, w_vec):
	# Nonlinear dynamics function w/ or without noise/control

	if type(x_k) is not np.ndarray:  # for the odd accidental list input
		x_k = np.array(x_k).reshape((-1, 1))
		
	# gotta be a better way to do this: 
	u_k = u_k if u_k is not None else [0, 0]

	# NOTE: the following lines accepting ZOH noise input
	# may be more clean (but slower?) if the noise is sampled
	# here (from an input Q, see h_func) and the random seed 
	# is set based on floor(t_k/delta_t) to maintain the same noise 
	# sample for t_k < t < t_k+delta_t.
	k = int(t_k/delta_t)
	w_k = w_vec[k] if w_vec is not None else [0, 0]
	
	x_dot_k_obj = f_lam(*[*x_k, *u_k, *w_k])
	x_dot_k = np.array(x_dot_k_obj.tolist()).astype(np.float64)

	return x_dot_k

# Instantiate a nonlinear integrator to propagate nonlinear trajectory
# need to set initial conditions and whatnot yourself. 
nl_orbit_prop = ode(f_func).set_integrator('dopri5')

# Functions to discretize CT jacobians
def F_k_eval(x_nom_k, dt):
	"""Single step Euler integration to discretize CT dynamics""" 
	return I + dt*np.array(A_tilde_func(*x_nom_k).tolist()).astype(np.float64)


def G_k_eval(dt):
	"""ZOH DT control"""
	return dt * B_tilde


def Omega_k_eval(dt):
	"""ZOH DT noise (?) I don't think it's technically ZOH but it 
		works like that
	"""
	return dt * Gam_tilde;


"""Measurements - move to a different file"""
# Create h(x) = y
h1 = ((x1 - xs)**2 + (x3 - ys)**2)**(1/2)
h2 = ((x1 - xs)*(x2 - xdots) + (x3 - ys)*(x4 - ydots))/h1
h3 = sp.atan2((x3 - ys),(x1 - xs))
h = sp.Matrix([[h1, h2, h3]]).T
h_lam = sp.lambdify([*x, *s], h, 'numpy')

# Differentiate h wrt u
H_tilde = h.jacobian(x)
H_tilde_func = sp.lambdify([*x, *s], H_tilde, 'numpy')

# Nonlinear measurement function
def h_func(x_k, t_k, id_list=None, noise_cov=None):

	"""Nonlinear measurement function - does not include noise

	    if station id is specifed, only measurements for those
	    stations will be generated

		x_k   		<array_like>	State to evaluate on 
		t_k	  		<np.float64>	Time at which to evaluate
		id_list 	<list>			List of station ids

		returns a number of visible stations times p length ndarray if stations 
		are visible and None otherwise.

	"""

	x_k = np.array(x_k).reshape((-1,))

	sites, ids = get_vis_sites(x_k, t_k, id_list=id_list)
	if len(sites) == 0:
		return None, None
	else:
		# sample noise if needed,
		# NOTE: it may be faster to sample all the noise values
		# at once (outside h_func) and just add it onto a set of 
		# measurement vectors rather than sample each time this is 
		# evaluated

		if noise_cov is not None:
			mnoise = mvn(mean=None, cov=noise_cov).rvs(size=len(ids))
			mnoise = mnoise.flatten(order='C') # row flatten (C-style) not column flatten
		else:
			mnoise = np.zeros((len(ids)*p, ))

		# Evaluate h(x) = y  
		y_list = [h_lam(*[*x_k, *site]) for site in sites]
		y = np.concatenate(y_list).reshape((-1, )) # lots of reshaping everywhere - looking for 1d vectors always and 2d matrices
		y = y + mnoise
		return y, ids


def H_k_eval(x_nom_k, t_k, id_list=None):
	"""Evaluate H along nominal state for every station visible
		
		x_k 		<array_like>	State to evaluate on 
		t_k			<np.float64>	Time at which to evaluate
		id_list		<list>			List of station ids
	
	"""
	x_nom_k = np.array(x_nom_k).reshape((-1, ))
	sites, _ = get_vis_sites(x_nom_k, t_k, id_list=id_list)

	if len(sites) != 0:
		H_k = [H_tilde_func(*[*x_nom_k, *xs]) for xs in sites]
		return np.concatenate(H_k).astype(np.float64)
	else: 
		return None


def get_vis_sites(x_k, t_k, id_list=None):
	"""Determine which sites can see the satellite for the given 
		measurment system in ASEN 5044 Final Project

		x_k   		<array_like>	State to evaluate on 
		t_k	  		<np.float64>	Time at which to evaluate
		id_list 	<list>			List of station ids

		returns the states of each station (x, y, vx, vy) in list of ndarrays
		if stations are visible and return an empty list otherwise

	"""


	x1 = x_k[0]
	x3 = x_k[2]

	check_vis = True # change back to True
	if id_list is not None:
		check_vis = False # assume specified stations can see the satellite
	else:
		id_list = np.arange(1, 13, 1)

	if not isinstance(id_list, list):
		id_list = list(id_list)

	# Calculate rotation angle for each station at time t
	rot = [omegaE*t_k + pi/6*(i-1) for i in id_list]	

	# Convert to position and velocity
	pos = [[rE*cos(r), rE*sin(r)] for r in rot]
	vel = [[-rE*omegaE*sin(r), rE*omegaE*cos(r)] for r in rot]

	if check_vis: # Remove stations that can't see the satellite
		vis_pos_vel = []
		vis_ids = []
		for s_id in id_list:
			p = pos[int(s_id-1)]
			v = vel[int(s_id-1)]

			xs_arr = np.array([p[0], p[1]]) # site
			x_arr = np.array([x1, x3]) # sat
			slant = x_arr - xs_arr # sat minus site to get slant vector
			
			anglediff = np.arccos(np.dot(slant, xs_arr) / 
				(la.norm(slant)*la.norm(xs_arr)))
					
			if (anglediff <= pi/2):
				vis_pos_vel.append([p[0], v[0], p[1], v[1]])
				vis_ids.append(s_id)
			
		return vis_pos_vel, vis_ids
	else: 
		pos_vel = [[p[0], v[0], p[1], v[1]] for p, v in zip(pos, vel)]
		return pos_vel, id_list

# Things to be imported by the filter definition to perform 
# time and measurement updates
dt_jac_eval_funcs = {
	'F': F_k_eval,
	'G': G_k_eval,
	'Omega': Omega_k_eval,
	'H': H_k_eval,
}
ct_nl_funcs = {
	'f': f_func,
	'h': h_func,
}


def mat_latex():
	"""Get the jacobians in latex markdown"""

	jacs = {
		'A_tilde':A_tilde, 
		'B_tilde':sp.Matrix(B_tilde), 
		'Gam_tilde':sp.Matrix(Gam_tilde), 
		'H_tilde':H_tilde}
	
	jac_latex = {}
	for key, val in jacs.items():

		print(sp.latex(val))
		jac_latex[key] = sp.latex(val)

	return jac_latex

