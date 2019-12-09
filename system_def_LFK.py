"""
Define the dynamic system for ASEN 5044 - Final Project

"""

# Standard imports
import sympy as sp
import numpy as np
import numpy.linalg as la

n = 4 # number of states 
m = 2
rE = 6378 # km - Earth radius
r0 = rE + 300 # km - orbit radius
delta_t = 20 # s
I = np.eye(n)


# Symbolic variable definitions
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3 = sp.symbols('y1 y2 y3')
u1, u2 = sp.symbols('u1 u2')
w1, w2 = sp.symbols('w1 w2')
xs, ys, xdots, ydots = sp.symbols('xs ys xdots ydots')

x = sp.Matrix([[x1, x2, x3, x4]]).T  # spacecraft state
s = sp.Matrix([xs, ys, xdots, ydots]).T  # station state
u = sp.Matrix([[u1, u2]]).T  # control vector
w = sp.Matrix([[w1, w2]]).T  # Noise vector


"""State Dynamcs"""
# Create f(x) = xdot
f1 = x2
f2 = -mu*x1*(x1**2 + x3**2)**(-3/2) + u1 + w1
f3 = x4
f4 = -mu*x3*(x1**2 + x3**2)**(-3/2) + u2 + w2
f = sym.Matrix([[f1, f2, f3, f4]]).T
f_func = sym.lambdify(x, f, 'numpy')

# Differentiate w/ respect to state/control/noise and convert to DT
# Differentiate f wrt x
A_tilde = f.jacobian(x)
A_tilde_func = sym.lambdify(x, A_tilde, 'numpy')

# Differentiate f wrt u and w
B_tilde = np.array(f.jacobian(u).tolist()).astype(np.float64)
Gam_tilde = np.array(f.jacobian(w).tolist()).astype(np.float64)

# Functions to discretize CT jacobians
def F_k_eval(x_nom_k, dt):
	"""Single step Euler integration to discretice CT dynamics""" 
	return I + dt * A_tilde_func(*x_nom_k)


def G_k_eval(dt):
	"""ZOH DT control"""
	return dt * B_tilde


def Omega_k_eval(dt):
	"""ZOH DT noise (?) I don't think it's technically ZOH but it 
		works like that
	"""
	return dt * Gam_tilde;


"""Measurements - maybe move to a different file"""
# Create h(x) = y
h1 = ((x1 - xs)**2 + (x3 - ys)**2)**(1/2)
h2 = ((x1 - xs)*(x2 - xdots) + (x3 - ys)*(x4 - ydots))/h1
h3 = sym.atan((x3 - ys)/(x1 - xs))
h = sym.Matrix([[h1, h2, h3]]).T
h_lam = sym.lambdify([*x, *s], h, 'numpy')

# Differentiate h wrt u
H_tilde = h.jacobian(x)
H_tilde_func = sym.lambdify([*x, *s], H_tilde, 'numpy')

def h_func(x_k, t_k, id_list=None):
    
    """Nonlinear measurement function - generally used for estimate
    	but could be used for simulated truth as well. 

	    if station id is specifed, only measurements for those
	    stations will be generated

		x_k   		<array_like>	State to evaluate on 
		t_k	  		<np.float64>	Time at which to evaluate
		id_list 	<list>			List of station ids

		returns a number of visible stations times p length ndarray if stations 
		are visible and None otherwise.

    """

    if type(x_k) is not np.ndarray:
    	x_k = np.array(x_k)

	sites = get_vis_sites(x_k, t_k, id_list=id_list)

    if len(sites) is 0:
    	return None
    else:
	   	# Evaluate h(x) = y	    
	   	y_list = [h_lam([*x_k, *site]) for site in sites]
	   	y = np.concatenate(y_list)


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

    check_vis = True
    if id_list is not None:
    	rot = [omegaE*t_k + pi/6*(i-1) for i in id_list]
    	check_vis = False # assume specified stations can see the satellite
   	else:
   		rot = [omegaE*t_k + pi/6*(i-1) for i in range(1, 12)]

    pos = [[rE*cos(r), rE*sin(r)] for r in rot]
    vel = [[-rE*omegaE*sin(r), rE*omegaE*cos(r)] for r in rot]
    
    if check_vis: # Remove stations that can't see the satellite
	    vis_pos_vel = []
	    for p, v in zip(pos, vel):
	        xs_arr = np.array([p[0], p[1]]) # site
	        x_arr = np.array([x1, x3]) # sat
	        slant = x_arr - xs_arr # sat minus site to get slant vector
	        
	        anglediff = np.arccos(np.dot(slant, xs_arr) / 
	        	(la.norm(slant)*la.norm(xs_arr)))
	        if (anglediff <= pi/2):
	            vis_pos_vel.append([p[0], v[0], p[1], v[1]])
	        
	    return vis_pos_vel
	else:
		return [p[0], v[0], p[1], v[1] for p, v in zip(pos, vel)]


def H_k_eval(x_nom_k, t_k, id_list=None):
	"""Evaluate H along nominal state for every station visible
		
		x_k 		<array_like>	State to evaluate on 
		t_k			<np.float64>	Time at which to evaluate
		id_list		<list>			List of station ids
	
	"""
	sites = get_vis_sites(x_k, t_k, id_list=id_list)

	H_k = [H_tilde_func(*x_nom_k, *xs) for xs in sites]
	return np.concatenate(H_k)


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




