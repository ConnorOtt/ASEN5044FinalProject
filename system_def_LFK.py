"""
Define the dynamic system for ASEN 5044 - Final Project

"""

# Standard imports
import sympy as sp
import numpy as np

n = 4 # number of states 
rE = 6378 # km - Earth radius
r0 = rE + 300 # km - orbit radius
delta_t = 20 # s
I = np.eye(n)

x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3 = sp.symbols('y1 y2 y3')
u1, u2 = sp.symbols('u1 u2')
w1, w2 = sp.symbols('w1 w2')

# Temporary variables to be filled in with different station parameters once h(x) needs to be elvaulated
xs, ys, xdots, ydots = sp.symbols('xs ys xdots ydots')

x = sp.Matrix([[x1, x2, x3, x4]]).T  # spacecraft state
s = sp.Matrix([xs, ys, xdots, ydots]).T  # station state
u = sp.Matrix([[u1, u2]]).T  # control vector
w = sp.Matrix([[w1, w2]]).T  # Noise vector

# Create f(x) = xdot
f1 = x2
f2 = -mu*x1*(x1**2 + x3**2)**(-3/2) + u1 + w1
f3 = x4
f4 = -mu*x3*(x1**2 + x3**2)**(-3/2) + u2 + w2
f = sym.Matrix([[f1, f2, f3, f4]]).T
f_func = sym.lambdify(x, f, 'numpy')

# Create h(x) = y
h1 = ((x1 - xs)**2 + (x3 - ys)**2)**(1/2)
h2 = ((x1 - xs)*(x2 - xdots) + (x3 - ys)*(x4 - ydots))/h1
h3 = sym.atan((x3 - ys)/(x1 - xs))
h = sym.Matrix([[h1, h2, h3]]).T

# Differentiate w/ respect to state/control/noise and convert to DT
# Differentiate f wrt x
A_tilde = f.jacobian(x)
A_tilde_func = sym.lambdify(x, A_tilde, 'numpy')

# Differentiate f wrt u and w
B_tilde = f.jacobian(u)
Gam_tilde = f.jacobian(w)

# Differentiate h wrt u
H_tilde = h.jacobian(x)
H_tilde_func = sym.lambdify([*x, *s], H_tilde, 'numpy')


def F_k_eval(x_k_nom, dt):
	# Single step Euler integration to discretice CT dynamics 
	return I + dt * A_tilde_func(*x_k_nom)

def G_k_eval(x_k_nom, dt):
	# ZOH control
	return dt * B_tilde

def Omega_k_eval(x_k_nom, dt):
	return dt * Gam_tilde ;

def H_k_eval(x_k_nom, x_station_list, dt):
	# Evaluate H along nominal state for every station visible
	H_k = [H_tilde_func(*x_k_nom, *xs) for xs in x_station_list]
	return np.concatenate(H_k)


dt_eval_funcs = {
	'F': F_k_eval,
	'G': G_k_eval,
	'Omega': Omega_k_eval,
	'H': H_k_eval,
}







