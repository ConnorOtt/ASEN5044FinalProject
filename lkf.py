"""
Define linearized Kalman filter
"""

from numpy.linalg import inv


class LKF(KF):

    def __init__(self, system):

        # CT nonlinear system functions
        self.f = system['f']
        self.h = system['h']

        # DT jacobian functions
        self.F_func = system['H']
        self.G_func = system['H']
        self.H_func = system['H']
        self.Omega_func = system['Omega']


    # TODO: define function that overrides the current propogation step
