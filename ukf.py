"""
Define unscented Kalman filter

maybebaby
"""

from numpy.linalg import inv


class UKF(KF):

    def __init__(self, system):

        # CT nonlinear system functions
        self.f = system['f']
        self.h = system['h']

        # DT jacobian functions
        self.F_func = system['H']
        self.G_func = system['H']
        self.H_func = system['H']
        self.Omega_func = system['Omega']



