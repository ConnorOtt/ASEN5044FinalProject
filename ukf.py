"""
Define unscented Kalman filter
"""

from numpy.linalg import inv
from kf import KF


class UKF(KF):


    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(self, system)

        # CT nonlinear system functions
        self.f = system['f']
        self.h = system['h']


    def time_update():
        """
        Override the general KF's time update
        """
        # TODO: write and test

        


