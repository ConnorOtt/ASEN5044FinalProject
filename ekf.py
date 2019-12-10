"""
Define extended Kalman filter
"""

from numpy.linalg import inv
from kf import KF


class EKF(KF):


    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(self, system)

        # CT nonlinear system functions
        self.f = system['f']
        self.h = system['h']


    def time_update(self, tk, yk):
        """
        Override the general KF's time update
        The EKF uses a nonlinear propagation step. It numerically integrates 
        the nonlinear system of equations in self.f to update perform the time
        update.
        """
        # TODO: write and test

        


