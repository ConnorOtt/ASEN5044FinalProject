"""
Define unscented Kalman filter

maybebaby
"""

from numpy.linalg import inv


class UKF(KF):

    def __init__(self, system):

