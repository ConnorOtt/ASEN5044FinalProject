"""
Define general Kalman filter
"""
#NOTE: 0% chance that this runs as of now...just throwin around ideas

from numpy.linalg import inv
from abc import ABC, abstractmethod


class KF(ABC):

    def __init__(self, system):

        # System definition
        self.F_func = system["F"]
        self.G_func = system["G"]
        self.H_func = system["H"]
        self.Omega_func = system["Omega"]

        # Initial conditions
        self.t0 = system["t0"]
        self.x0 = system["x0"]
        self.P0 = system["P0"]

        # State estimate history
        self.t_hist = [self.t0]
        self.x_hist = [self.x0]
        self.P_hist = [self.P0]
        self.y_hist = []

    
    @abstractmethod
    def time_update(self): pass


    @abstractmethod
    def meas_update(self): pass


    def kalman_gain(self, P_pre_kp1, H_kp1, R_kp1, **kwargs):
        # Kalman gain at time t = t_{k+1}
        K_kp1 = P_pre_kp1@H_kp1.T @ inv(H_kp1@P_pre_kp1@H_kp1.T + R_kp1)
        return K_kp1


    def update(self, t_kp1, y_kp1):
        """
        Updates system's state estimate by processing an incoming measurement
         - Performs the time update (propagation to time of measurement)
         - Performs the measurement update (correction)

        Output is new state estimate (mean and covariance) at time of measurement
        """
        
        # Update filter's latest t and y
        self.t_hist.append(t_kp1)
        self.y_hist.append(y_kp1)

        x_pre_kp1, P_pre_kp1 = self.time_update(self.x_hist[-1], self.P_hist[-1])
        x_post_kp1, P_post_kp1 = self.meas_update(x_pre_kp1, P_pre_kp1, y_kp1)

        # Update filter's state estimate
        self.x_hist.append(x_post_kp1)
        self.P_hist.append(P_post_kp1)

