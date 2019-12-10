"""
Define general Kalman filter

Conventions:
    _k, _kp1, _km1      -Indicates time steps k, k (p)lus 1 and k (m)inus 1. This file does not
                        use _km1, and all time updates bring state and covariance from time k to 
                        kp1.
    _pre, _post         -Indicates pre or post measurement update for the specified time step.
                        Matches ^- and ^+ notation respectively.
    _nom                -Indicates state or measurement pulled from or evaluated on some 
                        predetermined nominal trajectory.
"""

#NOTE: 0% chance that this runs as of now...just throwin around ideas

from numpy.linalg import inv
from numpy import empty
from scipy.linalg import block_diag
from abc import ABC, abstractmethod


class KF(ABC):

    def __init__(self, system):

        # System definition
        self.F_func = system["F"]
        self.G_func = system["G"]
        self.H_func = system["H"]
        self.Omega_func = system["Omega"]

        # Initial conditions
        self.t_0 = system["t_0"]
        self.x_0 = system["x_0"]
        self.P_0 = system["P_0"]

        # State estimate history
        self.t_hist = [self.t_0]
        self.x_hist = [self.x_0]
        self.P_hist = [self.P_0]
        self.y_hist = []

    
    @abstractmethod
    def time_update(self): pass


    @abstractmethod
    def meas_update(self): pass


    def kalman_gain(self, P_pre_kp1, H_kp1, R_kp1):
        # Kalman gain at time t = t_{k+1}

        # If H is bigger because >1 measurement, then increase dimensions of R
        # Just replicating R with block_diag until it's big enough
        # NOTE: Does this seem reasonable?
        if H_kp1.shape[0] > R_kp1.shape[0]:

            p = R_kp1.shape[0]              # size of a normal measurement set
            m = int(H_kp1.shape[0] / p)     # number of measurements

            R = R_kp1
            for i in range(2,m+1):
                R = block_diag(R, R_kp1)    # Replicate R along diagonal
            R_kp1 = R                       # Update R_kp1



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
        self.t_hist.append(t_kp1)  # current estimate is at t_k when this line goes
        self.y_hist.append(y_kp1)

        x_pre_kp1, P_pre_kp1 = self.time_update(self.x_hist[-1], self.P_hist[-1])
        x_post_kp1, P_post_kp1 = self.meas_update(x_pre_kp1, P_pre_kp1, y_kp1)

        # Update filter's state estimate
        self.x_hist.append(x_post_kp1)
        self.P_hist.append(P_post_kp1)


    def report_hist(self): 
        """Output time history of everything (requested)

        """
        pass


    def plot_hist(self):
        """yeah
        """
        pass
