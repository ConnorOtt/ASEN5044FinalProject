"""
Define linearized Kalman filter

Conventions:
    _k, _kp1, _km1      -Indicates time steps k, k (p)lus 1 and k (m)inus 1. This file does not
                        use _km1, and all time updates bring state and covariance from time k to 
                        kp1.
    _pre, _post         -Indicates pre or post measurement update for the specified time step.
                        Matches ^- and ^+ notation respectively.
    _nom                -Indicates state or measurement pulled from or evaluated on some 
                        predetermined nominal trajectory.
"""

# Standard imports
from numpy.linalg import inv
import numpy as np
from scipy.linalg import block_diag

# Local imports
from kf import KF
from constants import I #dentity
from system_def import nl_orbit_prop as nom_prop

class LKF(KF):


    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(system)

        # CT nonlinear system functions
        self.f = system["f"]
        self.h = system["h"]


        # LKF-specific properties, here all of these are static (TODO: verify)
        self.dx_0 = system['dx_0']
        self.x_nom_0 = system["x_nom_0"]
        self.delta_t = system["dt"]
        self.u_k = np.zeros((2,))  # no control for now or ever lol
        self.Q_k = system["Q"]
        self.R_kp1 = system["R"]

        self.x_nom_k = self.x_nom_0
        self.x_nom_th = [self.x_nom_0]

        # 'Current' things in the filter (all post meas where applicable)
        self.dx_k = self.dx_0
        self.P_k = self.P_0
        self.x_k = system['x_0']  # this is an estimate


    def time_update(self, dx_post_k, P_post_k):
        """
        Override the general KF's time update.
        """

        # Do not propagate nominal, it's already caught up from last
        # measurement update
        x_nom_k = self.x_nom_k

        F_k = self.F_func(x_nom_k, self.delta_t)
        G_k = self.G_func(self.delta_t)
        Omega_k = self.Omega_func(self.delta_t)
        # print(G_k.shape)
        # print(self.u_k.shape)
        # print(F_k.shape)
        # print(dx_post_k.shape)
        # print(self.Q_k.shape)
        # print(Omega_k.shape)



        dx_pre_kp1 = F_k @ dx_post_k + G_k @ self.u_k
        P_pre_kp1 = F_k @ P_post_k @ F_k.T + Omega_k @ self.Q_k @ Omega_k.T

        # print(dx_pre_kp1.shape)   

        return dx_pre_kp1, P_pre_kp1


    def meas_update(self, dx_pre_kp1, P_pre_kp1, y_kp1, t_kp1):
        """
        Override the general KF's measurement update.
        """
        id_list = y_kp1['stationID']
        y_kp1 = y_kp1['meas']
        if y_kp1 is None:
            return dx_pre_kp1, P_pre_kp1

        # Bring the nominal up to evaluate H at kp1
        x_nom_kp1 = self.__update_nom()

        # Evaluate jacobians and Kalman gain on nominal trajectory
        H_kp1 = self.H_func(x_nom_kp1, t_kp1, id_list=id_list)
        R_list = [self.R_kp1 for _ in range(int(H_kp1.shape[0]/3))]
        R_kp1 = block_diag(*R_list)

        K_kp1 = self.kalman_gain(P_pre_kp1, H_kp1, R_kp1)

        # Generate nominal measurement and pre-fit residual
        y_nom_kp1, _ = self.h(x_nom_kp1, t_kp1, id_list=id_list) # nominal measurement
        print(y_nom_kp1.shape)
        dy_kp1 = y_kp1 - y_nom_kp1
        pre_fit_residual = dy_kp1 - H_kp1 @ dx_pre_kp1;

        # Apply measurement update
        dx_post_kp1 = dx_pre_kp1 + K_kp1 @ pre_fit_residual
        P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

        print(dx_pre_kp1)

        # TODO: Package some of this up into a dict as output to include pre/post-fit 
        # residuals, pre/post measurement update stats and maybe some of the evaluated
        # jacobians. The dict can be parsed once all of them are collected and the 
        # filter finishes.

        return dx_post_kp1, P_post_kp1


    def __update_nom(self):
        """Propagate current nominal trajectory forward
            by one time step for linearization
        """ 

        nom_prop.set_initial_value(self.x_nom_k, 0)
        x_nom_kp1 = nom_prop.integrate(self.delta_t)
        self.x_nom_th.append(x_nom_kp1)
        self.x_nom_k = self.x_nom_th[-1]

        return self.x_nom_k


    """
    @property
    def R_kp1(self):
        
        if self.

        return self.R_kp1
    """


