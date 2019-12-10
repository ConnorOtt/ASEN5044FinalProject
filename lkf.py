"""
Define linearized Kalman filter
"""

from kf import KF
from numpy.linalg import inv
import numpy as np


class LKF(KF):


    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(system)

        # CT nonlinear system functions
        self.f = system["f"]
        self.h = system["h"]

        # LKF-specific properties, here all of these are static (TODO: verify)
        self.x_nom_k = system["x_nom"]
        self.x_nom_kp1 = system["x_nom"]
        self.delta_t = system["dt"]
        self.u_k = np.zeros((2,1))  # no control for now
        self.Q_k = system["Q"]
        self.R_kp1 = system["R"]


    def time_update(self, dx_post_k, P_post_k):
        """
        Override the general KF's time update.
        """

        # time update to bring x and P up to k+1 from k (LKF)
        #x_nom_k = kwargs['x_nom_k']
        #delta_t = kwargs['dt']
        #u_k = kwargs['u_k']
        #Q_k = kwargs['Q_k']

        F_k = self.F_func(self.x_nom_k, self.delta_t)
        G_k = self.G_func(self.delta_t)
        Omega_k = self.Omega_func(self.delta_t)

        dx_pre_kp1 = F_k @ dx_post_k + G_k @ self.u_k
        P_pre_kp1 = F_k @ dx_post_k @ F_k.T + Omega_k @ self.Q_k @ Omega_k.T

        return dx_pre_kp1, P_pre_kp1


    def meas_update(self, dx_pre_kp1, P_pre_kp1, y_kp1):
        """
        Override the general KF's measurement update.
        """
        
        # Pull the latest measurement and associated data
        # TODO: maybe think of a better way to do this. Seems kinda sketch, idk
        t_kp1 = self.t_hist[-1]
        y_kp1 = self.y_hist[-1]["meas"]
        id_list = self.y_hist[-1]["stationID"]


        # Evaluate jacobians and Kalman gain on nominal trajectory
        H_kp1 = self.H_func(self.x_nom_kp1, t_kp1, id_list=id_list)
        K_kp1 = self.kalman_gain(P_pre_kp1, H_kp1, self.R_kp1)

        # Generate nominal measurement and pre-fit residual
        y_nom_kp1 = self.h(self.x_nom_kp1, t_kp1, id_list=id_list)[0] # nominal measurement
        # FIXME: This is erroring^^ because it's returning a couple outputs
        #        This is probably a good spot to be thinking about the dynamic 
        #        sizing of y and H
        dy_kp1 = y_kp1 - y_nom_kp1
        pre_fit_residual = dy_kp1 - H_kp1 @ dx_pre_kp1;
        # FIXME: All sorts of sizing issues going on right here

        # Apply mu
        dx_post_kp1 = dx_pre_kp1 + K_kp1 @ pre_fit_residual
        P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

        # TODO: Package some of this up into a dict as output to include pre/post-fit 
        # residuals, pre/post measurement update stats and maybe some of the evaluated
        # jacobians. The dict can be parsed once all of them are collected and the 
        # filter finishes.

        return dx_post_kp1, P_post_kp1






