"""
Define extended Kalman filter

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
from constants import I, n, p, pi
from system_def import nl_orbit_prop as nom_prop
from system_def import nl_orbit_prop as nl_prop


class EKF(KF):

    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(system)

        # CT nonlinear system functions
        self.f = system["f"]
        self.h = system["h"]

        # EKF-specific properties, here all of these are static
        #self.x_nom_0 = system["x_nom_0"]
        self.delta_t = system["dt"]
        self.u_k = np.zeros((2,))  # no control for now or ever lol
        self.Q_k = system["Q"]
        self.R_kp1 = system["R"]


    def time_update(self, x_post_k, P_post_k, t_kp1):
        """
        Override the general KF's time update
        The EKF uses a nonlinear propagation step. It numerically integrates
        the nonlinear system of equations in self.f to update perform the time
        update.
        """

        t_k = t_kp1 - self.delta_t

        # Propagate the covariance first because it's based on linearized
        # jacobians about previous time step
        F_k = self.F_func(x_post_k, self.delta_t)
        Omega_k = self.Omega_func(self.delta_t)
        P_pre_kp1 = F_k @ P_post_k @ F_k.T + Omega_k @ self.Q_k @ Omega_k.T

        G_k = self.G_func(self.delta_t)

        # Nonlinear state propagation
        nl_prop.set_initial_value(x_post_k, t_k)
        nl_prop.set_f_params(None, None)
        x_pre_kp1 = nl_prop.integrate(t_kp1)

        return x_pre_kp1, P_pre_kp1


    def meas_update(self, x_pre_kp1, P_pre_kp1, y_kp1, t_kp1):
        """
        Override the general KF's measurement update.
        """

        id_list = y_kp1['stationID']
        y_kp1 = y_kp1['meas']
        if y_kp1 is None:
            # NOTE: didn't put the time into calculating all of the other things
            # TODO: This this the heck up
            none_meas = [None for _ in range(p)]
            out = {
                'x_pre_kp1': x_pre_kp1,
                'x_post_kp1': x_pre_kp1,
                'x_update': np.zeros((n, )),
                'P_pre_kp1': P_pre_kp1,
                'P_post_kp1': P_pre_kp1,
                'pre_fit_residual': none_meas,
                'post_fit_residual': none_meas,
                'y_kp1': none_meas,
                'y_nom_kp1': none_meas,
                'y_nom_kp1': none_meas,
                'y_est_kp1': none_meas,
                'y_pre_est_kp1': none_meas,
                'y_post_est_kp1': none_meas,
                'y_update': none_meas,
            }
            return out

        # Nonlinear measurement mapping
        y_pre_kp1, _ = self.h(x_pre_kp1, t_kp1, id_list=id_list)
        nl_innov = y_kp1 - y_pre_kp1     # etilde_ykp1 in lecture slides

        # Evaluate jacobians and Kalman gain on nominal trajectory
        H_kp1 = self.H_func(x_pre_kp1, t_kp1, id_list=id_list)
        R_list = [self.R_kp1 for _ in range(len(id_list))]
        R_kp1 = block_diag(*R_list)
        K_kp1 = self.kalman_gain(P_pre_kp1, H_kp1, R_kp1)

        innov_cov = H_kp1 @ P_pre_kp1 @ H_kp1.T + R_kp1

        # Apply measurement update
        x_post_kp1 = x_pre_kp1 + K_kp1 @ nl_innov
        P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

        out = {
            'x_pre_kp1': x_pre_kp1,
            'x_post_kp1': x_post_kp1,
            'P_pre_kp1': P_pre_kp1,
            'P_post_kp1': P_post_kp1,
            
            'y_kp1': y_kp1,
            'y_pre_kp1': y_pre_kp1,
            'innov_cov': innov_cov
        }

        return out

