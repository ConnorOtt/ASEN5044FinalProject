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
from constants import I, n, p, pi
from system_def import nl_orbit_prop as nom_prop
from pprint import pprint

class LKF(KF):

    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(system)

        # CT nonlinear system functions
        self.f = system["f"]
        self.h = system["h"]

        # LKF-specific properties, here all of these are static
        self.x_nom_0 = system["x_nom_0"]
        self.delta_t = system["dt"]
        self.u_k = np.zeros((2,))  # no control for now or ever lol
        self.Q_k = system["Q"]
        self.R_kp1 = system["R"]

        self.x_nom_k = self.x_nom_0
        self.x_nom_th = [self.x_nom_0]


    def time_update(self, dx_post_k, P_post_k, t_kp1):
        """
        Override the general KF's time update.
        """

        # Do not propagate nominal, it's already caught up from last
        # measurement update
        x_nom_k = self.x_nom_k # use nom at k to linearize and bring state to kp1

        F_k = self.F_func(x_nom_k, self.delta_t)
        G_k = self.G_func(self.delta_t)
        Omega_k = self.Omega_func(self.delta_t)

        dx_pre_kp1 = F_k @ dx_post_k + G_k @ self.u_k
        P_pre_kp1 = F_k @ P_post_k @ F_k.T + Omega_k @ self.Q_k @ Omega_k.T

        return dx_pre_kp1, P_pre_kp1


    def meas_update(self, dx_pre_kp1, P_pre_kp1, y_kp1, t_kp1):
        """
        Override the general KF's measurement update.
        """
        # Bring the nominal up to evaluate H at kp1
        x_nom_kp1 = self.__update_nom()  # do this even if there's no measurement
        id_list = y_kp1['stationID']
        y_kp1 = y_kp1['meas']

        if y_kp1 is None: # NOTE: this is getting ugly, any ideas?
            none_meas = [None for _ in range(p)]
            out = {
                # Necessary for KF to continue
                'x_post_kp1': dx_pre_kp1,
                'P_post_kp1': P_pre_kp1,

                # Necessary to fill in blank spots in outpu
                'x_full_kp1':x_nom_kp1 + dx_pre_kp1,
                'x_nom_kp1': x_nom_kp1,
                'y_kp1': none_meas,
                'y_nom_kp1': none_meas,
                'y_pre_est_kp1': none_meas,
                'innov_cov': None,  # that will not work if it's none but Omega_k
            }

            return out

        # Evaluate jacobians and Kalman gain on nominal trajectory
        H_kp1 = self.H_func(x_nom_kp1, t_kp1, id_list=id_list)
        R_list = [self.R_kp1 for _ in range(int(H_kp1.shape[0]/3))]
        R_kp1 = block_diag(*R_list)
        K_kp1 = self.kalman_gain(P_pre_kp1, H_kp1, R_kp1)

        innov_cov = H_kp1 @ P_pre_kp1 @ H_kp1.T + R_kp1

        # Generate nominal measurement and pre-fit residual
        y_nom_kp1, _ = self.h(x_nom_kp1, t_kp1, id_list=id_list) # nominal measurement
        dy_nom_kp1 = self.__wrap_angle(y_kp1 - y_nom_kp1)  # only operates on y[2]
        dy_est_kp1 = H_kp1 @ dx_pre_kp1
        pre_fit_residual = dy_nom_kp1 - dy_est_kp1;

        # Apply measurement update
        dx_post_kp1 = dx_pre_kp1 + K_kp1 @ pre_fit_residual
        P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

        dy_est_post_km1 = H_kp1 @ dx_post_kp1

        out = {
            # Actually needed for KF
            'x_post_kp1': dx_post_kp1,
            'P_post_kp1': P_post_kp1,

            # Whatever the fuck else you want to output
            'x_full_kp1': x_nom_kp1 + dx_post_kp1,
            'x_nom_kp1': x_nom_kp1,
            'y_kp1':y_kp1,
            'y_pre_est_kp1': y_nom_kp1 + dy_est_kp1,
            'y_nom_kp1': y_nom_kp1,
            'innov_cov': innov_cov,
        }

        return out


    def __update_nom(self):
        """Propagate current nominal trajectory forward
            by one time step for linearization
        """ 

        nom_prop.set_initial_value(self.x_nom_k, 0)
        nom_prop.set_f_params(None, None)  # Not sure if this works yet
        x_nom_kp1 = nom_prop.integrate(self.delta_t)

        self.x_nom_k = x_nom_kp1
        self.x_nom_th.append(x_nom_kp1)

        return x_nom_kp1


    def __wrap_angle(self, diff):

        if diff[2] > pi:
            diff[2] -= 2*pi
        elif diff[2] < -pi:
            diff[2] += 2*pi 

        return diff


