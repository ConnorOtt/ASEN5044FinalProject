"""
Define unscented Kalman filter

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
from scipy.linalg import block_diag, cholesky

# Local imports
from kf import KF
from constants import I, n, p, pi, matprint
from system_def import nl_orbit_prop as nl_prop


class UKF(KF):

    def __init__(self, system):

        # Inherit basic system definition properties from general KF
        super().__init__(system)

        # CT nonlinear system functions
        self.f = system["f"]
        self.h = system["h"]
        self.n = n

        # EKF-specific properties, here all of these are static
        self.u_k = np.zeros((2,))  # no control for now or ever lol
        self.Q_k = system["Q"]
        self.R_kp1 = system["R"]

        self.a = system['a']
        self.b = system['b']
        self.k = system['k']


        self.lam = self.a**2 * (self.n + self.k) - self.n


    def time_update(self, x_post_k, P_post_k, t_kp1):
        """
        Override the general KF's time update
        The UKF uses a nonlinear propagation step for 2n+1
        sigma points, which are sampled from the current covariance. 
        

        The mean and covariance of the propagated sigma points
        are the x_pre_kp1 and P_pre_kp1
        """

        delta_t = t_kp1 - self.t_k

        # Nonlinear state propagation        
        sigmas_post_k = self.__gen_sigma_points(x_post_k, P_post_k)

        sigmas_pre_kp1 = []
        for x_s in sigmas_post_k:
            nl_prop.set_initial_value(x_s, self.t_k)
            nl_prop.set_f_params(None, None)
            nl_prop.integrate(t_kp1)
            sigmas_pre_kp1.append(nl_prop.y)

        x_pre_kp1, P_pre_kp1 = self.__recombine_sigma(sigmas_pre_kp1)

        Omega_k = self.Omega_func(delta_t)
        P_pre_kp1 = P_pre_kp1 + Omega_k @ self.Q_k @ Omega_k.T
        
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
                'x_post_kp1': x_pre_kp1,
                'P_post_kp1': P_pre_kp1,

                'nis': np.nan
            }
            return out

        # Break the sigma points out again
        sigmas_pre_kp1 = self.__gen_sigma_points(x_pre_kp1, P_pre_kp1)

        # Apply non-linear measurement mapping to each sigma point
        y_sigma_pre_kp1 = []
        for x_s in sigmas_pre_kp1:
            y_sig, _ = self.h(x_s, t_kp1, id_list=id_list)
            y_sigma_pre_kp1.append(y_sig)

        y_pre_kp1, Pyy_kp1, Pxy_kp1 = self.__recombine_sigma(sigmas_pre_kp1, y_sigma_pre_kp1)
        R_list = [self.R_kp1 for _ in range(len(id_list))]
        R_kp1 = block_diag(*R_list)
        Pyy_kp1 = Pyy_kp1 + R_kp1

        # Not using the KF kalman gain function
        K_kp1 = Pxy_kp1 @ inv(Pyy_kp1)

        # Nonlinear prefit residuals
        innov = y_kp1 - y_pre_kp1     

        # Apply measurement update
        x_post_kp1 = x_pre_kp1 + K_kp1 @ innov
        P_post_kp1 = P_pre_kp1 - K_kp1 @ Pyy_kp1 @ K_kp1.T

        # calculate nis value - average the long ones
        nis = innov.T @ inv(Pyy_kp1) @ innov
        if len(innov) > p:
            nis /= len(id_list)

        out = {
            'x_post_kp1': x_post_kp1,
            'P_post_kp1': P_post_kp1,

            'nis': nis,
        }

        return out


    def __gen_sigma_points(self, x, P):

        """Generate sigma points sampled from current pdf of the 
        state estimate 
        
        """
        lam = self.lam

        sigma_points = [x]
        S = cholesky(P, lower=False)

        for sj in S:
            pos_xi_j = x + np.sqrt(n + lam) * sj
            neg_xi_j = x - np.sqrt(n + lam) * sj

            sigma_points.extend([pos_xi_j, neg_xi_j])


        return sigma_points

    def __recombine_sigma(self, sigma_points, meas_sigma=None):
        """Calculate mean and covariance of a set of sigma points
    
            If meas_sigma is not None it's a list of measurement sigma points
            that will be used to calculate the y mean, covariance, and cross-
            covariance between x and y sigma points. 
        """
        lam = self.lam
        n = self.n
        a = self.a
        b = self.b

        weights_m = [lam / (n + lam)]
        weights_c = [lam / (n + lam) + 1 - a**2 + b]
        for _ in sigma_points[1:]: # already got the first one
            weights_m.append(1/(2*(n + lam)))
            weights_c.append(1/(2*(n + lam)))

        x = sum([w * xi for w, xi in zip(weights_m, sigma_points)])    
        if meas_sigma is None:
            P = sum([w * np.outer((xi - x), (xi - x)) for w, xi in zip(weights_c, sigma_points)])
        else: 
            y = sum([w * yi for w, yi in zip(weights_m, meas_sigma)])
            Pyy = sum([w * np.outer((yi - y), (yi - y)) for w, yi in zip(weights_c, meas_sigma)])
            Pxy = sum([w * np.outer((xi - x), (yi - y)) for w, xi, yi in zip(weights_c, sigma_points, meas_sigma)])
            return y, Pyy, Pxy

        return x, P






