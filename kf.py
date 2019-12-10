"""
Define general Kalman filter
"""
#NOTE: 0% chance that this runs as of now...just throwin around ideas

from numpy.linalg import inv


class KF:

    def __init__(self, system):

        # System definition
        self.F_func = system['F']
        self.G_func = system['G']
        self.H_func = system['H']
        self.Omega_func = system['Omega']

        # State estimate history
        self.t_hist = []
        self.x_hist = []
        self.P_hist = []
        self.y_hist = []


    def update(self, tk, yk):
        """
        Updates system's state estimate by processing an incoming measurement
         - Performs the time update (propagation to time of measurement)
         - Performs the measurement update (correction)

        Output is new state estimate (mean and covariance) at time of measurement
        """
        x_pre_kp1, P_pre_kp1 = time_update(x_post_k, P_post_k)
        x_post_kp1, P_post_kp1 = meas_update(x_pre_kp1, P_pre_kp1, y_kp1, R_kp1)

        # Update filter's state estimate
        self.t_hist.append(tk)
        self.y_hist.append(yk)
        self.x_hist.append(x_post_kp1)
        self.P_hist.append(P_post_kp1)


    def kalman_gain(self, P_pre_kp1, H_kp1, R_kp1, **kwargs):
        # Kalman gain at time t = t_{k+1}
        K_kp1 = P_pre_kp1@H_kp1 @ inv(H_kp1@P_pre_kp1@H_kp1.T + R_kp1)
        return K_kp1



    # NOTE: time_update() and meas_update() are not actually used in this 
    #       project, so this hasn't actually been tested. (Beware if 
    #       adapting this code for another project.)
    """
    def time_update(dx_post_k, P_post_k, **kwargs):
        # time update to bring x and P up to k+1 from k
        t_k = kwargs['t_k']
        u_k = kwargs['u_k']
        Q_k = kwargs['Q_k']

        # Assumes F and G are only functions of t
        F_k = F_func(t_k)
        G_k = G_func(t_k)
        Omega_k = Omega_func(t_k)

        x_pre_kp1 = F_k @ x_post_k + G_k @ u_k
        P_pre_kp1 = F_k @ x_post_k @ F_k.T + Q_k

        return x_pre_kp1, P_pre_kp1


    def meas_update(x_pre_kp1, P_pre_kp1, y_kp1, R_kp1, **kwargs):
        # Apply measurement update for LKF
        id_list = kwargs['id_list']
        t_kp1 = kwargs['t_kp1']

        # Evaluate H and K at time t_kp1
        H_kp1 = H_func(t_kp1, id_list=id_list)
        K_kp1 = kalman_gain(P_pre_kp1, H_kp1, R_kp1)

        # Generate nominal measurement and pre-fit residual
        y_nom_kp1 = h(t_kp1, id_list=id_list) # nominal measurement
        dy_kp1 = y_kp1 - y_nom_kp
        pre_fit_residual = dy_kp1 - H_kp1 @ x_pre_kp1;

        # Apply mu
        x_post_kp1 = x_pre_kp1 + K_kp1 @ pre_fit_residual
        P_post_kp1 = (I - K_kp1 @ H_kp1) @ P_pre_kp1

        # TODO: Package some of this up into a dict as output to include pre/post-fit 
        # residuals, pre/post measurement update stats and maybe some of the evaluated
        # jacobians. The dict can be parsed once all of them are collected and the 
        # filter finishes.

        return x_post_kp1, P_post_kp1
    """






