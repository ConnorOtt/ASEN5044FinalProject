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
from numpy import empty, diag, sqrt
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 12


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
        self.dict_th = []

        self.n = self.x_0.shape[0]

    
    @abstractmethod
    def time_update(self): pass


    @abstractmethod
    def meas_update(self): pass


    def kalman_gain(self, P_pre_kp1, H_kp1, R_kp1):
        # Kalman gain at time t = t_{k+1}

        # NOTE: I implemented the R block diag outside Kalman gain since it's
        # kind of problem-specific

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
        out_dict = self.meas_update(x_pre_kp1, P_pre_kp1, y_kp1, t_kp1)

        # Update filter's state estimate
        self.x_hist.append(out_dict['x_post_kp1'])
        self.P_hist.append(out_dict['P_post_kp1'])
        self.dict_th.append(out_dict)


    def report_hist(self, desired_output): 
        """Output time history of everything (requested)"""
        return {k:[el[k] for el in self.dict_th] for k in desired_output}

    def plot_hist(self, savefig=False):
        """
        plot estimate error and 2sigma bounds 
        """

        n = self.n
        plt.rcParams['figure.figsize'] = 12, 4*n
        time_unit = None

        # Create subplots for each state with 2-sigma bounds
        fig, ax = plt.subplots(n, 1, sharex=True)
        ax[0].set_title(u'State Estimate Error w/ $2\sigma$ Bounds')
        ax[-1].set_xlabel('time, {}'.format(time_unit if time_unit is not None else 's'))

        diagP = [diag(P) for P in self.P_hist]
        for i in range(n):
            state = [x[i] for x in self.x_hist]
            two_sig = [2*sqrt(dp[i]) for dp in diagP]
            ax[i].plot(self.t_hist, state, '-', color='dodgerblue')
            ax[i].plot(self.t_hist, [(ts, -ts) for ts in two_sig], '--', color='black')

        plt.show()
        # ax1.plot([x[0] - xt[0] for x, xt in zip(x_th, x_kgte0.T)], color='dodgerblue', label='State Estimate')
        # ax1.plot([2*s[0] for s in P_diag_root], '--', color='black', label='$2\sigma$ Uncertainty\n Bounds')
        # ax1.plot([-2*s[0] for s in P_diag_root], '--', color='black')
        # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # ax1.set_ylabel('Easting, m')


        pass
