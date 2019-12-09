"""
********************************************************************************

ASEN 5044 Statistical Estimation - Final Project: OD

This main script performs the following:
    - Initializes the system and and its initial conditions
    - Implements and tunes a linearized Kalman filter
    - Implements and tunes an extended Kalman filter

Authors:        Keith Covington, Connor Ott
Created:        09 Dec 2019
Last Modified:  09 Dec 2019

********************************************************************************
"""

import numpy as np
from constants import *


# Initial state
X0 = 6678                   # [km]
Y0 = 0                      # [km]
r0 = sqrt(X0**2 + Y0**2)
X0dot = 0                   # [km/s]
Y0dot = r0 * sqrt(mu/r0**3) # [km/s]
x0 = np.array([X0, Y0, X0dot, Y0dot])


