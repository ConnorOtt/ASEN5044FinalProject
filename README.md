# State Estimation for Dynamical Systems - Final Project

This repo holds code and documentation for the ASEN 5044 final project. In this project, 
we implement three variants of the Kalman Filter. These are the Linearized, Extended, and, as extra credit, 
Unscented Kalman Filters. These were used to estimate a spacecraft state in a simplified orbit determination 
problem. The project assignment, and problem definition can be found in the Assignment directory. 

## Filter definitions and uses 
Each filter is defined in its own file (lkf.py, ekf.py, ukf.py) as a Python class. Each of these classes, at minimum,
define time and measurement update methods specific to that filter. The LKF, EKF, and UKF classes each
inherit a general Kalman gain method (calculating Kalman gain matrix), update method, and some 
data handling functionalities from the KF parent class. This KF class is defined in kf.py and handles
some of the more basic, but necessary aspects of practical implementations of Kalman Filters such as 
system definition and data handling (making sure the indices align in the arrays and whatnot.) 

These classe are instantiated with a dictionary containing various system defintions such as the nonlinear and 
linearized mapping functions, initial state, and any extraneous parameters needed for specific filters. An example
is shown below: 

``` 
system = {
    # Required by KF algorithm
    "t_0": t_0, 			# Initial time
    "x_0": dx_est_0,		# Initial state estimate
    "P_0": P_0,				# Initial estimate error covariance
    "Q": Q, 				# Process noise covariance
    "R": R,		 			# Measurement noise covariance
    **dt_jac_eval_funcs,	# Discrete-Time linearized system functions 
    **ct_nl_funcs,			# Continuous-Time nonlinear system functions

    # LKF specific
    "x_nom_0":x_nom_0,		# Initial nominal trajectory (updated internally)
    "dt": 10,				# Time between discrete time steps in system 
}

lkf = LKF(system)
while len(measurements) > 0:  # Loop through measurements 
    y = measurements.pop(0)
    t = time_list.pop(0)
    lkf.update(t, y)  # Perform time and measurement update. 
```
 
## System Definition
The nonlinear and linearized dynamic and measurement functions are defined in ```system_def.py```, and imported to the main functions
to instantiate the filters. More detailed information acn be found in ```system_def.py```.

## Tuning and testing
The majority of validating, tuning, and testing the filters is done in ```main_(filtername).py```. In these files we perform NEES and NIS tests and plot filter estimates and 2-sigma covariance bounds to better understand filter performance. 

## The Report
### Introduction: 

Our full report .tex can be found in ./Report/ and compiled locally with pdflatex. 


