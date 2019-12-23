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
to instantiate the filters. More detailed information can be found in ```system_def.py```.

## Tuning and testing
The majority of validating, tuning, and testing the filters is done in ```main_(filtername).py```. In these files we perform NEES and NIS tests and plot filter estimates and 2-sigma covariance bounds to better understand filter performance. These scripts rely on data generated in 
```truth_gen_save.py```, as well as some variables from ```constants.py``` and functions from ```system_def.py```.

## The Report
#### Introduction: 
>**In order to keep track of Earth-orbiting objects, a typical observation scheme includes ground stations which measure range and range-rate to a passing satellite. 
These measurements, when combined with the known locations of the stations, allows for precises estimates of a satellites orbit state. 
In addition to an orbit estimate, it's usually useful to quantify the uncertainty in the estimate. 
This quantity is a based on the uncertainty in the motion of the satellite in orbit, and the uncertainty in the incoming measurements. 
Neither of these uncertainties are necessarily known.
In order to accurately predict an estimate uncertainty, both of these uncertainties must be balanced with each other. 
This is frequently carried out with predictor-corrector estimation algorithm such as the Linearized Kalman Filter (LKF) or Extended Kalman Filter (EKF.)**
>
>**In this report, we explore the performance of the LKF and EKF on nonlinear systems with dynamic uncertainty (process noise) and measurement uncertainty (measurement nose.) 
In developing these algorithms, we assume the process and measurement noise are unknown.
We then iterate on different process noise and measurement noise covariance matrices and evaluate the algorithms using Normalized Estimation Error Squared (NEES) and Normalized Innovation Squared (NIS) tests until suitable values for uncertainty are found.**


Our full, final report be found in ./Report/ as CovingtonOtt_FinalProject.pdf.  


## Dependencies
This codebase requires no packages outside Python's standard library. 


## Authors
Feel free to contact either of the authors at the contact information below with quesitions
or comments about this repo. 
If you would like to point out an error in this code, we ask that you do so gently. We are but 
delicate engineering students, after all.  

Blaine Covington - Master's Student at the University of Colorado - Boulder 	
(kbcovingtonjr@gmail.com)

Connor Ott - Master's Student at the University of Colorado - Boulder
(connor.ott@colorado.edu)



