"""From here to next docstring is just 'splorin junk"""
# for y in ydata[19:22]:
#     pprint(y)

"""

"""
import numpy as np

"""
out = lkf.report_hist(['y_kp1', 
                        'y_nom_kp1',
                        'dy_update', 
                        'dy_est_kp1',
                        'x_post_kp1', 
                        'dy_nom_kp1', 
                        'pre_fit_residual', 
                        'y_pre_est_kp1',
                        'y_post_est_kp1',
                        'x_update', 
                        'x_full_kp1'])

#pprint(lkf.t_hist)

# for pfr in out['x_post_kp1'][200:300]:
#     print(pfr)

# Plot measurement predictions and nominal
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12, 4*p
fig, ax = plt.subplots(p, 1, sharex=True)
start = 0
end = -1
for i in range(p):
    meas_quant1 = [dy[i] for dy in out['y_pre_est_kp1'][start:end]]
    meas_quant2 = [dy[i] for dy in out['y_post_est_kp1'][start:end]]
    meas_quant3 = [dy[i] for dy in out['y_kp1'][start:end]]

    ax[i]. plot(lkf.t_hist[start+1:end], meas_quant3, '-', color='green', label='y')
    ax[i]. plot(lkf.t_hist[start+1:end], meas_quant1, '-', color='orangered', label='y prefit est')
    #ax[i]. plot(lkf.t_hist[start+1:end], meas_quant2, '-', color='dodgerblue', label='y postfit est')
    
ax[0].legend()
# # plt.show()


plt.rcParams['figure.figsize'] = 12, 4*n
# fig, ax = plt.subplots(n, 1, sharex=True)
# start = 0
# end = -1
# for i in range(n):
#     state_quant1 = [x[i] for x in out['x_post_kp1'][start:end]]
#     # state_quant2 = [x[i] for dy in out['x_post_kp1'][start:end]]

#     ax[i]. plot(lkf.t_hist[start+1:end], state_quant1, '-', color='orangered')
#     # ax[i]. plot(lkf.t_hist[start:end], post_state, '-', color='dodgerblue')


# Plot measurement predictions and nominal
fig, ax = plt.subplots(n, 1, sharex=True)
start = 0
end = -1
ax[0].set_title('$x_{nom} + \delta x$')
for i in range(n):
    state_quant1 = [x[i] for x in out['x_full_kp1'][start:end]]
    # state_quant2 = [x[i] for dy in out['x_post_kp1'][start:end]]

    ax[i]. plot(lkf.t_hist[start+1:end], state_quant1, '-', color='orangered')
    # ax[i]. plot(lkf.t_hist[start:end], post_state, '-', color='dodgerblue')

# plt.show()

lkf.plot_hist()

plt.rcParams['figure.figsize'] = 12, 4*n
fig, ax = plt.subplots(n, 1, sharex=True)
start = 0
end = -1
ax[0].set_title('noisey truth trajectories')
for i in range(n):
    for traj in truth_trajectories:
        state_quant1 = [x[i] for x in traj]
        ax[i]. plot(tvec_sim, state_quant1, '-')


plt.rcParams['figure.figsize'] = 12, 4*p
fig, ax = plt.subplots(p, 1, sharex=True)
start = 0
end = -1
for i in range(p):
    for traj_meas in truth_measurements:
        meas_quant1 = [dy[i] for dy in traj_meas]
        ax[i]. plot(tvec_sim[1:], meas_quant1, '-')
    
plt.show()

"""

"""end LKF 'splorin"""



def test(a, b, *args):

    print(a, b)
    inp = args[0] if len(args) is not 0 else 'poop'
    print('inp is {}'.format(inp))



if __name__ == "__main__":

    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr)
    print(arr.T.reshape((-1, )))





