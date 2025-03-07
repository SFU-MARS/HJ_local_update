from direct_numpy import direct_comp   
from decomposition_numpy import decomposition

import numpy as np
import heterocl as hcl
# import cv2

# from set_2plot import *
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

import plotly.express as px

from odp.Grid import Grid
from odp.Shapes import *

from system import couple_u

# The function of computing first order spatial derivative
from update_V_numpy import spa_derivX, spa_derivY

from leaking_corner import ground_truth_L, new_theory_L

import time

num = 101
lookback_length = 0.2

grid, result_true = direct_comp(num, saveAllTimeStep=True, lookback_length=lookback_length)
result_decomp, result_upper, result_lower = decomposition(num, saveAllTimeStep=True, lookback_length=lookback_length)

print('The size of the decomposition result', result_decomp.shape)

true_final = result_true[-1]
decomp_final = result_decomp[-1]
result_diff = decomp_final - true_final

fig = px.imshow(result_diff)
# fig.show()

indice = new_theory_L(result_decomp, result_true[0].copy(), result_upper, result_lower)
indice_gt = ground_truth_L(true_final, decomp_final)


indice = indice
indice_transpose = indice.transpose()
indices_ref = np.ravel_multi_index(indice_transpose, decomp_final.shape)

plot_diff = result_diff.copy()
np.put(plot_diff, indices_ref, 1)
fig1 = px.imshow(plot_diff)
# fig1.show()

# Set computational time steps
t_step = 0.02
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

sys = couple_u(x=[0,0], uMax=1, dMax=0.0, uMode='min', dMode='min')

# Initialize the value function
# data_init = ShapeRectangle(grid, [-1.0, -1.0], [1.0, 1.0])
data_init = result_true[0].copy()

# data = decomp_final.copy()
# np.put(data, indices_ref, data_init[indice_transpose[0], indice_transpose[1]])

data = data_init.copy()

# Plotting Option
po = PlotOptions(do_plot=True, plot_type="value", plotDims=[0,1],
                colorscale="Bluered", save_fig=True, filename="local_update_3", interactive_html=True)


'''
Below is the local updating loop of the value function with respect to the indices
pure numpy version
'''

list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])

data_change = np.zeros(tuple(grid.pts_each_dim))

tNow = tau[0]

start = time.time()

for i in range(1, len(tau)):
    t = np.array([tNow, tau[i]])
    print("Time step: ", t)

    # while tNow < tau[i]:
        # Update the value function

        
    for ind in range(len(indice)):
        # print(indices[i])
        # get indices
        x = indice[ind][0]
        y = indice[ind][1]
        # get spatial derivative
        dV_dx_L, dV_dx_R = spa_derivX(x,y,data,grid)
        dV_dy_L, dV_dy_R = spa_derivY(x,y,data,grid)
        # Get the average gradient
        dV_dx = (dV_dx_L + dV_dx_R)/2
        dV_dy = (dV_dy_L + dV_dy_R)/2
        # get the dynamical rates of change
        uOpt = sys.opt_ctrl_numpy(t, [list_x1[x], list_x2[y]], [dV_dx, dV_dy])
        dx_dt, dy_dt = sys.dynamics_numpy(t, [list_x1[x], list_x2[y]], uOpt, 0)
        # Updating the value function
        data_change[x,y] = (dx_dt*dV_dx + dy_dt*dV_dy)
    
    data = data + data_change*t_step    
    tNow += t_step
    data_change = np.zeros(tuple(grid.pts_each_dim))
    
    data_ref = result_decomp[i].copy()
    np.put(data_ref, indices_ref, data[indice_transpose[0], indice_transpose[1]])
    
    data = data_ref.copy()

    execution_time = time.time() - start
print("Total kernel time local: ", execution_time)
print("Finished updating the value function")


'''
Combination process: local updating result, and decomposition result
'''
result_combine = decomp_final.copy()

np.put(result_combine, indices_ref, data[indice_transpose[0], indice_transpose[1]])
print('Im here')


'''
Comparision with direct computation
'''
# plot_overlay_set(grid, true_final, decomp_final, result_combine, po)
# <<<<<<< HEAD
# np.save('result_true_2d.npy', result_true)
# np.save('result_decomp_2d.npy', result_decomp)
# np.save('result_combine_2d.npy', result_combine)
# np.save('indices_detected', indices_ref)
# =======

print('The total number of points: ', true_final.shape[0]*true_final.shape[1])
print('The number of points getting locally updated: ', indice.shape)
print('Original number of points with different values: ', np.argwhere(abs(decomp_final-true_final)>1e-3).shape)
print('The number of points with different values after local updating: ', np.argwhere(abs(result_combine-true_final)>1e-3).shape)

count = true_final.shape[0]*true_final.shape[1]

# diff_decomp = np.sum(result_diff) / count
# diff_corrected = np.sum(result_combine - true_final) / count
# print('The average error between direct computation and decomposition: ', diff_decomp)
# print('The average error between direct computation and local updates: ', diff_corrected)

diff_decomp = np.sum(abs(result_diff)) / count
diff_corrected = np.sum(abs(result_combine - true_final)) / count
print('The average absolute error between direct computation and decomposition: ', diff_decomp)
print('The average absolute error between direct computation and local updates: ', diff_corrected)

diff_decomp_max = np.max(abs(result_diff))
diff_corrected_max = np.max(abs(result_combine - true_final))
print('The maximum absolute error between direct computation and decomposition: ', diff_decomp_max)
print('The maximum absolute error between direct computation and local updates: ', diff_corrected_max)
