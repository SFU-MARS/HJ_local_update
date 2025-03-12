from direct_numpy import direct_comp   
from decomposition_numpy import decomposition

import numpy as np
import heterocl as hcl
import math
# import cv2

from set_2plot import *
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

import plotly.express as px

from odp.Grid import Grid
from odp.Shapes import *

from system import quadrotor

# The function of computing first order spatial derivative
from spatial_V_numpy import spa_deriv6X1, spa_deriv6X2, spa_deriv6X3, spa_deriv6X4, spa_deriv6X5, spa_deriv6X6

from leaking_corner import *

import time

num = 7
lookback_length = 0.02

grid, result_true = direct_comp(num, saveAllTimeStep=True, lookback_length=lookback_length)
result_decomp, result_upper, result_lower = decomposition(num, saveAllTimeStep=True, lookback_length=lookback_length)

print('The size of the decomposition result', result_decomp.shape)

decomp_final = result_decomp[-1]
true_final = result_true[-1]
result_diff = decomp_final - true_final

# fig = px.imshow(result_diff)
# fig.show()

# Create Grid
grid_min = np.array([-1.0, -1.0, -2.0, -2.0, -2.0, -2.0])
grid_max = np.array([4.0, 4.0, 2.0, 2.0, 2.0, 2.0])
dims = grid_min.shape[0]
N = np.array([num, num, num, num, num, num])
grid = Grid(grid_min, grid_max, dims, N)

## Initialize value function
data1 = Lower_Half_Space(grid, 0, 0)
data2 = Lower_Half_Space(grid, 1, 0)
data3 = Lower_Half_Space(grid, 4, -1)
data4 = Upper_Half_Space(grid, 4, 1)
data = np.minimum(data1, data2)
# data = np.minimum(data, data3)
# data = np.minimum(data, data4)



# data = np.minimum(data_upper, data_lower)  
delta = ground_truth_delta(true_final, decomp_final, result_upper[-1], result_lower[-1]) 


indice = new_theory_L(result_decomp, data, result_upper, result_lower)
# indice_gt = ground_truth_L(true_final, decomp_final)


indice = indice
# print(indice.shape)
indice_transpose = indice.transpose()
# print(indice_transpose.shape)
indices_ref = np.ravel_multi_index(indice_transpose, decomp_final.shape)
# print(indices_ref.shape)

# plot_diff = result_diff.copy()
# np.put(plot_diff, indices_ref, 1)
# fig1 = px.imshow(plot_diff)
# fig1.show()

# Set computational time steps
t_step = 0.02
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

sys = quadrotor(uTMax=1, utMax=1, dMax=0.0, uMode='max', dMode='min')

# Initialize the value function
# data_init = ShapeRectangle(grid, [-1.0, -1.0, -1.0, -1.0, -math.pi, -1.0], [1.0, 1.0, 1.0, 1.0, math.pi, 1.0])
# data_init = data

# data = decomp_final.copy()
# np.put(data, indices_ref, data_init[indice_transpose[0], indice_transpose[1]])

# data = data_init.copy()


'''
Below is the local updating loop of the value function with respect to the indices
pure numpy version
'''
list_x0 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
list_x1 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
list_x2 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
list_x3 = np.reshape(grid.vs[3], grid.pts_each_dim[3])
list_x4 = np.reshape(grid.vs[4], grid.pts_each_dim[4])
list_x5 = np.reshape(grid.vs[5], grid.pts_each_dim[5])

data_change = np.zeros(tuple(grid.pts_each_dim))

tNow = tau[0]

start = time.time()

for i in range(1, len(tau)):
    t = np.array([tNow, tau[i]])
    print("Time step: ", t)

        
    for ind in range(len(indice)):
        # print(indices[i])
        # get indices
        x0 = indice[ind][0]
        x1 = indice[ind][1]
        x2 = indice[ind][2]
        x3 = indice[ind][3]
        x4 = indice[ind][4]
        x5 = indice[ind][5]
        
        # get spatial derivative
        dV_d0_L, dV_d0_R = spa_deriv6X1(x0,x1,x2,x3,x4,x5,data,grid)
        dV_d1_L, dV_d1_R = spa_deriv6X2(x0,x1,x2,x3,x4,x5,data,grid)
        dV_d2_L, dV_d2_R = spa_deriv6X3(x0,x1,x2,x3,x4,x5,data,grid)
        dV_d3_L, dV_d3_R = spa_deriv6X4(x0,x1,x2,x3,x4,x5,data,grid)
        dV_d4_L, dV_d4_R = spa_deriv6X5(x0,x1,x2,x3,x4,x5,data,grid)
        dV_d5_L, dV_d5_R = spa_deriv6X6(x0,x1,x2,x3,x4,x5,data,grid)
        
        
        # Get the average gradient
        dV_d0 = (dV_d0_L + dV_d0_R)/2
        dV_d1 = (dV_d1_L + dV_d1_R)/2
        dV_d2 = (dV_d2_L + dV_d2_R)/2
        dV_d3 = (dV_d3_L + dV_d3_R)/2
        dV_d4 = (dV_d4_L + dV_d4_R)/2
        dV_d5 = (dV_d5_L + dV_d5_R)/2
        
        # get the dynamical rates of change
        uTOpt, utOpt = sys.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3], list_x4[x4], list_x5[x5]], [dV_d0, dV_d1, dV_d2, dV_d3, dV_d4, dV_d5])
        d0_dt, d1_dt, d2_dt, d3_dt, d4_dt, d5_dt = sys.dynamics_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3], list_x4[x4], list_x5[x5]], [uTOpt, utOpt], 0)
        # Updating the value function
        data_change[x0, x1, x2, x3, x4, x5] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3 + d4_dt*dV_d4 + d5_dt*dV_d5
    
    data = data + data_change*t_step    
    tNow += t_step
    data_change = np.zeros(tuple(grid.pts_each_dim))
    
    data_ref = result_decomp[i].copy()
    np.put(data_ref, indices_ref, data[indice_transpose[0], indice_transpose[1], indice_transpose[2], indice_transpose[3], indice_transpose[4], indice_transpose[5]])
    
    data = data_ref.copy()

    execution_time = time.time() - start
print("Total kernel time local: ", execution_time)
print("Finished updating the value function")


'''
Combination process: local updating result, and decomposition result
'''
result_combine = decomp_final.copy()

np.put(result_combine, indices_ref, data[indice_transpose[0], indice_transpose[1], indice_transpose[2], indice_transpose[3], indice_transpose[4], indice_transpose[5]])
print('Im here')


'''
Comparision with direct computation
'''
# po = PlotOptions(dims_plot=[0,1,3], slices=[10,10,10],opacity=0.5,
#                 colorscale="Bluered",)
# plot_overlay_3dsets(grid, true_final, decomp_final, po)
# plot_overlay_3dsets(grid, true_final, result_combine, po)


# grid, result_true = direct_comp(num, saveAllTimeStep=True, lookback_length=lookback_length)
# true_final = result_true[-1]

# np.save('result_direct_6d.npy', result_true)
np.save('result_decomp_6d_002.npy', result_decomp)
np.save('result_local_6d_002.npy', result_combine)

print('The total number of points: ', true_final.shape[0]*true_final.shape[1]*true_final.shape[2]*true_final.shape[3]*true_final.shape[4]*true_final.shape[5])  
print('The number of points getting locally updated: ', indice.shape)
print('Original number of points with different values: ', np.argwhere(abs(decomp_final-true_final)>1e-6).shape)
print('The number of points with different values after local updating: ', np.argwhere(abs(result_combine-true_final)>1e-6).shape)

count = true_final.shape[0]*true_final.shape[1]

diff_decomp = np.sum(result_diff) / count
diff_corrected = np.sum(result_combine - true_final) / count
print('The average error between direct computation and decomposition: ', diff_decomp)
print('The average error between direct computation and local updates: ', diff_corrected)

diff_decomp = np.sum(abs(result_diff)) / count
diff_corrected = np.sum(abs(result_combine - true_final)) / count
print('The average absolute error between direct computation and decomposition: ', diff_decomp)
print('The average absolute error between direct computation and local updates: ', diff_corrected)

diff_decomp_max = np.max(abs(result_diff))
diff_corrected_max = np.max(abs(result_combine - true_final))
print('The maximum absolute error between direct computation and decomposition: ', diff_decomp_max)
print('The maximum absolute error between direct computation and local updates: ', diff_corrected_max)
