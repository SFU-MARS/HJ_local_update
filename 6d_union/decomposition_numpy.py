import imp
import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *

from subsystem import subsys_1, subsys_2
from spatial_V_numpy import spa_deriv4X1, spa_deriv4X2, spa_deriv4X3, spa_deriv4X4

# Plot Options
from odp.Plots import *
import time

def decomposition(num, saveAllTimeStep=True, lookback_length=0.02):
    # num = 51  
    # Create Grid
    grid_min = np.array([-1.0, -2.0, -2.0, -2.0])
    grid_max = np.array([4.0, 2.0, 2.0, 2.0])
    dims = grid_min.shape[0]
    N = np.array([num, num, num, num])
    g = Grid(grid_min, grid_max, dims, N)
    
    # Initialize value function
    data_sub_1 = ShapeRectangle(g, [-1.0, -3.0, -1.0, -3.0], [0.0, 3.0, 1.0, 3.0])
    data_sub_2 = ShapeRectangle(g, [-1.0, -3.0, -1.0, -3.0], [0.0, 3.0, 1.0, 3.0])
    
    # Visualization
    po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0],
                    colorscale="Bluered", save_fig=True, filename="direct_2Int_leaking", interactive_html=True)
    
    
    ## Look-back length and time step of computation
    # lookback_length = 0.02
    t_step = 0.02
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
    
    # Set system dynamics
    sys_1 = subsys_1(uTMax=1, utMax=1, dMax=0.0, uMode='max', dMode='min')
    sys_2 = subsys_2(uTMax=1, utMax=1, dMax=0.0, uMode='max', dMode='min')
    
    list_x0 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x1 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x2 = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_x3 = np.reshape(g.vs[3], g.pts_each_dim[3])
    
    data_change = np.zeros(tuple(g.pts_each_dim))
    
    # Set computation task and compute HJ PDE
    '''
    Computation in subsystem 1
    '''
    if saveAllTimeStep:
        data_list_1 = []
        data_list_1.append(data_sub_1.copy())      
        # print('The shape of data list is: ', data_list)
        
    tNow = tau[0]
    start = time.time()
    
    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)
        
        # while tNow < tau[i]:
            # Update the value function
        for x0 in range(len(list_x0)):
            for x1 in range(len(list_x1)):
                for x2 in range(len(list_x2)):
                    for x3 in range(len(list_x3)):
                        # Get spatial derivative
                        dV_d0_L, dV_d0_R = spa_deriv4X1(x0, x1, x2, x3, data_sub_1, g)
                        dV_d1_L, dV_d1_R = spa_deriv4X2(x0, x1, x2, x3, data_sub_1, g)
                        dV_d2_L, dV_d2_R = spa_deriv4X3(x0, x1, x2, x3, data_sub_1, g)
                        dV_d3_L, dV_d3_R = spa_deriv4X4(x0, x1, x2, x3, data_sub_1, g)
                        
                        # Get the average gradient
                        dV_d0 = (dV_d0_L + dV_d0_R)/2
                        dV_d1 = (dV_d1_L + dV_d1_R)/2
                        dV_d2 = (dV_d2_L + dV_d2_R)/2
                        dV_d3 = (dV_d3_L + dV_d3_R)/2
                        
                        # Get the dynamical rates of change
                        uTOpt, utOpt = sys_1.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [dV_d0, dV_d1, dV_d2, dV_d3])
                        d0_dt, d1_dt, d2_dt, d3_dt = sys_1.dynamics_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [uTOpt, utOpt], 0)
                        
                        # Updating the value function
                        data_change[x0, x1, x2, x3] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3
                        
        data_sub_1 = data_sub_1 + t_step * data_change
        if saveAllTimeStep:
            data_list_1.append(data_sub_1)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow = tNow + t_step
        
    '''
    Computation in subsystem 2
    '''
    if saveAllTimeStep:
        data_list_2 = []
        data_list_2.append(data_sub_2.copy())
        # print('The shape of data list is: ', data_list)
        
    tNow = tau[0]
    # start = time.time()
    
    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)
        
        # while tNow < tau[i]:
            # Update the value function
        for x0 in range(len(list_x0)):
            for x1 in range(len(list_x1)):
                for x2 in range(len(list_x2)):
                    for x3 in range(len(list_x3)):
                        
                        # Get spatial derivative
                        dV_d0_L, dV_d0_R = spa_deriv4X1(x0, x1, x2, x3, data_sub_2, g)
                        dV_d1_L, dV_d1_R = spa_deriv4X2(x0, x1, x2, x3, data_sub_2, g)
                        dV_d2_L, dV_d2_R = spa_deriv4X3(x0, x1, x2, x3, data_sub_2, g)
                        dV_d3_L, dV_d3_R = spa_deriv4X4(x0, x1, x2, x3, data_sub_2, g)
                        
                        # Get the average gradient
                        dV_d0 = (dV_d0_L + dV_d0_R)/2
                        dV_d1 = (dV_d1_L + dV_d1_R)/2
                        dV_d2 = (dV_d2_L + dV_d2_R)/2
                        dV_d3 = (dV_d3_L + dV_d3_R)/2
                        
                        # Get the dynamical rates of change
                        uTOpt, utOpt = sys_2.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [dV_d0, dV_d1, dV_d2, dV_d3])
                        d0_dt, d1_dt, d2_dt, d3_dt = sys_2.dynamics_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [uTOpt, utOpt], 0)
                        
                        # Updating the value function
                        data_change[x0, x1, x2, x3] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3
                        
        data_sub_2 = data_sub_2 + t_step * data_change
        if saveAllTimeStep:
            data_list_2.append(data_sub_2)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow = tNow + t_step
            
    # execution_time = time.time() - start
    
    # print("Total kernel time: ", execution_time)
    # print("Finished updating the value function")
    
    
    '''
    Combine the results from 2 subsystems
    '''
        
    grid_min = np.array([-4.0, -4.0, -4.0, -4.0, -math.pi, -4.0])
    grid_max = np.array([4.0, 4.0, 4.0, 4.0, math.pi, 4.0])
    dims = grid_min.shape[0]
    pDim = [4]
    N = np.array([num, num, num, num, num, num])
    g = Grid(grid_min, grid_max, dims, N, periodicDims=pDim)
    
    if not saveAllTimeStep:
        result_upper_flip = np.flip(data_sub_1, axis=0) 
        result_upper_expand = np.tile(result_upper_flip, (num, num, 1, 1, 1, 1))
        print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (2, 0, 3, 1, 4, 5))

        result_lower_flip = np.flip(data_sub_2, axis=0)
        result_lower_expand = np.tile(result_lower_flip, (num, num, 1, 1, 1, 1))
        print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (0,2,1,3,4,5))

        result_full = np.maximum(result_upper, result_lower)   
        
    else:
        
        # result_upper_flip = np.flip(data_list_1, axis=0) 
        result_upper_expand = np.tile(data_list_1, (num, num, 1, 1, 1, 1, 1))
        # print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (2, 3, 0, 4, 1, 5, 6))
        # print(result_upper.shape)

        # result_lower_flip = np.flip(data_list_2, axis=0)
        result_lower_expand = np.tile(data_list_2, (num, num, 1, 1, 1, 1, 1))
        # print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (2, 0, 3, 1, 4, 5, 6))
        # print(result_lower.shape)

        result_full = np.maximum(result_upper, result_lower)
        
        # print('Decomposition Correct')
        
    execution_time = time.time() - start
    
    print("Total kernel time decomposition: ", execution_time)
    print("Finished updating the value function")
        
    return result_full, result_upper, result_lower
            
    
if __name__ == "__main__":
    decomposition(21, saveAllTimeStep=True, lookback_length=0.02)