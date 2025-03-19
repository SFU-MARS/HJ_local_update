import imp
import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *

from system import quadrotor
from spatial_V_numpy import spa_deriv6X1, spa_deriv6X2, spa_deriv6X3, spa_deriv6X4, spa_deriv6X5, spa_deriv6X6

# Plot Options
from odp.Plots import *
# Solver Core
from odp.solver import HJSolver

import time


def direct_comp(num, saveAllTimeStep=True, lookback_length=0.02):

# num = 101
# saveAllTimeStep = True
    # Create Grid
    grid_min = np.array([-1.0, -1.0, -2.0, -2.0, -2.0, -2.0])
    grid_max = np.array([4.0, 4.0, 2.0, 2.0, 2.0, 2.0])
    dims = grid_min.shape[0]
    N = np.array([num, num, num, num, num, num])
    g = Grid(grid_min, grid_max, dims, N)

    ## Initialize value function
    data1 = Lower_Half_Space(g, 0, 0)
    data2 = Lower_Half_Space(g, 1, 0)
    data3 = Lower_Half_Space(g, 4, -1)
    data4 = Upper_Half_Space(g, 4, 1)
    data = np.minimum(data1, data2)
    # data = np.minimum(data, data3)
    # data = np.minimum(data, data4)
    
    # lookback_length = 0.02
    t_step = 0.02
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    sys = quadrotor(uTMax=1, utMax=1, dMax=0.0, uMode='max', dMode='min')
    

    '''
    Direct updating loop
    '''
    if saveAllTimeStep:
        data_list = []
        data_list.append(data)
        # print('The shape of data list is: ', len(data_list))

    list_x0 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x1 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x2 = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_x3 = np.reshape(g.vs[3], g.pts_each_dim[3])
    list_x4 = np.reshape(g.vs[4], g.pts_each_dim[4])
    list_x5 = np.reshape(g.vs[5], g.pts_each_dim[5])

    data_change = np.zeros(tuple(g.pts_each_dim))

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
                       for x4 in range(len(list_x4)):
                           for x5 in range(len(list_x5)): 

                                # Get spatial derivative
                                dV_d0_L, dV_d0_R = spa_deriv6X1(x0, x1, x2, x3, x4, x5, data, g)
                                dV_d1_L, dV_d1_R = spa_deriv6X2(x0, x1, x2, x3, x4, x5, data, g)
                                dV_d2_L, dV_d2_R = spa_deriv6X3(x0, x1, x2, x3, x4, x5, data, g)
                                dV_d3_L, dV_d3_R = spa_deriv6X4(x0, x1, x2, x3, x4, x5, data, g)
                                dV_d4_L, dV_d4_R = spa_deriv6X5(x0, x1, x2, x3, x4, x5, data, g)
                                dV_d5_L, dV_d5_R = spa_deriv6X6(x0, x1, x2, x3, x4, x5, data, g)
                                
                                # Get the average gradient
                                dV_d0 = (dV_d0_L + dV_d0_R)/2
                                dV_d1 = (dV_d1_L + dV_d1_R)/2
                                dV_d2 = (dV_d2_L + dV_d2_R)/2
                                dV_d3 = (dV_d3_L + dV_d3_R)/2
                                dV_d4 = (dV_d4_L + dV_d4_R)/2
                                dV_d5 = (dV_d5_L + dV_d5_R)/2
                                
                                # Get the dynamical rates of change
                                uTOpt, utOpt = sys.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3], list_x4[x4], list_x5[x5]], [dV_d0, dV_d1, dV_d2, dV_d3, dV_d4, dV_d5])
                                d0_dt, d1_dt, d2_dt, d3_dt, d4_dt, d5_dt = sys.dynamics_numpy(t,  [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3], list_x4[x4], list_x5[x5]], [uTOpt, utOpt], 0)
                                
                                # Updating the value function
                                data_change[x0, x1, x2, x3, x4, x5] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3 + d4_dt*dV_d4 + d5_dt*dV_d5
                
        data = data + data_change*t_step
        if saveAllTimeStep:
            data_list.append(data)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow += t_step
        # print('The shape of data list is: ', len(data_list))
    execution_time = time.time() - start

    # print('The shape of data list is: ', data_list.shape)
        
    print("Total kernel time direct: ", execution_time)
    print("Finished updating the value function")
    
    if saveAllTimeStep:
        return g, data_list

    return g, data

if __name__ == "__main__":
    direct_comp(21)

    
    

