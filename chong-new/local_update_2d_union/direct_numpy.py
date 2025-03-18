import imp
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from system import couple_u
from update_V_numpy import spa_derivX, spa_derivY

# Plot Options
from odp.Plots import *
# Solver Core
from odp.solver import HJSolver

import time

def direct_comp(num, saveAllTimeStep=True, lookback_length=0.02):

    # Create Grid
    grid_min = np.array([-4.0, -4.0])
    grid_max = np.array([4.0, 4.0])
    dims = grid_min.shape[0]
    N = np.array([num, num])
    g = Grid(grid_min, grid_max, dims, N)

    ## Initialize value function
    data_1 = ShapeRectangle(g, [-5.0, -1.0], [5.0, 1.0])
    data_2 = ShapeRectangle(g, [-1.0, -5.0], [1.0, 5.0])
    data = np.minimum(data_1, data_2)
    # data = CylinderShape(g, [], [0.0, 0.0], 1)

    # lookback_length = 0.02
    t_step = 0.02
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    sys = couple_u(x=[0,0], uMax=1, dMax=0.0, uMode='max', dMode='min')

    '''
    Direct updating loop
    '''
    if saveAllTimeStep:
        data_list = []
        data_list.append(data)
        # print('The shape of data list is: ', len(data_list))

    list_x1 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x2 = np.reshape(g.vs[1], g.pts_each_dim[1])

    data_change = np.zeros(tuple(g.pts_each_dim))

    tNow = tau[0]

    start = time.time()

    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)

        # while tNow < tau[i]:
            # Update the value function

            
        for x in range(len(list_x1)):
            for y in range(len(list_x2)):

                # Get spatial derivative
                dV_dx_L, dV_dx_R = spa_derivX(x, y, data, g)
                dV_dy_L, dV_dy_R = spa_derivY(x, y, data, g)
                
                # Get the average gradient
                dV_dx = (dV_dx_L + dV_dx_R)/2
                dV_dy = (dV_dy_L + dV_dy_R)/2
                
                # Get the dynamical rates of change
                uOpt = sys.opt_ctrl_numpy(t, [list_x1[x], list_x2[y]], [dV_dx, dV_dy])
                dx_dt, dy_dt = sys.dynamics_numpy(t, [list_x1[x], list_x2[y]], uOpt, 0)
                
                # Updating the value function
                data_change[x, y] = (dx_dt*dV_dx + dy_dt*dV_dy)
                
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


    
    

