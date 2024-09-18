import imp
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from subsystem import subsys
from update_V_numpy import spa_deriv

# Plot Options
from odp.Plots import *
import time

def decomposition(num, saveAllTimeStep=True, lookback_length=0.02):
    # num = 51
    
    # Create Grid
    grid_min = np.array([-4.0])
    grid_max = np.array([4.0])
    dims = grid_min.shape[0]
    N = np.array([num])
    g = Grid(grid_min, grid_max, dims, N)
    
    # Initialize value function
    data_sub = ShapeRectangle(g, [-1.0], [1.0])
    
    # Visualization
    po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0],
                    colorscale="Bluered", save_fig=True, filename="direct_2Int_leaking", interactive_html=True)
    
    ## Look-back length and time step of computation
    # lookback_length = 0.02
    t_step = 0.02
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
    
    # Set system dynamics
    sys = subsys(x=[0], uMax=1, dMax=0.0, uMode='min', dMode='min')
    
    list_x = np.reshape(g.vs[0], g.pts_each_dim[0])
    
    data_change = np.zeros(tuple(g.pts_each_dim))
    
    # Set computation task and compute HJ PDE
    '''
    Computation in subsystems
    '''
    if saveAllTimeStep:
        data_list = []
        data_list.append(data_sub)
        # print('The shape of data list is: ', data_list)
        
    tNow = tau[0]
    start = time.time()
    
    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)
        
        # while tNow < tau[i]:
            # Update the value function
        for x in range(len(list_x)):
            # Get spatial derivative
            dV_dx_L, dV_dx_R = spa_deriv(x, data_sub, g)
            # Get the average gradient
            dV_dx = (dV_dx_L + dV_dx_R)/2
            # Get the dynamical rates of change
            uOpt = sys.opt_ctrl_numpy(t, [list_x[x]], dV_dx)
            dx_dt = sys.dynamics_numpy(t, [list_x[x]], uOpt, 0)
            
            # Updating the value function
            data_change[x] = dx_dt*dV_dx
                        
        data_sub = data_sub + t_step * data_change
        if saveAllTimeStep:
            data_list.append(data_sub)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow = tNow + t_step
            
    execution_time = time.time() - start
    
    print("Total kernel time: ", execution_time)
    print("Finished updating the value function")
    
    
    '''
    Combine the results from 2 subsystems
    '''
        
    grid_min = np.array([-4.0, -4.0])
    grid_max = np.array([4.0, 4.0])
    dims = grid_min.shape[0]
    N = np.array([num, num])
    g = Grid(grid_min, grid_max, dims, N)
    
    if not saveAllTimeStep:
        result_upper_flip = np.flip(data_sub, axis=0) 
        result_upper_expand = np.tile(result_upper_flip, (num, 1))
        print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (1,0))

        result_lower_flip = np.flip(data_sub, axis=0)
        result_lower_expand = np.tile(result_lower_flip, (num, 1))
        print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (0,1))

        result_full = np.maximum(result_upper, result_lower)   
        
    else:
        
        result_upper_flip = np.flip(data_list, axis=0) 
        result_upper_expand = np.tile(data_list, (num, 1, 1))
        # print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (1,2,0))
        # print(result_upper.shape)

        result_lower_flip = np.flip(data_list, axis=0)
        result_lower_expand = np.tile(data_list, (num, 1, 1))
        # print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (1,0,2))
        # print(result_lower.shape)

        result_full = np.maximum(result_upper, result_lower)
        
        # print('Decomposition Correct')
        
    return result_full
            
    