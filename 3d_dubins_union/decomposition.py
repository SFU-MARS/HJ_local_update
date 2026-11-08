import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *
from odp.solver import HJSolver

from odp.Plots import PlotOptions
from odp.Plots import visualize_plots

from subsystem import subsys_1, subsys_2

import time
import os


def decomposition(num, saveAllTimeStep=True, lookback_length=1.0, t_step=0.05,
                  obstacle_min=(-0.5, -0.5), obstacle_max=(0.5, 0.5), accuracy="medium"):
    r'''
        Decompose the 3D Dubins car
            \dot{x} = v*cos(theta), \dot{y} = v*sin(theta), \dot{theta} = w
        into
            subsystem 1: (x, theta)
            subsystem 2: (y, theta)
        Each subsystem is solved with the optimized_dp HJSolver (uMode='max', avoiding the
        obstacle), and the 3D value function is reconstructed as V(x, y, theta) =
        min(V1(x, theta), V2(y, theta)), i.e. the obstacle is the union of the rectangles
        x_min <= x <= x_max and y_min <= y <= y_max (theta unbounded).
    '''
    # Create 2D grids for the subsystems, theta is periodic
    g_1 = Grid(np.array([-4.0, -math.pi]), np.array([4.0, math.pi]), 2, np.array([num, num]), [1])
    g_2 = Grid(np.array([-4.0, -math.pi]), np.array([4.0, math.pi]), 2, np.array([num, num]), [1])

    # Initialize value function (negative inside the obstacle)
    data_sub_1 = ShapeRectangle(g_1, np.array([obstacle_min[0], -np.inf]), np.array([obstacle_max[0], np.inf]))
    data_sub_2 = ShapeRectangle(g_2, np.array([obstacle_min[1], -np.inf]), np.array([obstacle_max[1], np.inf]))

    ## Look-back length and time step of computation
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics, control tries to avoid the obstacle
    sys_1 = subsys_1(v=1, wMax=1, dMax=0.0, uMode='max', dMode='min')
    sys_2 = subsys_2(v=1, wMax=1, dMax=0.0, uMode='max', dMode='min')

    # Backward reachable tube of the obstacle
    compMethods = {"TargetSetMode": "None"}

    start = time.time()

    '''
    Computation in subsystem 1
    '''
    result_1 = HJSolver(sys_1, g_1, data_sub_1, tau, compMethods,
                        saveAllTimeSteps=saveAllTimeStep, accuracy=accuracy)

    '''
    Computation in subsystem 2
    '''
    result_2 = HJSolver(sys_2, g_2, data_sub_2, tau, compMethods,
                        saveAllTimeSteps=saveAllTimeStep, accuracy=accuracy)

    '''
    Combine the results from 2 subsystems
    '''
    grid_min = np.array([-4.0, -4.0, -math.pi])
    grid_max = np.array([4.0, 4.0, math.pi])
    dims = grid_min.shape[0]
    N = np.array([num, num, num])
    g = Grid(grid_min, grid_max, dims, N, periodicDims=[2])

    if not saveAllTimeStep:
        # result_1: (x, theta), result_2: (y, theta) -> (x, y, theta)
        result_upper = result_1[:, np.newaxis, :]
        result_lower = result_2[np.newaxis, :, :]
    else:
        # HJSolver stores the initial value at index -1, flip so that index 0 is t = 0
        data_list_1 = np.flip(result_1, axis=-1)
        data_list_2 = np.flip(result_2, axis=-1)

        # (x, theta, t), (y, theta, t) -> (x, y, theta, t)
        result_upper = data_list_1[:, np.newaxis, :, :]
        result_lower = data_list_2[np.newaxis, :, :, :]

    result_full = np.minimum(result_upper, result_lower)
    result_upper = np.broadcast_to(result_upper, result_full.shape)
    result_lower = np.broadcast_to(result_lower, result_full.shape)

    execution_time = time.time() - start

    print("Total kernel time decomposition: ", execution_time)
    print("Finished updating the value function")

    np.save('result_decomp_3d.npy', result_full)

    # Visualization of 3D value function
    po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2], slicesCut=[],colorscale="Bluered", 
                    save_fig=True, filename="plots/decomposition_result", interactive_html=True)
    visualize_plots(result_full, g, po)

    return result_full, result_upper, result_lower, g


if __name__ == "__main__":
    if os.path.exists("plots") == False:
        os.mkdir("plots")
    decomposition(41, saveAllTimeStep=True, lookback_length=1.0)
