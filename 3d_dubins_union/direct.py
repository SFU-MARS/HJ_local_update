import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *
from odp.solver import HJSolver
from odp.dynamics import DubinsCar

from odp.Plots import PlotOptions
from odp.Plots import visualize_plots

import time
import os


def direct_comp(num, saveAllTimeStep=True, lookback_length=1.0, t_step=0.05,
                obstacle_min=(-0.5, -0.5), obstacle_max=(0.5, 0.5), accuracy="medium"):
    r'''
        Directly solve the 3D Dubins car
            \dot{x} = v*cos(theta), \dot{y} = v*sin(theta), \dot{theta} = w
        with the optimized_dp HJSolver (uMode='max', avoiding the obstacle). The obstacle is
        the same as in decomposition.py: the union of the rectangles
        x_min <= x <= x_max and y_min <= y <= y_max (theta unbounded).
    '''
    # Create 3D grid, theta is periodic
    grid_min = np.array([-4.0, -4.0, -math.pi])
    grid_max = np.array([4.0, 4.0, math.pi])
    dims = grid_min.shape[0]
    N = np.array([num, num, num])
    g = Grid(grid_min, grid_max, dims, N, periodicDims=[2])

    # Initialize value function (negative inside the obstacle)
    data_1 = ShapeRectangle(g, np.array([obstacle_min[0], -np.inf, -np.inf]), np.array([obstacle_max[0], np.inf, np.inf]))
    data_2 = ShapeRectangle(g, np.array([-np.inf, obstacle_min[1], -np.inf]), np.array([np.inf, obstacle_max[1], np.inf]))
    data = np.minimum(data_1, data_2)

    ## Look-back length and time step of computation
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics, control tries to avoid the obstacle
    sys = DubinsCar(wMax=1, speed=1, uMode="max", dMode="min")

    # Same computation method as decomposition.py
    compMethods = {"TargetSetMode": "None"}

    start = time.time()

    result = HJSolver(sys, g, data, tau, compMethods,
                      saveAllTimeSteps=saveAllTimeStep, accuracy=accuracy)

    if saveAllTimeStep:
        # HJSolver stores the initial value at index -1, flip so that index 0 is t = 0
        result = np.flip(result, axis=-1)

    execution_time = time.time() - start

    print("Total kernel time direct computation: ", execution_time)
    print("Finished updating the value function")

    np.save('results/result_direct_3d.npy', result)

    # Visualization of 3D value function
    po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2], slicesCut=[], colorscale="Bluered",
                    save_fig=True, filename="plots/direct_result", interactive_html=True)
    visualize_plots(result, g, po)

    return g, result


if __name__ == "__main__":
    if os.path.exists("plots") == False:
        os.mkdir("plots")
    if os.path.exists("results") == False:
        os.mkdir("results")
    direct_comp(41, saveAllTimeStep=True, lookback_length=1.0)
