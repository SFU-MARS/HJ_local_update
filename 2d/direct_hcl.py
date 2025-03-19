import imp
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from system import couple_u

# Plot Options
from odp.Plots import *
# Solver Core
from odp.solver import HJSolver

def direct_comp(num):

    # Create Grid
    grid_min = np.array([-4.0, -4.0])
    grid_max = np.array([4.0, 4.0])
    dims = grid_min.shape[0]
    N = np.array([num, num])
    g = Grid(grid_min, grid_max, dims, N)

    ## Initialize value function
    data = ShapeRectangle(g, [-1.0, -1.0], [1.0, 1.0])

    # Visualization
    po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0, 1],
                    colorscale="Bluered", save_fig=True, filename="direct_2Int_leaking", interactive_html=True)
    # plot_valuefunction(g, data, po)

    ## Look-back length and time step of computation
    lookback_length = 0.02
    t_step = 0.02
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics
    sys = couple_u(x=[0,0], uMax=1, dMax=0.0, uMode='min', dMode='min')

    # Set computation task and compute HJ PDE
    compMethods = {"TargetSetMode": "None"}
    result = HJSolver(sys, g, data, tau, compMethods, po, saveAllTimeSteps=True, untilConvergent=False)
    
    ## Compare with ground truth
    ## The time span is 1 second and the maximum speed is 1
    ## With the target set as a square of side length 2
    ## One of the corner that should be insid of the BRS is (1.707, 1.707)
    
    # get the index
    # step = 8/(num-1)
    # step_n = np.int((4-1.707)/step)
    # n = 50-step_n
    test_value = g.get_value(result[:,:,0], [1.707, 1.707])

    print("The test value from direct computation is", test_value)  
    
    return g, result




