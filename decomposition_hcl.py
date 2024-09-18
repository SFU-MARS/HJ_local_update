import imp
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from subsystem import subsys

# Plot Options
from odp.Plots import *
# Solver Core
from odp.solver import HJSolver


# def decomposition(num):
num = 51

# Create Grid
grid_min = np.array([-4.0])
grid_max = np.array([4.0])
dims = grid_min.shape[0]
N = np.array([num])
g = Grid(grid_min, grid_max, dims, N)

# Initialize value function
data = ShapeRectangle(g, [-1.0], [1.0])

# Visualization
po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0],
                colorscale="Bluered", save_fig=True, filename="direct_2Int_leaking", interactive_html=True)

## Look-back length and time step of computation
lookback_length = 0.02
t_step = 0.02
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# Set system dynamics
sys = subsys(x=[0], uMax=1, dMax=0.0, uMode='min', dMode='min')

# Set computation task and compute HJ PDE
compMethods = {"TargetSetMode": "None"}
result_sub = HJSolver(sys, g, data, tau, compMethods, po, saveAllTimeSteps=True, untilConvergent=False)

# combined result and grid
grid_min = np.array([-4.0, -4.0])
grid_max = np.array([4.0, 4.0])
dims = grid_min.shape[0]
N = np.array([num, num])
grid_full = Grid(grid_min, grid_max, dims, N)

result_upper_flip = np.flip(result_sub, axis=0) 
result_upper_expand = np.tile(result_upper_flip, (num, 1, 1))
print(result_upper_expand.shape)
result_upper = np.transpose(result_upper_expand, (1,0,2))

result_lower_flip = np.flip(result_sub, axis=0)
result_lower_expand = np.tile(result_lower_flip, (num, 1, 1))
print(result_lower_expand.shape)
result_lower = np.transpose(result_lower_expand, (0,1,2))

result_full = np.maximum(result_upper, result_lower)

## Compare with truthground 
po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0, 1],
                    colorscale="Bluered", save_fig=False, filename="direct_2Int_leaking", interactive_html=True)
plot_isosurface(grid_full, result_full, po)

    # test_value = grid_full.get_value(result_full[:,:,0], [1.707, 1.707])
    # print("The test value from system decomposition is", test_value)  
    
    # return result_full
