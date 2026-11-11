# Settings of the 6d quadrotor case shared by compute/direct_6d.py, compute/decomposition_6d.py and
# numpy_impl/local_update_6d.py: the grid, the initial value function and the arguments of the compute scripts.
#   state (x, y, vx, vy, theta, w), subsystem 1 (x, vx, theta, w), subsystem 2 (y, vy, theta, w)
import numpy as np

from odp.Grid import Grid
from odp.Shapes import Lower_Half_Space

GRID_MIN_6D = np.array([-1.0, -1.0, -2.0, -2.0, -2.0, -2.0])
GRID_MAX_6D = np.array([4.0, 4.0, 2.0, 2.0, 2.0, 2.0])
# Subsystem grids: (x, vx, theta, w) and (y, vy, theta, w), the matching dimensions of the 6D grid
SUB_DIMS_6D = ((0, 2, 4, 5), (1, 3, 4, 5))


def grid_6d(num):
    '''
        6D grid with num points per dimension, no periodic dimension.
    '''
    return Grid(GRID_MIN_6D, GRID_MAX_6D, 6, np.array([num] * 6))


def grid_sub_6d(num):
    '''
        4D grid of either subsystem (both have the same bounds).
    '''
    dims = SUB_DIMS_6D[0]
    return Grid(GRID_MIN_6D[list(dims)], GRID_MAX_6D[list(dims)], 4, np.array([num] * 4))


def init_values_6d(num):
    '''
        Initial values of the union case: V1 = x (negative for x <= 0), V2 = y (negative for y <= 0),
        on the 6D grid. The full initial value is min(V1, V2).
    '''
    g = grid_6d(num)
    return Lower_Half_Space(g, 0, 0), Lower_Half_Space(g, 1, 0)


def add_compute_args_6d(parser):
    '''
        Arguments of compute/direct_6d.py and compute/decomposition_6d.py. Use the same values for both,
        the local update reads the grid size and number of time steps from the saved files.
    '''
    from dynamics import add_dynamics_args
    parser.add_argument("--grid_size", type=int, default=21, help="Grid points per dimension, e.g. 21 for 21^6.")
    parser.add_argument("--combine", choices=["union", "intersection"], default="union",
                        help="union = min(V1, V2) with uMode max (avoid). Only union is implemented for 6d.")
    parser.add_argument("--t_step", type=float, default=0.02, help="Time step.")
    parser.add_argument("--lookback_length", type=float, default=0.1,
                        help="Look-back length, lookback_length / t_step time steps.")
    parser.add_argument("--results_dir", default="results", help="Directory of the saved results.")
    add_dynamics_args(parser, default="quadrotor_6d")


def result_files_6d(combine, results_dir="results"):
    '''
        Names of the saved 6d results:
            direct, decomp: shape (x, y, vx, vy, theta, w, t), index 0 of the last axis is t = 0
            init_1, init_2: subsystem initial values V1, V2 on the 6D grid, shape (x, y, vx, vy, theta, w)
    '''
    return {
        "direct": f"{results_dir}/result_direct_6d_{combine}.npy",
        "decomp": f"{results_dir}/result_decomp_6d_{combine}.npy",
        "init_1": f"{results_dir}/init_1_6d.npy",
        "init_2": f"{results_dir}/init_2_6d.npy",
    }
