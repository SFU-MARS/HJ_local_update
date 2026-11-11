# Settings of the 2d coupled control case shared by compute/direct_2d.py, compute/decomposition_2d.py and the
# 2d local updates: the grid, the initial value function, the file names and the defaults.
#   state (x, y), subsystem 1 (x), subsystem 2 (y), both with the same 1D dynamics
import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle

GRID_MIN_2D = np.array([-4.0, -4.0])
GRID_MAX_2D = np.array([4.0, 4.0])


def grid_2d(num):
    '''
        2D grid with num points per dimension, no periodic dimension.
    '''
    return Grid(GRID_MIN_2D, GRID_MAX_2D, 2, np.array([num] * 2))


def grid_sub_2d(num):
    '''
        1D grid of either subsystem (both have the same bounds).
    '''
    return Grid(GRID_MIN_2D[:1], GRID_MAX_2D[:1], 1, np.array([num]))


def init_value_sub_2d(grid):
    '''
        Subsystem initial value: the interval |s| <= 1 (negative inside), on a grid whose first dimension is s.
    '''
    return ShapeRectangle(grid, [-1.0], [1.0])


def init_values_2d(num):
    '''
        Initial values on the 2D grid: V1 = |x| <= 1, V2 = |y| <= 1 (negative inside).
        The full initial value is min(V1, V2) (union, a cross) or max(V1, V2) (intersection, a square).
    '''
    g = grid_2d(num)
    return (ShapeRectangle(g, [-1.0, -np.inf], [1.0, np.inf]),
            ShapeRectangle(g, [-np.inf, -1.0], [np.inf, 1.0]))


def add_compute_args_2d(parser):
    '''
        Arguments of compute/direct_2d.py and compute/decomposition_2d.py. Use the same values for both,
        the local update reads the grid size and number of time steps from the saved files.
    '''
    from common.params import add_compute_args
    add_compute_args(parser, grid_size=101, combine="intersection", t_step=0.02, lookback_length=0.20, dynamics="coupled_2d")


def result_files_2d(combine, results_dir="results"):
    '''
        Names of the saved 2d results:
            direct: shape (x, y, t), time as the last axis (as HJSolver returns it), index 0 is t = 0
            decomp: shape (t, x, y), time as the first axis (as the local updates read it), index 0 is t = 0
            init_1, init_2: subsystem initial values V1, V2 on the 2D grid, shape (x, y)
    '''
    return {
        "direct": f"{results_dir}/result_direct_2d_{combine}.npy",
        "decomp": f"{results_dir}/result_decomp_2d_{combine}.npy",
        "init_1": f"{results_dir}/init_1_2d.npy",
        "init_2": f"{results_dir}/init_2_2d.npy",
    }


# Local update settings (common/local_update_setup.py). With coupled control the decomposition also differs
# from the direct computation away from the leaking set (the Lax-Friedrichs dissipation of the subsystems is
# larger), so the region has to grow within each time step.
CASE_2D = {
    "dim": 2,
    "grid": grid_2d,
    "axes": "(x, y)",
    "dynamics": "coupled_2d",
    "combine": "intersection",
    "t_step": 0.02,
    "grow_within_step": True,
}
