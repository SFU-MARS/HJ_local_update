# Settings of the 4d Dubins car case used by the 4d local updates: the grid, the file names and the defaults.
#   state (x, y, v, theta), subsystem 1 (x, v, theta), subsystem 2 (y, v, theta); theta is periodic
# (compute/direct_4d.py and compute/decomposition_4d.py build the same grid.)
import math

import numpy as np

from odp.Grid import Grid

GRID_MIN_4D = np.array([-4.0, -4.0, 0.0, -math.pi])
GRID_MAX_4D = np.array([4.0, 4.0, 2.0, math.pi])


def grid_4d(num):
    '''
        4D grid with num points per dimension, theta (dimension 3) is periodic.
    '''
    return Grid(GRID_MIN_4D, GRID_MAX_4D, 4, np.array([num] * 4), [3])


def result_files_4d(combine, results_dir="results"):
    '''
        Names of the saved 4d results:
            direct: shape (x, y, v, theta, t), time as the last axis (as HJSolver returns it), index 0 is t = 0
            decomp: shape (t, x, y, v, theta), time as the first axis (as the local updates read it), index 0 is t = 0
            init_1, init_2: subsystem initial values V1, V2 on the 4D grid, shape (x, y, v, theta)
    '''
    return {
        "direct": f"{results_dir}/result_direct_4d_{combine}.npy",
        "decomp": f"{results_dir}/result_decomp_4d_{combine}.npy",
        "init_1": f"{results_dir}/init_1_4d.npy",
        "init_2": f"{results_dir}/init_2_4d.npy",
    }


# Local update settings (common/local_update_setup.py)
CASE_4D = {
    "dim": 4,
    "grid": grid_4d,
    "axes": "(x, y, v, theta)",
    "dynamics": "dubins_4d",
    "combine": "union",
    "t_step": 0.01,
    "grow_within_step": False,
}
