# Settings of the 3d Dubins car case used by the 3d local updates: the grid, the file names and the defaults.
#   state (x, y, theta), subsystem 1 (x, theta), subsystem 2 (y, theta); theta is periodic
# (compute/direct_3d.py and compute/decomposition_3d.py build the same grid.)
import math

import numpy as np

from odp.Grid import Grid

GRID_MIN_3D = np.array([-4.0, -4.0, -math.pi])
GRID_MAX_3D = np.array([4.0, 4.0, math.pi])


def grid_3d(num):
    '''
        3D grid with num points per dimension, theta (dimension 2) is periodic.
    '''
    return Grid(GRID_MIN_3D, GRID_MAX_3D, 3, np.array([num] * 3), [2])


def result_files_3d(combine, results_dir="results"):
    '''
        Names of the saved 3d results:
            direct: shape (x, y, theta, t), time as the last axis (as HJSolver returns it), index 0 is t = 0
            decomp: shape (t, x, y, theta), time as the first axis (as the local updates read it), index 0 is t = 0
            init_1, init_2: subsystem initial values V1, V2 on the 3D grid, shape (x, y, theta)
    '''
    return {
        "direct": f"{results_dir}/result_direct_3d_{combine}.npy",
        "decomp": f"{results_dir}/result_decomp_3d_{combine}.npy",
        "init_1": f"{results_dir}/init_1_3d.npy",
        "init_2": f"{results_dir}/init_2_3d.npy",
    }


# Local update settings (common/local_update_setup.py)
CASE_3D = {
    "dim": 3,
    "grid": grid_3d,
    "axes": "(x, y, theta)",
    "dynamics": "dubins_3d",
    "combine": "union",
    "t_step": 0.04,
    "grow_within_step": False,
}
