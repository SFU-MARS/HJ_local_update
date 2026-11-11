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


def add_local_update_args_6d(parser):
    '''
        Arguments shared by numpy_impl/local_update_6d.py and hcl_impl/local_update_6d.py.
    '''
    from dynamics import add_dynamics_args
    parser.add_argument("--params", help="JSON parameter file (see numpy_impl/params/, hcl_impl/params/), its keys are "
                        "the argument names below without '--'. Arguments given on the command line override the file.",
                        default=None)
    parser.add_argument("--decomp_file", help="npy file of the decomposition result, shape (x, y, vx, vy, theta, w, t). "
                        "Required, on the command line or in --params.", default=None)
    parser.add_argument("--true_file", help="Optional npy file of the direct computation result, same shape as --decomp_file. "
                        "If not given, the local update runs without the comparison.", default=None)
    parser.add_argument("--init_1", help="npy file of the subsystem 1 initial value V1 on the 6D grid, "
                        "saved by compute/decomposition_6d.py.", default="results/init_1_6d.npy")
    parser.add_argument("--init_2", help="npy file of the subsystem 2 initial value V2 on the 6D grid, "
                        "saved by compute/decomposition_6d.py.", default="results/init_2_6d.npy")
    parser.add_argument("--combine", choices=["union", "intersection"], default="union",
                        help="How the subsystems are combined: 'union' = min(V1, V2), avoid (uMode max). "
                        "Only union is implemented for 6d.")
    parser.add_argument("--leaking_mode", choices=["smallest", "delta"], default="smallest",
                        help="Initial leaking set: 'smallest' = switching surface V1 = V2 plus its neighbours (default), "
                        "'delta' = |V1 - V2| < delta.")
    parser.add_argument("--t_step", help="Time step used to compute the files.", default=0.02, type=float)
    add_dynamics_args(parser, default="quadrotor_6d")


def setup_local_update_6d(parser, args):
    '''
        Check the arguments of add_local_update_args_6d and the files, build the dynamics, the 6D grid
        (grid size read from the decomposition file) and the initial leaking set.
        Returns (sys, grid, leaking).
    '''
    import os
    from common.combine import check_combine
    from common.leaking import get_leaking_set
    from dynamics import make_dynamics

    if args.decomp_file is None:
        parser.error("--decomp_file is required (on the command line or in --params)")
    try:
        check_combine(args.combine, ("union",), "6d")
    except NotImplementedError as e:
        parser.error(str(e))
    for path in (args.decomp_file, args.true_file, args.init_1, args.init_2):
        if path is not None and not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    decomp_shape = np.load(args.decomp_file, mmap_mode='r').shape
    if len(decomp_shape) != 7:
        parser.error(f"{args.decomp_file} must have shape (x, y, vx, vy, theta, w, t), got {decomp_shape}")
    if args.true_file is not None:
        true_shape = np.load(args.true_file, mmap_mode='r').shape
        if decomp_shape != true_shape:
            parser.error(f"shape mismatch: {args.decomp_file} {decomp_shape} vs {args.true_file} {true_shape}, "
                         "generate both with the same grid_size, t_step and lookback_length")
    data_init_upper = np.load(args.init_1)
    data_init_lower = np.load(args.init_2)
    for path, data in ((args.init_1, data_init_upper), (args.init_2, data_init_lower)):
        if data.shape != decomp_shape[:-1]:
            parser.error(f"shape mismatch: {path} {data.shape} vs {args.decomp_file} {decomp_shape[:-1]} (without t), "
                         "generate them with the same compute/decomposition_6d.py run")

    # Dynamics (--dynamics, --dynamics_params), use the same as the files
    try:
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=6)
    except (ValueError, TypeError) as e:
        parser.error(f"--dynamics {args.dynamics} --dynamics_params {args.dynamics_params}: {e}")
    print(f"Dynamics: {args.dynamics} {vars(sys)}")

    # 6D grid with the grid size read from the decomposition file
    num = decomp_shape[0]
    grid = grid_6d(num)
    print(f"Grid size: {num}^6, time steps: {decomp_shape[-1] - 1}")

    # Initial leaking set from the subsystem initial values (--init_1, --init_2)
    leaking = get_leaking_set(data_init_upper, data_init_lower, np.load(args.decomp_file, mmap_mode='r'), grid,
                              args.leaking_mode)
    return sys, grid, leaking
