# Arguments and setup shared by every local update script (numpy_impl/ and hcl_impl/ local_update_<dim>.py).
# Each case gives its settings as a dict (CASE_2D, CASE_3D, CASE_6D in compute/config_<dim>.py):
#   dim, grid (function of the grid size), axes (for the help text), and the defaults of dynamics, combine,
#   t_step and grow_within_step.
import os

import numpy as np

from common.combine import check_combine
from common.leaking import get_leaking_set
from common.params import parse_args_with_params


def bool_arg(value):
    '''
        argparse type for true/false, from a --params file (JSON true/false) or the command line.
    '''
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"expected true or false, got {value!r}")


def add_local_update_args(parser, case, hcl=False):
    '''
        Arguments of a local update script, the same for every case, only the defaults differ.
            case: CASE_<dim> from compute/config_<dim>.py
            hcl: True for the HeteroCL version (adds --cache_dir)
    '''
    from dynamics import add_dynamics_args
    name = f"{case['dim']}d"
    folder = "hcl_impl" if hcl else "numpy_impl"
    parser.add_argument("--params", default=None,
                        help=f"JSON parameter file (see {folder}/params/), its keys are the argument names below "
                        "without '--'. Arguments given on the command line override the file.")
    parser.add_argument("--decomp_file", default=None,
                        help=f"npy file of the decomposition result, shape {case['axes'][:-1]}, t). "
                        "Required, on the command line or in --params.")
    parser.add_argument("--true_file", default=None,
                        help="Optional npy file of the direct computation result, same shape as --decomp_file. "
                        "If not given (or null), the local update runs without the comparison.")
    parser.add_argument("--init_1", default=f"results/init_1_{name}.npy",
                        help=f"npy file of the subsystem 1 initial value V1 on the full grid {case['axes']}, "
                        f"saved by compute/decomposition_{name}.py.")
    parser.add_argument("--init_2", default=f"results/init_2_{name}.npy",
                        help="npy file of the subsystem 2 initial value V2, same as --init_1.")
    parser.add_argument("--combine", choices=["union", "intersection"], default=case["combine"],
                        help="How the subsystems are combined: 'union' = min(V1, V2), avoid (uMode max), "
                        "'intersection' = max(V1, V2), reach (uMode min). Use the same as for the files.")
    parser.add_argument("--leaking_mode", choices=["smallest", "delta"], default="smallest",
                        help="Initial leaking set: 'smallest' = switching surface V1 = V2 plus its neighbours (default), "
                        "'delta' = |V1 - V2| < delta.")
    parser.add_argument("--grow_within_step", type=bool_arg, default=case["grow_within_step"],
                        help="true: within each time step, repeat the update until no new point changes "
                        f"(default {str(case['grow_within_step']).lower()} for {name}).")
    parser.add_argument("--t_step", type=float, default=case["t_step"], help="Time step used to compute the files.")
    if hcl:
        parser.add_argument("--cache_dir", default="kernels",
                            help="Directory of the compiled HeteroCL kernels. A kernel built before with the same "
                            "settings is loaded from here instead of being rebuilt.")
    add_dynamics_args(parser, default=case["dynamics"])


def setup_local_update(parser, case, hcl=False):
    '''
        Parse and check the arguments of add_local_update_args and the files, build the dynamics, the grid
        (grid size read from the decomposition file) and the initial leaking set.
        Returns (args, sys, grid, leaking).
    '''
    from dynamics import make_dynamics
    add_local_update_args(parser, case, hcl)
    args = parse_args_with_params(parser)
    dim = case["dim"]
    name = f"{dim}d"

    if args.decomp_file is None:
        parser.error("--decomp_file is required (on the command line or in --params)")
    try:
        check_combine(args.combine, ("union", "intersection"), name)
    except NotImplementedError as e:
        parser.error(str(e))
    for path in (args.decomp_file, args.true_file, args.init_1, args.init_2):
        if path is not None and not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    decomp = np.load(args.decomp_file, mmap_mode='r')
    if decomp.ndim != dim + 1:
        parser.error(f"{args.decomp_file} must have shape {case['axes'][:-1]}, t), got {decomp.shape}")
    if args.true_file is not None:
        true_shape = np.load(args.true_file, mmap_mode='r').shape
        if decomp.shape != true_shape:
            parser.error(f"shape mismatch: {args.decomp_file} {decomp.shape} vs {args.true_file} {true_shape}, "
                         "generate both with the same grid_size, t_step and lookback_length")
    data_init_upper = np.load(args.init_1)
    data_init_lower = np.load(args.init_2)
    for path, data in ((args.init_1, data_init_upper), (args.init_2, data_init_lower)):
        if data.shape != decomp.shape[:-1]:
            parser.error(f"shape mismatch: {path} {data.shape} vs {args.decomp_file} {decomp.shape[:-1]} (without t), "
                         f"generate them with the same compute/decomposition_{name}.py run")

    # Dynamics (--dynamics, --dynamics_params), use the same as the files
    try:
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=dim)
    except (ValueError, TypeError) as e:
        parser.error(f"--dynamics {args.dynamics} --dynamics_params {args.dynamics_params}: {e}")
    print(f"Dynamics: {args.dynamics} {vars(sys)}")

    # Grid with the grid size read from the decomposition file
    num = decomp.shape[0]
    grid = case["grid"](num)
    print(f"Grid size: {num}^{dim}, time steps: {decomp.shape[-1] - 1}, combine: {args.combine}, "
          f"grow_within_step: {args.grow_within_step}")

    # Initial leaking set from the subsystem initial values (--init_1, --init_2)
    leaking = get_leaking_set(data_init_upper, data_init_lower, decomp, grid, args.leaking_mode)
    return args, sys, grid, leaking


def save_and_compare(args, result_local, name, hcl=False):
    '''
        Save the local update result to results/result_local[_hcl]_<name>_<combine>.npy and, if --true_file
        is given, compare the decomposition and the local update with the direct computation.
    '''
    from common.stats import printCorrectnessStatistics
    os.makedirs("results", exist_ok=True)
    path = f"results/result_local{'_hcl' if hcl else ''}_{name}_{args.combine}.npy"
    np.save(path, result_local)
    print(f"Saved {path}")
    if args.true_file is not None:
        printCorrectnessStatistics(np.load(args.true_file, mmap_mode='r'), np.load(args.decomp_file, mmap_mode='r'),
                                   result_local)
    else:
        print("No --true_file given, skipping the comparison with the direct computation")
