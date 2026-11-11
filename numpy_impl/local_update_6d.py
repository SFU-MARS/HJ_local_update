# Local update of the 6D quadrotor decomposition result, pure numpy version.
# Corrects the decomposition result near the leaking corners so it matches the direct computation
# (compute/direct_6d.py). Same growing-region update as numpy_impl/local_update_3d.py, with the
# numerical scheme of compute/direct_6d.py: average of the one-sided first order derivatives,
# no dissipation, forward Euler.
import argparse
import os
import time

import numpy as np

from compute.config_6d import grid_6d
from common.combine import check_combine
from common.leaking import dilate, get_leaking_set
from common.params import parse_args_with_params
from common.spatial_deriv import spa_deriv
from common.stats import printCorrectnessStatistics
from dynamics import add_dynamics_args, make_dynamics


def recomputeValueChange(data, index, grid, sys, t, t_step, state_values):
    '''
        Value change of one forward Euler step at index, same arithmetic as compute/direct_6d.py.
            index: tuple of grid indices
            state_values: list of the grid coordinates of each dimension
    '''
    dims = len(index)
    # Average of the left and right spatial derivatives (1st order)
    grad = []
    for dim in range(dims):
        dV_L, dV_R = spa_deriv(index, data, grid, dim, boundary="one_sided")
        grad.append((dV_L + dV_R)/2)

    # Optimal control and dynamical rates of change
    state = [state_values[dim][index[dim]] for dim in range(dims)]
    uOpt = sys.opt_ctrl_numpy(t, state, grad)
    rates = sys.dynamics_numpy(t, state, uOpt, 0)

    # Same order of additions as compute/direct_6d.py
    data_change = rates[0]*grad[0]
    for dim in range(1, dims):
        data_change = data_change + rates[dim]*grad[dim]
    return data_change*t_step


def local_update_from_file(decomp_file, leaking, grid, sys, t_step, threshold=1e-6):
    '''
        Local updating based on a saved decomposition result
            decomp_file: npy file of the decomposition result, shape (x, y, vx, vy, theta, w, t),
                index 0 of the last axis is t = 0
            leaking: boolean mask of the initial leaking set, see get_leaking_set in common/leaking.py
            threshold: a point counts as corrected if it differs from the decomposition by more than this

        The update region grows over time: at each time step, the points updated are the leaking
        set plus every point (and its neighbours) whose value at the previous time step differs
        from the decomposition result. Every other point keeps the decomposition value.
    '''
    # Step 1: load the decomposition result, the initial value function is the decomposition at t = 0
    result_decomp = np.load(decomp_file)
    data = result_decomp[..., 0].astype(np.float64)
    state_values = [np.reshape(grid.vs[dim], grid.pts_each_dim[dim]) for dim in range(grid.dims)]

    # Step 2: local updating loop over time steps (pure numpy version)
    result_local = result_decomp.copy()
    total_updated = 0
    start = time.time()

    for i in range(1, result_decomp.shape[-1]):
        t = np.array([(i-1)*t_step, i*t_step])

        # Step 2a: points corrected at the previous time step (different from the decomposition)
        corrected = abs(data - result_decomp[..., i-1]) > threshold

        # Step 2b: update region = leaking set + corrected points and their neighbours
        region = dilate(corrected, grid) | leaking

        # Step 2c: start from the decomposition result, recompute one Euler step inside the region
        new_data = result_decomp[..., i].astype(np.float64)
        for index in np.argwhere(region):
            index = tuple(index)
            new_data[index] = data[index] + recomputeValueChange(data, index, grid, sys, t, t_step, state_values)
        data = new_data

        # Step 2d: save the corrected value function of this time step
        result_local[..., i] = data
        num_updated = int(region.sum())
        total_updated += num_updated
        print(f"Time step {i}: {num_updated} points updated, {time.time() - start:.1f} s")

    print(f"Total correction time: {time.time() - start:.4f} seconds")
    print(f"Total point updates: {total_updated} / {data.size * (result_decomp.shape[-1] - 1)}")
    return result_local


def main():
    '''
        Read the decomposition (compute/decomposition_6d.py) and, if given, the direct computation
        (compute/direct_6d.py), run the local update, save it to results/result_local_6d_<combine>.npy
        and compare it with the direct computation.
    '''
    # Step 1: parse and check the inputs (from the command line and/or the --params file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="JSON parameter file (see numpy_impl/params/), its keys are the argument names below "
                        "without '--'. Arguments given on the command line override the file.", default=None)
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
    args = parse_args_with_params(parser)

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

    # Step 2: set the dynamics (--dynamics, --dynamics_params), use the same as the files
    try:
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=6)
    except (ValueError, TypeError) as e:
        parser.error(f"--dynamics {args.dynamics} --dynamics_params {args.dynamics_params}: {e}")
    print(f"Dynamics: {args.dynamics} {vars(sys)}")

    # Step 3: 6D grid with the grid size read from the decomposition file
    num = decomp_shape[0]
    grid = grid_6d(num)
    print(f"Grid size: {num}^6, time steps: {decomp_shape[-1] - 1}")

    # Step 4: initial leaking set from the subsystem initial values (--init_1, --init_2)
    leaking = get_leaking_set(data_init_upper, data_init_lower, np.load(args.decomp_file, mmap_mode='r'), grid,
                              args.leaking_mode)

    # Step 5: run the local update and save the corrected result
    result_local = local_update_from_file(args.decomp_file, leaking, grid, sys, args.t_step)
    os.makedirs("results", exist_ok=True)
    path = f"results/result_local_6d_{args.combine}.npy"
    np.save(path, result_local)
    print(f"Saved {path}")

    # Step 6: compare the decomposition and local update results with the direct computation (if given)
    if args.true_file is not None:
        printCorrectnessStatistics(np.load(args.true_file), np.load(args.decomp_file), result_local)
    else:
        print("No --true_file given, skipping the comparison with the direct computation")


if __name__ == "__main__":
    main()
