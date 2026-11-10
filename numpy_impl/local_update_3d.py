# Local update of the 3D Dubins car decomposition result, pure numpy version.
# Corrects the decomposition result near the leaking corners so it matches the direct computation.
import argparse

import numpy as np
import math
import os
import sys as py_sys

from odp.Grid import Grid

from dynamics.dubins_3d import dubins_car
from common.combine import check_combine
from common.leaking import get_leaking_set, dilate
from common.params import parse_args_with_params
from common.stats import printCorrectnessStatistics

# The function of computing first order spatial derivative
from common.spatial_deriv import spa_deriv

import time

np.set_printoptions(threshold=py_sys.maxsize)
np.set_printoptions(precision=5)


def recomputeValueChange(data, x, y, z, grid, sys, t, t_step):
    # recompute value change at index data[x, y, z] using the value function data

    # Get the left and right spatial derivatives (1st order)
    dV_dx_L, dV_dx_R = spa_deriv((x, y, z), data, grid, 0, boundary="extrapolate")
    dV_dy_L, dV_dy_R = spa_deriv((x, y, z), data, grid, 1, boundary="extrapolate")
    dV_dT_L, dV_dT_R = spa_deriv((x, y, z), data, grid, 2, boundary="extrapolate")

    # Get the average gradient
    dV_dx = (dV_dx_L + dV_dx_R) / 2
    dV_dy = (dV_dy_L + dV_dy_R) / 2
    dV_dT = (dV_dT_L + dV_dT_R) / 2

    # Get the dynamical rates of change
    state = [grid.grid_points[0][x], grid.grid_points[1][y], grid.grid_points[2][z]]
    uOpt = sys.opt_ctrl_numpy(t, state, [dV_dx, dV_dy, dV_dT])
    dx_dt, dy_dt, dT_dt = sys.dynamics_numpy(t, state, uOpt, 0)

    # Lax-Friedrichs dissipation, alpha is the max rate of change over the control set,
    # same as optimized_dp
    rates_1 = sys.dynamics_numpy(t, state, (sys.wMax,), 0)
    rates_2 = sys.dynamics_numpy(t, state, (-sys.wMax,), 0)
    alpha_x, alpha_y, alpha_T = np.maximum(np.abs(rates_1), np.abs(rates_2))
    diss = 0.5 * ((dV_dx_R - dV_dx_L) * alpha_x + (dV_dy_R - dV_dy_L) * alpha_y + (dV_dT_R - dV_dT_L) * alpha_T)

    # Updating the value function
    data_change = dx_dt * dV_dx + dy_dt * dV_dy + dT_dt * dV_dT + diss
    return data_change * t_step


def local_update_from_file(decomp_file, leaking, grid, sys, t_step, threshold=1e-6):
    '''
        Local updating based on a saved decomposition result
            decomp_file: npy file of the decomposition result, shape (x, y, theta, t), index 0 of the last axis is t = 0
                (index 0 is the initial value function)
            leaking: boolean mask of the initial leaking set, shape (x, y, theta), see get_leaking_set
            threshold: a point counts as corrected if it differs from the decomposition by more than this

        The update region grows over time: at each time step, the points updated are the leaking
        set plus every point (and its neighbours) whose value at the previous time step differs
        from the decomposition result. Every other point keeps the decomposition value.
    '''
    # Step 1: load the decomposition result, the initial value function is the decomposition at t = 0
    result_decomp = np.load(decomp_file)
    data_init = result_decomp[..., 0]

    # Step 2: local updating loop over time steps (pure numpy version)
    # data holds the corrected value function at the previous time step, starting from t = 0
    result_local = result_decomp.copy()
    data = data_init.astype(np.float64)
    total_updated = 0

    start = time.time()

    for i in range(1, result_decomp.shape[-1]):
        t = np.array([(i-1)*t_step, i*t_step])

        # Step 2a: find the points corrected at the previous time step (different from the decomposition)
        corrected = abs(data - result_decomp[..., i-1]) > threshold

        # Step 2b: the update region is the leaking set plus the corrected points and their
        # neighbours, since the first order scheme at a point reads its neighbours' previous values
        region = dilate(corrected, grid) | leaking

        # Step 2c: start from the decomposition result, then recompute one Euler step inside the region
        new_data = result_decomp[..., i].astype(np.float64)
        for x, y, z in np.argwhere(region):
            new_data[x, y, z] = data[x, y, z] + recomputeValueChange(data, x, y, z, grid, sys, t, t_step)
        data = new_data

        # Step 2d: save the corrected value function of this time step
        result_local[..., i] = data
        total_updated += int(region.sum())

    execution_time = time.time() - start
    print(f"Total correction time: {execution_time:.4f} seconds")
    print(f"Total point updates: {total_updated} / {data_init.size * (result_decomp.shape[-1] - 1)}")

    return result_local


def main():
    # Step 1: parse and check the inputs (from the command line and/or the --params file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="JSON parameter file (see numpy_impl/params/), its keys are the argument names below "
                        "without '--'. Arguments given on the command line override the file.", default=None)
    parser.add_argument("--decomp_file", help="npy file of the decomposition result, shape (x, y, theta, t). "
                        "Required, on the command line or in --params.", default=None)
    parser.add_argument("--true_file", help="Optional npy file of the direct computation result, same shape as --decomp_file. "
                        "If not given, the local update runs without the comparison.", default=None)
    parser.add_argument("--init_1", help="npy file of the subsystem 1 initial value V1, shape (x, y, theta), "
                        "saved by compute/decomposition_3d.py.", default="results/init_1_3d.npy")
    parser.add_argument("--init_2", help="npy file of the subsystem 2 initial value V2, shape (x, y, theta), "
                        "saved by compute/decomposition_3d.py.", default="results/init_2_3d.npy")
    parser.add_argument("--combine", choices=["union", "intersection"], default="union",
                        help="How the subsystems are combined: 'union' = min(V1, V2), 'intersection' = max(V1, V2). "
                        "Only union is implemented for 3d.")
    parser.add_argument("--leaking_mode", choices=["smallest", "delta"], default="smallest",
                        help="Initial leaking set: 'smallest' = switching surface V1 = V2 plus its neighbours (default), "
                        "'delta' = |V1 - V2| < delta.")
    parser.add_argument("--t_step", help="Time step used to compute both files. Keep it below the CFL bound of HJSolver "
                        "(about 0.048 for a 41^3 grid) so that HJSolver takes exactly one Euler step per time step.",
                        default=0.04, type=float)
    args = parse_args_with_params(parser)

    if args.decomp_file is None:
        parser.error("--decomp_file is required (on the command line or in --params)")
    try:
        check_combine(args.combine, ("union",), "3d")
    except NotImplementedError as e:
        parser.error(str(e))
    for path in (args.decomp_file, args.true_file, args.init_1, args.init_2):
        if path is not None and not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    decomp_shape = np.load(args.decomp_file, mmap_mode='r').shape
    if args.true_file is not None:
        true_shape = np.load(args.true_file, mmap_mode='r').shape
        if decomp_shape != true_shape:
            parser.error(f"shape mismatch: {args.decomp_file} {decomp_shape} vs {args.true_file} {true_shape}, "
                         "generate both files with the same t_step and accuracy (see README)")
    data_init_upper = np.load(args.init_1)
    data_init_lower = np.load(args.init_2)
    for path, data in ((args.init_1, data_init_upper), (args.init_2, data_init_lower)):
        if data.shape != decomp_shape[:-1]:
            parser.error(f"shape mismatch: {path} {data.shape} vs {args.decomp_file} {decomp_shape[:-1]} (without t), "
                         "generate them with the same compute/decomposition_3d.py run")

    # Step 2: set the dynamics, same as the direct computation, control tries to avoid the obstacle
    sys = dubins_car(v=1, wMax=1, dMax=0.0, uMode='max', dMode='min')

    # Step 3: build the 3D grid with the grid size read from the decomposition file, theta is periodic
    num = np.load(args.decomp_file, mmap_mode='r').shape[0]
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([num, num, num]), [2])

    # Step 4: initial leaking set from the subsystem initial values (--init_1, --init_2)
    leaking = get_leaking_set(data_init_upper, data_init_lower, np.load(args.decomp_file), grid, args.leaking_mode)

    # Step 5: run the local update and save the corrected result
    result_local = local_update_from_file(args.decomp_file, leaking, grid, sys, args.t_step)
    np.save(f'results/result_local_3d_{args.combine}.npy', result_local)

    # Step 6: compare the decomposition and local update results with the direct computation (if given)
    if args.true_file is not None:
        printCorrectnessStatistics(np.load(args.true_file), np.load(args.decomp_file), result_local)
    else:
        print("No --true_file given, skipping the comparison with the direct computation")


if __name__ == "__main__":
    if os.path.exists("results") == False:
        os.mkdir("results")
    main()
