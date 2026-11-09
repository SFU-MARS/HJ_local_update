# Local update of the 3D Dubins car decomposition result, pure numpy version.
# Corrects the decomposition result near the leaking corners so it matches the direct computation.
import argparse

import numpy as np
import math
import os
import sys as py_sys

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle

from system import dubins_car

# The function of computing first order spatial derivative
from update_V_numpy import spa_deriv3X1, spa_deriv3X2, spa_deriv3X3

import time

np.set_printoptions(threshold=py_sys.maxsize)
np.set_printoptions(precision=5)


def recomputeValueChange(data, x, y, z, grid, sys, t, t_step):
    # recompute value change at index data[x, y, z] using the value function data

    # Get the left and right spatial derivatives (1st order)
    dV_dx_L, dV_dx_R = spa_deriv3X1(x, y, z, data, grid)
    dV_dy_L, dV_dy_R = spa_deriv3X2(x, y, z, data, grid)
    dV_dT_L, dV_dT_R = spa_deriv3X3(x, y, z, data, grid)

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


def dilate(mask, grid):
    '''
        Add the neighbours of every True point along each dimension (the stencil of the
        first order scheme). Periodic dimensions wrap around.
    '''
    out = mask.copy()
    for dim in range(mask.ndim):
        if dim in grid.pDim:
            out |= np.roll(mask, 1, axis=dim) | np.roll(mask, -1, axis=dim)
        else:
            lo = [slice(None)] * mask.ndim
            hi = [slice(None)] * mask.ndim
            lo[dim] = slice(None, -1)
            hi[dim] = slice(1, None)
            out[tuple(hi)] |= mask[tuple(lo)]
            out[tuple(lo)] |= mask[tuple(hi)]
    return out


def get_leaking_set(data_init_upper, data_init_lower, result_decomp):
    '''
        Initial leaking set: points where the two subsystem initial values are close,
        |V1 - V2| < delta, so the decomposition min(V1, V2) can be wrong there (new_theory_L in 6d_union).
        delta = max|initial value - final decomposition result|.
    '''
    delta = np.max(abs(result_decomp[..., 0] - result_decomp[..., -1]))
    print('The delta value is', delta)
    leaking = abs(data_init_upper - data_init_lower) < delta
    print(f"The number of leaking corner points: {leaking.sum()} / {leaking.size}")
    return leaking


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


def compareArrays(array1, array2, number_of_precision_points=8):
    diff_array = abs(array1 - array2)
    for precision in range(1, number_of_precision_points):
        precision_value = 1 / (10**precision)
        number_of_entires_with_error = len(np.argwhere(diff_array > precision_value))
        print(f"{precision_value}: {number_of_entires_with_error}")


def printCorrectnessStatistics(result_true, result_decomp, result_local):
    print("Processing stats")
    print("================================================")

    '''
    Comparision with direct computation at the final time step
    '''
    true_final = result_true[..., -1]
    decomp_final = result_decomp[..., -1]
    local_final = result_local[..., -1]

    total_num_of_points = true_final.size
    print(f"The total number of points: {total_num_of_points}")
    print(f"Decomposition method: Number of points with different values (1e-6): {np.argwhere(abs(decomp_final - true_final) > 1e-6).shape}")
    print(f"After local update, Number of points with different values (1e-6): {np.argwhere(abs(local_final - true_final) > 1e-6).shape}")

    print(f"The average absolute error between direct computation and decomposition: {np.sum(abs(decomp_final - true_final)) / total_num_of_points:0.10f}")
    print(f"The average absolute error between direct computation and local updates: {np.sum(abs(local_final - true_final)) / total_num_of_points:0.10f}")
    print(f"The maximum absolute error between direct computation and decomposition: {np.max(abs(decomp_final - true_final)):0.10f}")
    print(f"The maximum absolute error between direct computation and local updates: {np.max(abs(local_final - true_final)):0.10f}")

    # Zero level set comparison
    print(f"Points with different sign (set mismatch), decomposition: {np.sum((decomp_final < 0) != (true_final < 0))}")
    print(f"Points with different sign (set mismatch), local update: {np.sum((local_final < 0) != (true_final < 0))}")

    print("Number of values with error b/w direct computation and decomposition:")
    compareArrays(true_final, decomp_final)
    print("Number of values with error b/w direct computation and local update:")
    compareArrays(true_final, local_final)


def main():
    # Step 1: parse and check the command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp_file", help="npy file of the decomposition result, shape (x, y, theta, t).", default=None)
    parser.add_argument("--true_file", help="npy file of the direct computation result, same shape as --decomp_file.", default=None)
    parser.add_argument("--t_step", help="Time step used to compute both files. Keep it below the CFL bound of HJSolver "
                        "(about 0.048 for a 41^3 grid) so that HJSolver takes exactly one Euler step per time step.",
                        default=0.04, type=float)
    args = parser.parse_args()

    missing = [name for name, value in (("--decomp_file", args.decomp_file), ("--true_file", args.true_file)) if value is None]
    if missing:
        parser.error(f"need file inputs {' and '.join(missing)} (e.g. --decomp_file results/result_decomp_3d.npy "
                     "--true_file results/result_direct_3d.npy)")
    for path in (args.decomp_file, args.true_file):
        if not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    decomp_shape = np.load(args.decomp_file, mmap_mode='r').shape
    true_shape = np.load(args.true_file, mmap_mode='r').shape
    if decomp_shape != true_shape:
        parser.error(f"shape mismatch: {args.decomp_file} {decomp_shape} vs {args.true_file} {true_shape}, "
                     "generate both files with the same t_step and accuracy (see README)")

    # Step 2: set the dynamics, same as the direct computation, control tries to avoid the obstacle
    sys = dubins_car(v=1, wMax=1, dMax=0.0, uMode='max', dMode='min')

    # Step 3: build the 3D grid with the grid size read from the decomposition file, theta is periodic
    num = np.load(args.decomp_file, mmap_mode='r').shape[0]
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([num, num, num]), [2])

    # Step 4: initial leaking set from the subsystem initial values, same obstacle as decomposition.py
    data_init_upper = ShapeRectangle(grid, np.array([-0.5, -np.inf, -np.inf]), np.array([0.5, np.inf, np.inf]))
    data_init_lower = ShapeRectangle(grid, np.array([-np.inf, -0.5, -np.inf]), np.array([np.inf, 0.5, np.inf]))
    leaking = get_leaking_set(data_init_upper, data_init_lower, np.load(args.decomp_file))

    # Step 5: run the local update and save the corrected result
    result_local = local_update_from_file(args.decomp_file, leaking, grid, sys, args.t_step)
    np.save('results/result_local_3d.npy', result_local)

    # Step 6: compare the decomposition and local update results with the direct computation
    printCorrectnessStatistics(np.load(args.true_file), np.load(args.decomp_file), result_local)


if __name__ == "__main__":
    if os.path.exists("results") == False:
        os.mkdir("results")
    main()
