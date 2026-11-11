# Local update of the 6D quadrotor decomposition result, pure numpy version.
# Corrects the decomposition result near the leaking corners so it matches the direct computation
# (compute/direct_6d.py). Same growing-region update as numpy_impl/local_update_3d.py, with the
# numerical scheme of compute/direct_6d.py: average of the one-sided first order derivatives,
# no dissipation, forward Euler.
import argparse
import os
import time

import numpy as np

from compute.config_6d import add_local_update_args_6d, setup_local_update_6d
from common.combine import check_combine
from common.leaking import dilate
from common.params import parse_args_with_params
from common.spatial_deriv import spa_deriv
from common.stats import printCorrectnessStatistics


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
    # Steps 1-4: inputs, dynamics, grid and initial leaking set (see compute/config_6d.py)
    parser = argparse.ArgumentParser()
    add_local_update_args_6d(parser)
    args = parse_args_with_params(parser)
    sys, grid, leaking = setup_local_update_6d(parser, args)

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
