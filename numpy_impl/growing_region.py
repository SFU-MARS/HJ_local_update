# Growing-region local update in numpy, for any number of dimensions (used by every numpy_impl/local_update_<dim>.py).
# Same update as hcl_impl/growing_region.py, in 32-bit floats like HJSolver (common/hj_update.py).
import time

import numpy as np

from common.hj_update import DTYPE, deriv_bounds_numpy, value_change_numpy
from common.leaking import dilate


def local_update_from_file(decomp_file, leaking, grid, sys, t_step, threshold=1e-6, grow_within_step=False):
    '''
        Local updating based on a saved decomposition result
            decomp_file: npy file of the decomposition result, grid axes then time, index 0 of the last axis is t = 0
            leaking: boolean mask of the initial leaking set, see get_leaking_set in common/leaking.py
            grid: odp Grid of the full system
            sys: dynamics with numpy methods (see dynamics/__init__.py)
            t_step: time step used to compute decomp_file
            threshold: a point counts as corrected if it differs from the decomposition by more than this
            grow_within_step: also grow the region within each time step, see below

        At each time step, the points updated (one HJSolver step from the corrected previous step) are the
        leaking set plus every point (and its neighbours) whose value at the previous time step differs
        from the decomposition result. Every other point keeps the decomposition value.
        With grow_within_step, every updated point that differs from the decomposition at this time step
        also adds its neighbours, repeated until no new point joins (needed when the decomposition is also
        wrong away from the leaking set, e.g. the 2d coupled control).
        Returns the corrected value function, same shape as the decomposition result.
    '''
    # Step 1: load the decomposition result, the initial value function is the decomposition at t = 0
    result_decomp = np.load(decomp_file).astype(DTYPE)
    data = result_decomp[..., 0].copy()
    state_values = [np.reshape(grid.vs[dim], grid.pts_each_dim[dim]).astype(DTYPE) for dim in range(grid.dims)]

    # Step 2: local updating loop over time steps
    result_local = result_decomp.copy()
    total_updated = 0
    total_sweeps = 0
    start = time.time()

    for i in range(1, result_decomp.shape[-1]):
        t = np.array([(i-1)*t_step, i*t_step])

        # Step 2a: update region = leaking set + points corrected at the previous time step and their
        # neighbours, since the first order scheme at a point reads its neighbours' previous values
        corrected = abs(data - result_decomp[..., i-1]) > threshold
        region = dilate(corrected, grid) | leaking

        # Step 2b: start from the decomposition result, recompute one HJSolver step inside the region
        # (with the global derivative bounds of the previous step if the dissipation needs them)
        bounds = deriv_bounds_numpy(data, grid) if sys.DISSIPATION_USES_DERIV_BOUNDS else (None, None)
        new_data = result_decomp[..., i].copy()
        done = np.zeros(region.shape, dtype=bool)
        while True:
            todo = region & ~done
            if not todo.any():
                break
            total_sweeps += 1
            for index in np.argwhere(todo):
                index = tuple(index)
                new_data[index] = data[index] + value_change_numpy(data, index, grid, sys, t, t_step, state_values, *bounds)
            done |= todo
            if not grow_within_step:
                break
            # Step 2c: grow within the step, the neighbours of updated points that differ from the decomposition
            region |= dilate(done & (abs(new_data - result_decomp[..., i]) > threshold), grid)
        data = new_data

        # Step 2d: save the corrected value function of this time step
        result_local[..., i] = data
        num_updated = int(done.sum())
        total_updated += num_updated
        print(f"Time step {i}: {num_updated} points updated, {time.time() - start:.1f} s")

    print(f"Total correction time: {time.time() - start:.4f} seconds")
    print(f"Total point updates: {total_updated} / {data.size * (result_decomp.shape[-1] - 1)}, sweeps: {total_sweeps}")
    return result_local
