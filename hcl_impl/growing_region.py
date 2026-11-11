# Growing-region local update in a parallel HeteroCL kernel, for any number of dimensions (used by every
# hcl_impl/local_update_<dim>.py). Same update as numpy_impl/growing_region.py, in 32-bit floats like HJSolver
# (common/hj_update.py).
import inspect
import time

import heterocl as hcl
from heterocl import tvm
import numpy as np

from common import hj_update, spatial_deriv
from common.hj_update import DTYPE, DTYPE_HCL, abs_hcl, deriv_bounds_hcl, for_all_points, value_change_hcl
from hcl_impl.kernel_cache import load_or_build_kernel


def build_local_update_kernel(sys, grid, num_steps, threshold):
    '''
        Used by local_update_hcl_from_file. Returns the compiled HeteroCL kernel of one local update time step s:
            1. LocalUpdate (parallel over the first dimension), at every point:
                   region = leaking set + points (and their neighbours) where V_prev differs from the
                            decomposition result at step s-1 by more than threshold
                   V_new = V_prev + one HJSolver step   inside the region (common/hj_update.py)
                   V_new = decomposition result at step s   outside the region
            2. Grow (only with grow = 1, grow_within_step): every updated point that differs from the decomposition
               result at step s by more than threshold adds its neighbours to the region, and the points added are
               updated, repeated until no new point joins. Sequential, in the same kernel call.
        Every update reads only V_prev, so the order of the updates does not change the result: the same region
        and values as numpy_impl/growing_region.py.
            region, done: work arrays for grow = 1, set to s when a point is marked / updated at time step s,
                          so they never need to be reset
            counts: running number of points updated, per index of the first dimension (read only at the end)
            sweeps: running number of Grow passes that updated a point
        Same scheme as HJSolver: 1st order ENO derivatives, Lax-Friedrichs dissipation (with the global
        derivative bounds of V_prev, computed once per time step, if the dynamics need them), forward Euler,
        32-bit floats.
    '''
    shape = tuple(int(n) for n in grid.pts_each_dim)
    dims = len(shape)
    V_new = hcl.placeholder(shape, name="V_new", dtype=DTYPE_HCL)
    V_prev = hcl.placeholder(shape, name="V_prev", dtype=DTYPE_HCL)
    V_decomp = hcl.placeholder((num_steps,) + shape, name="V_decomp", dtype=DTYPE_HCL)
    leaking = hcl.placeholder(shape, name="leaking", dtype=hcl.Int())
    xs = [hcl.placeholder((shape[d],), name=f"x{d}", dtype=DTYPE_HCL) for d in range(dims)]
    step = hcl.placeholder((1,), name="step", dtype=hcl.Int())
    grow = hcl.placeholder((1,), name="grow", dtype=hcl.Int())
    t_step = hcl.placeholder((1,), name="t_step", dtype=DTYPE_HCL)
    region = hcl.placeholder(shape, name="region", dtype=hcl.Int())
    done = hcl.placeholder(shape, name="done", dtype=hcl.Int())
    # Points updated, counted per i0 (each i0 runs in one thread) so no integer mask is copied out
    counts = hcl.placeholder((shape[0],), name="counts", dtype=hcl.Int())
    sweeps = hcl.placeholder((1,), name="sweeps", dtype=hcl.Int())

    def graph_create(V_new, V_prev, V_decomp, leaking, *rest):
        x = rest[:dims]
        step, grow, t_step, region, done, counts, sweeps = rest[dims:]
        s = step[0]
        # Threshold as a constant of the kernel's float type
        thr = tvm.const(threshold, "float32")

        def neighbours(idx, action):
            # Call action(neighbour index) for each neighbour along each dimension (same stencil as dilate in
            # common/leaking.py): periodic dimensions wrap around, other dimensions stop at the grid edge
            for dim in range(dims):
                n = shape[dim]
                for offset in (-1, 1):
                    nb = hcl.scalar(0, "nb", dtype=hcl.Int())
                    nb[0] = idx[dim] + offset
                    inside = hcl.scalar(1, "inside", dtype=hcl.Int())
                    if dim in grid.pDim:
                        with hcl.if_(nb[0] < 0):
                            nb[0] = n - 1
                        with hcl.if_(nb[0] > n - 1):
                            nb[0] = 0
                    else:
                        with hcl.if_(nb[0] < 0):
                            inside[0] = 0
                        with hcl.if_(nb[0] > n - 1):
                            inside[0] = 0
                    with hcl.if_(inside[0] == 1):
                        nb_idx = list(idx)
                        nb_idx[dim] = nb[0]
                        action(tuple(nb_idx))

        def corrected(idx):
            # True if the point idx was corrected at the previous time step
            return abs_hcl(V_prev[tuple(idx)] - V_decomp[(s - 1,) + tuple(idx)]) > thr

        # Global derivative bounds of V_prev, for the dissipation like HJSolver (only if the dynamics need them)
        deriv_min = deriv_max = None
        if sys.DISSIPATION_USES_DERIV_BOUNDS:
            deriv_min, deriv_max = deriv_bounds_hcl(V_prev, grid)

        def update(idx):
            # One HJSolver step at idx (the dynamics do not depend on t); with grow = 1, record it and mark
            # the neighbours if the new value differs from the decomposition
            state = tuple(x[dim][idx[dim]] for dim in range(dims))
            V_new[idx] = V_prev[idx] + value_change_hcl(V_prev, list(idx), grid, sys, state, t_step[0],
                                                        deriv_min, deriv_max)
            with hcl.if_(grow[0] == 1):
                done[idx] = s

                def mark(nb_idx):
                    region[nb_idx] = s
                with hcl.if_(abs_hcl(V_new[idx] - V_decomp[(s,) + idx]) > thr):
                    neighbours(idx, mark)

        def first_sweep(idx, count):
            idx = tuple(idx)
            in_region = hcl.scalar(0, "in_region", dtype=hcl.Int())
            with hcl.if_(hcl.or_(leaking[idx] == 1, corrected(idx))):
                in_region[0] = 1

            def mark_if_corrected(nb_idx):
                with hcl.if_(corrected(nb_idx)):
                    in_region[0] = 1
            neighbours(idx, mark_if_corrected)

            with hcl.if_(in_region[0] == 1):
                count[0] = count[0] + 1
                update(idx)
            with hcl.else_():
                # Outside the update region keep the decomposition result
                V_new[idx] = V_decomp[(s,) + idx]

        with hcl.Stage("LocalUpdate"):
            # Loops i0, j0, j1, ... over the grid; each i0 (one thread) counts its updates in a local scalar
            with hcl.for_(0, shape[0], name="i0") as i0:
                count = hcl.scalar(0, "count", dtype=hcl.Int())
                for_all_points(shape[1:], lambda rest_idx: first_sweep([i0] + rest_idx, count), prefix="j")
                counts[i0] = counts[i0] + count[0]

        with hcl.Stage("Grow"):
            # Update the marked points that are not updated yet, until a pass updates none
            again = hcl.scalar(0, "again", dtype=hcl.Int())
            again[0] = grow[0]
            with hcl.while_(again[0] == 1):
                again[0] = 0

                def grow_point(idx):
                    idx = tuple(idx)
                    with hcl.if_(hcl.and_(region[idx] == s, done[idx] != s)):
                        counts[idx[0]] = counts[idx[0]] + 1
                        again[0] = 1
                        update(idx)
                for_all_points(shape, grow_point, prefix="g")
                with hcl.if_(again[0] == 1):
                    sweeps[0] = sweeps[0] + 1

    s = hcl.create_schedule([V_new, V_prev, V_decomp, leaking] + xs + [step, grow, t_step, region, done, counts, sweeps],
                            graph_create)
    # Thread parallelize the loop over the first dimension of the first sweep
    s_U = graph_create.LocalUpdate
    s[s_U].parallel(s_U.i0)
    return hcl.build(s)


def local_update_hcl_from_file(decomp_file, leaking, grid, sys, t_step, threshold=1e-6, cache_dir="kernels",
                               grow_within_step=False):
    '''
        Growing-region local update of a saved decomposition result, each time step runs in one HeteroCL kernel call.
            decomp_file: npy file of the decomposition result, grid axes then time, index 0 of the last axis is t = 0
            leaking: boolean mask of the initial leaking set, see get_leaking_set in common/leaking.py
            grid: odp Grid of the full system
            sys: dynamics with HeteroCL methods opt_ctrl, dynamics and dissipation_coeffs (see dynamics/__init__.py)
            t_step: time step used to compute decomp_file
            threshold: a point counts as corrected if it differs from the decomposition by more than this
            cache_dir: directory of the compiled kernels (see hcl_impl/kernel_cache.py)
            grow_within_step: repeat the sweep within each time step until no new point is updated
                              (see numpy_impl/growing_region.py)
        Returns the corrected value function, same shape as the decomposition result.
    '''
    # Step 1: load the decomposition result with time as the first axis, one time step at a time
    # (the file is several GB at 21^6, so avoid extra copies)
    decomp_file_data = np.load(decomp_file, mmap_mode='r')
    num_steps = decomp_file_data.shape[-1]
    shape = decomp_file_data.shape[:-1]
    dims = len(shape)
    result_decomp = np.empty((num_steps,) + shape, dtype=DTYPE)
    for i in range(num_steps):
        result_decomp[i] = decomp_file_data[..., i]
    del decomp_file_data

    # Step 2: build (or load the cached) HeteroCL kernel and copy all inputs to HeteroCL arrays once
    hcl.init(DTYPE_HCL)
    total_start = time.time()
    key = (tuple(grid.pts_each_dim), tuple(grid.min), tuple(grid.max), tuple(grid.pDim), num_steps, threshold,
           type(sys).__name__, sorted(vars(sys).items()), inspect.getsource(type(sys)),
           inspect.getsource(build_local_update_kernel), inspect.getsource(spatial_deriv), inspect.getsource(hj_update))
    kernel, how = load_or_build_kernel(lambda: build_local_update_kernel(sys, grid, num_steps, threshold),
                                       key, f"local_update_{dims}d", cache_dir)
    print(f"HeteroCL kernel {how} in {time.time() - total_start:.4f} seconds")

    xs = [hcl.asarray(np.reshape(grid.vs[d], grid.pts_each_dim[d]).astype(DTYPE), dtype=DTYPE_HCL) for d in range(dims)]
    t_step_hcl = hcl.asarray(np.array([t_step], dtype=DTYPE), dtype=DTYPE_HCL)
    V_decomp = hcl.asarray(result_decomp, dtype=DTYPE_HCL)
    leaking_hcl = hcl.asarray(leaking.astype(np.int32), dtype=hcl.Int())
    region = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
    done = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
    counts = hcl.asarray(np.zeros(shape[0], dtype=np.int32), dtype=hcl.Int())
    steps = [hcl.asarray(np.array([i], dtype=np.int32), dtype=hcl.Int()) for i in range(num_steps)]
    grow = hcl.asarray(np.array([int(grow_within_step)], dtype=np.int32), dtype=hcl.Int())
    sweeps = hcl.asarray(np.zeros(1, dtype=np.int32), dtype=hcl.Int())
    # Two value buffers, swapped after each time step
    V_prev = hcl.asarray(result_decomp[0], dtype=DTYPE_HCL)
    V_new = hcl.asarray(np.zeros(shape, dtype=DTYPE), dtype=DTYPE_HCL)
    # Free the numpy copy of the decomposition (it is in V_decomp now) before allocating the result
    init = result_decomp[0].copy()
    del result_decomp
    result_local = np.empty((num_steps,) + shape, dtype=DTYPE)
    result_local[0] = init
    del init

    # Step 3: local updating loop over time steps
    start = time.time()
    for i in range(1, num_steps):
        # Step 3a: find the update region, update the value function inside it and (grow_within_step) grow the
        # region within the step, all in one kernel call
        kernel(V_new, V_prev, V_decomp, leaking_hcl, *xs, steps[i], grow, t_step_hcl, region, done, counts, sweeps)

        # Step 3b: save the corrected value function of this time step, then swap the buffers
        result_local[i] = V_new.asnumpy()
        V_prev, V_new = V_new, V_prev

    print(f"Total correction time: {time.time() - start:.4f} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")
    total_updated = int(counts.asnumpy().sum())
    total_sweeps = (num_steps - 1) + int(sweeps.asnumpy()[0])
    print(f"Total point updates: {total_updated} / {int(np.prod(shape)) * (num_steps - 1)}, sweeps: {total_sweeps}")

    return np.moveaxis(result_local, 0, -1)
