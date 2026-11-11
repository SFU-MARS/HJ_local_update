# Growing-region local update in a parallel HeteroCL kernel, for any number of dimensions (used by every
# hcl_impl/local_update_<dim>.py). Same update as numpy_impl/growing_region.py, in 32-bit floats like HJSolver
# (common/hj_update.py).
import ctypes
import inspect
import time

import heterocl as hcl
from heterocl import tvm
import numpy as np

from common import hj_update, spatial_deriv
from common.npy_io import read_time_first
from common.hj_update import DTYPE, DTYPE_HCL, abs_hcl, deriv_bounds_hcl, for_all_points, value_change_hcl
from hcl_impl.kernel_cache import load_or_build_kernel


def hcl_empty(shape, np_dtype):
    '''
        Uninitialized HeteroCL array of the given shape and numpy type (np.float32 for DTYPE_HCL, np.int32 for
        hcl.Int()), and a numpy view of its memory, so it can be filled without hcl.asarray: hcl.asarray copies
        everything once more, and converts integer arrays one element at a time in Python (seconds for large grids).
        Returns (HeteroCL array, numpy view).
    '''
    from heterocl.tvm import ndarray as tvm_nd
    np_dtype = np.dtype(np_dtype)
    arr = tvm_nd.empty(tuple(int(n) for n in shape), str(np_dtype))
    ptr = ctypes.cast(arr.handle.contents.data, ctypes.POINTER(np.ctypeslib.as_ctypes_type(np_dtype)))
    return arr, np.ctypeslib.as_array(ptr, shape=tuple(int(n) for n in shape))


def int_array(values=None, shape=None):
    '''
        HeteroCL Int (32-bit) array with the values of values (array-like), or zeros of the given shape.
    '''
    if values is not None:
        values = np.asarray(values)
        shape = values.shape
    arr, view = hcl_empty(shape, np.int32)
    if values is not None:
        view[...] = values
    else:
        view.fill(0)
    return arr


def copy_to_numpy(src, out):
    '''
        Copy the float HeteroCL array src into the C-contiguous numpy array out (same size and float type), the
        same byte copy as src.asnumpy() but without allocating a new array and copying it again.
    '''
    from heterocl.tvm._ffi.base import _LIB, check_call
    assert out.flags['C_CONTIGUOUS'] and out.dtype == np.dtype(src.dtype) and out.shape == tuple(src.shape)
    check_call(_LIB.TVMArrayCopyToBytes(src.handle, out.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_size_t(out.nbytes)))


def build_local_update_kernel(sys, grid, num_steps, threshold):
    '''
        Used by local_update_hcl_from_file. Returns the compiled HeteroCL kernel of one local update time step s:
            1. LocalUpdate (parallel over the first dimension), at every point:
                   region = leaking set + points (and their neighbours) where V_prev differs from the
                            decomposition result at step s-1 by more than threshold, read from mark_read: the
                            points updated at step s-1 that changed marked themselves and their neighbours there
                            (points not updated keep the decomposition result, so they cannot differ)
                   V_new = V_prev + one HJSolver step   inside the region (common/hj_update.py)
                   V_new = decomposition result at step s   outside the region
            2. Grow (only with grow = 1, grow_within_step): every updated point that differs from the decomposition
               result at step s by more than threshold adds its neighbours to the region, and the points added are
               updated, repeated until no new point joins. Sequential, in the same kernel call.
        Every update reads only V_prev, so the order of the updates does not change the result: the same region
        and values as numpy_impl/growing_region.py.
            mark_read, mark_write: region marks of this step (== s) and the next step (set to s + 1); the host
                          swaps them after each step, so a mark is never overwritten while it is still read
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
    mark_read = hcl.placeholder(shape, name="mark_read", dtype=hcl.Int())
    mark_write = hcl.placeholder(shape, name="mark_write", dtype=hcl.Int())
    region = hcl.placeholder(shape, name="region", dtype=hcl.Int())
    done = hcl.placeholder(shape, name="done", dtype=hcl.Int())
    # Points updated, counted per i0 (each i0 runs in one thread) so no integer mask is copied out
    counts = hcl.placeholder((shape[0],), name="counts", dtype=hcl.Int())
    sweeps = hcl.placeholder((1,), name="sweeps", dtype=hcl.Int())

    def graph_create(V_new, V_prev, V_decomp, leaking, *rest):
        x = rest[:dims]
        step, grow, t_step, mark_read, mark_write, region, done, counts, sweeps = rest[dims:]
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

        # Global derivative bounds of V_prev, for the dissipation like HJSolver (only if the dynamics need them)
        deriv_min = deriv_max = None
        if sys.DISSIPATION_USES_DERIV_BOUNDS:
            deriv_min, deriv_max = deriv_bounds_hcl(V_prev, grid)

        def update(idx):
            # One HJSolver step at idx (the dynamics do not depend on t). If the new value differs from the
            # decomposition (the point is corrected), the point and its neighbours join the region of the next
            # step, and with grow = 1 the neighbours also join the region of this step
            state = tuple(x[dim][idx[dim]] for dim in range(dims))
            V_new[idx] = V_prev[idx] + value_change_hcl(V_prev, list(idx), grid, sys, state, t_step[0],
                                                        deriv_min, deriv_max)
            with hcl.if_(grow[0] == 1):
                done[idx] = s
            with hcl.if_(abs_hcl(V_new[idx] - V_decomp[(s,) + idx]) > thr):
                mark_write[idx] = s + 1

                def mark(nb_idx):
                    mark_write[nb_idx] = s + 1
                    with hcl.if_(grow[0] == 1):
                        region[nb_idx] = s
                neighbours(idx, mark)

        def first_sweep(idx, count):
            idx = tuple(idx)
            with hcl.if_(hcl.or_(leaking[idx] == 1, mark_read[idx] == s)):
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

    s = hcl.create_schedule([V_new, V_prev, V_decomp, leaking] + xs + [step, grow, t_step, mark_read, mark_write, region, done, counts, sweeps],
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
    file_shape = np.load(decomp_file, mmap_mode='r').shape
    num_steps = file_shape[-1]
    shape = file_shape[:-1]
    dims = len(shape)

    # Step 1: build (or load the cached) HeteroCL kernel
    hcl.init(DTYPE_HCL)
    total_start = time.time()
    key = (tuple(grid.pts_each_dim), tuple(grid.min), tuple(grid.max), tuple(grid.pDim), num_steps, threshold,
           type(sys).__name__, sorted(vars(sys).items()), inspect.getsource(type(sys)),
           inspect.getsource(build_local_update_kernel), inspect.getsource(spatial_deriv), inspect.getsource(hj_update))
    kernel, how = load_or_build_kernel(lambda: build_local_update_kernel(sys, grid, num_steps, threshold),
                                       key, f"local_update_{dims}d", cache_dir)
    print(f"HeteroCL kernel {how} in {time.time() - total_start:.4f} seconds")

    # Step 2: the inputs as HeteroCL arrays. The decomposition result is read from the file (time as the last
    # axis) straight into V_decomp (time as the first axis)
    V_decomp, V_decomp_view = hcl_empty((num_steps,) + shape, DTYPE)
    read_time_first(decomp_file, V_decomp_view)

    xs = [hcl.asarray(np.reshape(grid.vs[d], grid.pts_each_dim[d]).astype(DTYPE), dtype=DTYPE_HCL) for d in range(dims)]
    t_step_hcl = hcl.asarray(np.array([t_step], dtype=DTYPE), dtype=DTYPE_HCL)
    leaking_hcl = int_array(leaking)
    mark_read, mark_write, region, done = (int_array(shape=shape) for _ in range(4))
    counts = int_array(shape=(shape[0],))
    steps = [int_array([i]) for i in range(num_steps)]
    grow = int_array([int(grow_within_step)])
    sweeps = int_array(shape=(1,))
    # Two value buffers, swapped after each time step (the first sweep writes every point of V_new)
    V_prev, V_prev_view = hcl_empty(shape, DTYPE)
    V_prev_view[...] = V_decomp_view[0]
    V_new, _ = hcl_empty(shape, DTYPE)
    result_local = np.empty((num_steps,) + shape, dtype=DTYPE)
    result_local[0] = V_decomp_view[0]
    del V_decomp_view, V_prev_view

    # Step 3: local updating loop over time steps
    start = time.time()
    for i in range(1, num_steps):
        # Step 3a: find the update region, update the value function inside it and (grow_within_step) grow the
        # region within the step, all in one kernel call
        kernel(V_new, V_prev, V_decomp, leaking_hcl, *xs, steps[i], grow, t_step_hcl, mark_read, mark_write, region, done,
               counts, sweeps)

        # Step 3b: save the corrected value function of this time step, then swap the buffers
        copy_to_numpy(V_new, result_local[i])
        V_prev, V_new = V_new, V_prev
        mark_read, mark_write = mark_write, mark_read

    print(f"Total correction time: {time.time() - start:.4f} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")
    total_updated = int(counts.asnumpy().sum())
    total_sweeps = (num_steps - 1) + int(sweeps.asnumpy()[0])
    print(f"Total point updates: {total_updated} / {int(np.prod(shape)) * (num_steps - 1)}, sweeps: {total_sweeps}")

    return np.moveaxis(result_local, 0, -1)
