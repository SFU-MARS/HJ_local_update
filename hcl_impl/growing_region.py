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

LEAKING = -1  # mark of the leaking set: in the update region at every time step


class HclBackedArray(np.ndarray):
    '''
        numpy view of the memory of a HeteroCL array, which keeps the HeteroCL array alive (also through any
        further view, e.g. np.moveaxis).
    '''
    pass


def hcl_empty(shape, np_dtype):
    '''
        Uninitialized HeteroCL array of the given shape and numpy type (np.float32 for DTYPE_HCL, np.int32 for
        hcl.Int()), and a numpy view of its memory, so it can be filled and read without hcl.asarray / asnumpy:
        those copy everything once more, and convert integer arrays one element at a time in Python.
        Returns (HeteroCL array, numpy view).
    '''
    from heterocl.tvm import ndarray as tvm_nd
    shape = tuple(int(n) for n in shape)
    np_dtype = np.dtype(np_dtype)
    arr = tvm_nd.empty(shape, str(np_dtype))
    ptr = ctypes.cast(arr.handle.contents.data, ctypes.POINTER(np.ctypeslib.as_ctypes_type(np_dtype)))
    view = np.ctypeslib.as_array(ptr, shape=shape).view(HclBackedArray)
    view.hcl_array = arr
    return arr, view


def int_array(values):
    '''
        HeteroCL Int (32-bit) array with the values of values (array-like).
    '''
    values = np.asarray(values)
    arr, view = hcl_empty(values.shape, np.int32)
    view[...] = values
    return arr


class _StepOf:
    '''
        V[s] of a HeteroCL tensor V with time as the first axis, indexed like a tensor of the grid shape
        (what spa_deriv_hcl, deriv_bounds_hcl and value_change_hcl need: indexing and dtype).
    '''
    def __init__(self, V, s):
        self.V, self.s, self.dtype = V, s, V.dtype

    def __getitem__(self, idx):
        return self.V[(self.s,) + tuple(idx)]


def build_local_update_kernel(sys, grid, num_steps, threshold):
    '''
        Used by local_update_hcl_from_file. Returns the compiled HeteroCL kernel of one local update time step s,
        updating V (time as the first axis) in place: V[s-1] is the corrected value function of the previous step,
        V[s] holds the decomposition result and is overwritten where it is corrected.
            1. LocalUpdate (parallel over the first grid dimension): every point in the update region
                   region = leaking set + points (and their neighbours) corrected at step s-1, i.e. that differ from
                            the decomposition result by more than threshold
               gets V[s] = V[s-1] + one HJSolver step (common/hj_update.py). The region is read from mark_read:
               a point corrected at step s-1 marked itself and its neighbours there (a point not updated keeps the
               decomposition result, so it cannot be corrected), and the leaking set is marked LEAKING.
               Points outside the region are not touched: they already hold the decomposition result.
            2. Grow (only with grow = 1, grow_within_step): every corrected point adds its neighbours to the region
               of this step, and the points added are updated, repeated until no new point joins. Sequential, in
               the same kernel call.
        Every update reads only V[s-1], so the order of the updates does not change the result: the same region
        and values as numpy_impl/growing_region.py.
            mark_read, mark_write: region marks of this step (== s) and the next step (set to s + 1), LEAKING for the
                          leaking set; the host swaps them after each step, so a mark is never overwritten while it
                          is still read
            region, done: work arrays for grow = 1, set to s when a point is marked / updated at time step s,
                          so they never need to be reset
            counts: running number of points updated, per parallel task (index of the first two dimensions, or of
                    the first dimension in 2d), read only at the end
            sweeps: running number of Grow passes that updated a point
        Same scheme as HJSolver: 1st order ENO derivatives, Lax-Friedrichs dissipation (with the global
        derivative bounds of V[s-1], computed once per time step, if the dynamics need them), forward Euler,
        32-bit floats.
    '''
    shape = tuple(int(n) for n in grid.pts_each_dim)
    dims = len(shape)
    V = hcl.placeholder((num_steps,) + shape, name="V", dtype=DTYPE_HCL)
    xs = [hcl.placeholder((shape[d],), name=f"x{d}", dtype=DTYPE_HCL) for d in range(dims)]
    step = hcl.placeholder((1,), name="step", dtype=hcl.Int())
    grow = hcl.placeholder((1,), name="grow", dtype=hcl.Int())
    t_step = hcl.placeholder((1,), name="t_step", dtype=DTYPE_HCL)
    mark_read = hcl.placeholder(shape, name="mark_read", dtype=hcl.Int())
    mark_write = hcl.placeholder(shape, name="mark_write", dtype=hcl.Int())
    region = hcl.placeholder(shape, name="region", dtype=hcl.Int())
    done = hcl.placeholder(shape, name="done", dtype=hcl.Int())
    # Points updated, counted per parallel task (each runs in one thread) so no integer mask is copied out
    task_dims = 2 if dims >= 3 else 1
    counts = hcl.placeholder(shape[:task_dims], name="counts", dtype=hcl.Int())
    sweeps = hcl.placeholder((1,), name="sweeps", dtype=hcl.Int())

    def graph_create(V, *rest):
        x = rest[:dims]
        step, grow, t_step, mark_read, mark_write, region, done, counts, sweeps = rest[dims:]
        s = step[0]
        V_prev = _StepOf(V, s - 1)
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

        # Global derivative bounds of V[s-1], for the dissipation like HJSolver (only if the dynamics need them)
        deriv_min = deriv_max = None
        if sys.DISSIPATION_USES_DERIV_BOUNDS:
            deriv_min, deriv_max = deriv_bounds_hcl(V_prev, grid)

        def update(idx):
            # One HJSolver step at idx (the dynamics do not depend on t). If the new value differs from the
            # decomposition result (still in V[s]), the point is corrected: it and its neighbours join the region
            # of the next step, and with grow = 1 the neighbours also join the region of this step
            state = tuple(x[dim][idx[dim]] for dim in range(dims))
            new = hcl.scalar(0, "new", dtype=DTYPE_HCL)
            new[0] = V_prev[idx] + value_change_hcl(V_prev, list(idx), grid, sys, state, t_step[0],
                                                    deriv_min, deriv_max)
            with hcl.if_(grow[0] == 1):
                done[idx] = s

            def mark(nb_idx):
                with hcl.if_(mark_write[nb_idx] != LEAKING):
                    mark_write[nb_idx] = s + 1
            with hcl.if_(abs_hcl(new[0] - V[(s,) + idx]) > thr):
                mark(idx)

                def mark_neighbour(nb_idx):
                    mark(nb_idx)
                    with hcl.if_(grow[0] == 1):
                        region[nb_idx] = s
                neighbours(idx, mark_neighbour)
            V[(s,) + idx] = new[0]

        def first_sweep(idx, count):
            idx = tuple(idx)
            with hcl.if_(hcl.or_(mark_read[idx] == s, mark_read[idx] == LEAKING)):
                count[0] = count[0] + 1
                update(idx)

        def task(task_idx):
            # One parallel task: the points with these first indices, counting its updates in a local scalar
            count = hcl.scalar(0, "count", dtype=hcl.Int())
            for_all_points(shape[task_dims:], lambda rest_idx: first_sweep(task_idx + rest_idx, count), prefix="j")
            counts[tuple(task_idx)] = counts[tuple(task_idx)] + count[0]

        with hcl.Stage("LocalUpdate"):
            # Loops i0 (, i1), j0, ... over the grid
            with hcl.for_(0, shape[0], name="i0") as i0:
                if task_dims == 2:
                    with hcl.for_(0, shape[1], name="i1") as i1:
                        task([i0, i1])
                else:
                    task([i0])

        with hcl.Stage("Grow"):
            # Update the marked points that are not updated yet, until a pass updates none
            again = hcl.scalar(0, "again", dtype=hcl.Int())
            again[0] = grow[0]
            with hcl.while_(again[0] == 1):
                again[0] = 0

                def grow_point(idx):
                    idx = tuple(idx)
                    with hcl.if_(hcl.and_(region[idx] == s, done[idx] != s)):
                        counts[idx[:task_dims]] = counts[idx[:task_dims]] + 1
                        again[0] = 1
                        update(idx)
                for_all_points(shape, grow_point, prefix="g")
                with hcl.if_(again[0] == 1):
                    sweeps[0] = sweeps[0] + 1

    s = hcl.create_schedule([V] + xs + [step, grow, t_step, mark_read, mark_write, region, done, counts, sweeps],
                            graph_create)
    # Thread parallelize the first sweep. From 3 dimensions on, over the first two dimensions together (more,
    # smaller tasks than the first dimension alone, so the threads share the work more evenly); in 2d over the
    # first dimension (fusing would make every grid point a task)
    s_U = graph_create.LocalUpdate
    if dims >= 3:
        s[s_U].parallel(s[s_U].fuse(s_U.i0, s_U.i1))
    else:
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
            grow_within_step: grow the region within each time step until no new point is updated
                              (see numpy_impl/growing_region.py)
        Returns the corrected value function, same shape as the decomposition result (a view of the HeteroCL
        array, time as the first axis in memory).
    '''
    file_shape = np.load(decomp_file, mmap_mode='r').shape
    num_steps = file_shape[-1]
    shape = file_shape[:-1]
    dims = len(shape)

    # Step 1: build (or load the cached) HeteroCL kernel
    hcl.init(DTYPE_HCL)
    total_start = time.time()
    key = (tuple(grid.pts_each_dim), tuple(grid.min), tuple(grid.max), tuple(grid.pDim), num_steps, threshold,
           type(sys).__name__, sorted(vars(sys).items()), inspect.getsource(type(sys)), inspect.getsource(_StepOf),
           inspect.getsource(build_local_update_kernel), inspect.getsource(spatial_deriv), inspect.getsource(hj_update))
    kernel, how = load_or_build_kernel(lambda: build_local_update_kernel(sys, grid, num_steps, threshold),
                                       key, f"local_update_{dims}d", cache_dir)
    print(f"HeteroCL kernel {how} in {time.time() - total_start:.4f} seconds")

    # Step 2: the inputs as HeteroCL arrays. The decomposition result is read from the file (time as the last
    # axis) straight into V (time as the first axis), which the kernel then corrects in place
    V, V_view = hcl_empty((num_steps,) + shape, DTYPE)
    read_time_first(decomp_file, V_view)
    xs = [hcl.asarray(np.reshape(grid.vs[d], grid.pts_each_dim[d]).astype(DTYPE), dtype=DTYPE_HCL) for d in range(dims)]
    t_step_hcl = hcl.asarray(np.array([t_step], dtype=DTYPE), dtype=DTYPE_HCL)
    marks = np.where(leaking, LEAKING, 0).astype(np.int32)
    mark_read, mark_write = int_array(marks), int_array(marks)
    del marks
    zeros = np.zeros(shape, dtype=np.int32)
    region, done = int_array(zeros), int_array(zeros)
    del zeros
    counts = int_array(np.zeros(shape[:2] if dims >= 3 else shape[:1]))
    steps = [int_array([i]) for i in range(num_steps)]
    grow = int_array([int(grow_within_step)])
    sweeps = int_array([0])

    # Step 3: local updating loop over time steps, all in the kernel
    start = time.time()
    for i in range(1, num_steps):
        kernel(V, *xs, steps[i], grow, t_step_hcl, mark_read, mark_write, region, done, counts, sweeps)
        mark_read, mark_write = mark_write, mark_read

    print(f"Total correction time: {time.time() - start:.4f} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")
    total_updated = int(counts.asnumpy().sum())
    total_sweeps = (num_steps - 1) + int(sweeps.asnumpy()[0])
    print(f"Total point updates: {total_updated} / {int(np.prod(shape)) * (num_steps - 1)}, sweeps: {total_sweeps}")

    return np.moveaxis(V_view, 0, -1)
