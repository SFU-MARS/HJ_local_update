# Local update of the 6D quadrotor decomposition result, parallel HeteroCL version.
# Same update as numpy_impl/local_update_6d.py, with 64-bit floats like the numpy version and the direct
# computation; the compiled kernel is cached in kernels/ to skip rebuilding.
import argparse
import inspect
import os
import time

import heterocl as hcl
from heterocl import tvm
import numpy as np

from compute.config_6d import add_local_update_args_6d, setup_local_update_6d
from common import spatial_deriv
from common.params import parse_args_with_params
from common.spatial_deriv import spa_deriv_hcl
from common.stats import printCorrectnessStatistics
from hcl_impl.kernel_cache import load_or_build_kernel

DTYPE = hcl.Float(64)
DIMS = 6


def abs_same_type(x):
    # |x| with the type of x. odp's my_abs stores the result as a 32-bit float, which rounds 64-bit values.
    return hcl.select(x > 0, x, -x)


def build_local_update_kernel(sys, grid, num_steps, threshold):
    '''
        Used by local_update_hcl_from_file. Returns the compiled HeteroCL kernel of one local update time step s:
            region = leaking corners + points (and their neighbours) where V_prev differs from the
                     decomposition result at step s-1 by more than threshold
            V_new = V_prev + t_step * H   inside the region
            V_new = decomposition result at step s   outside the region
        Same scheme as numpy_impl/local_update_6d.py and compute/direct_6d.py: average of the one-sided
        first order derivatives, no dissipation, forward Euler, 64-bit floats.
            updated (output): 1 for every point in the region of this time step
        The loop over the first dimension is parallelized.
    '''
    shape = tuple(int(n) for n in grid.pts_each_dim)
    V_new = hcl.placeholder(shape, name="V_new", dtype=DTYPE)
    V_prev = hcl.placeholder(shape, name="V_prev", dtype=DTYPE)
    V_decomp = hcl.placeholder((num_steps,) + shape, name="V_decomp", dtype=DTYPE)
    leaking = hcl.placeholder(shape, name="leaking", dtype=hcl.Int())
    xs = [hcl.placeholder((shape[d],), name=f"x{d}", dtype=DTYPE) for d in range(DIMS)]
    step = hcl.placeholder((1,), name="step", dtype=hcl.Int())
    t_step = hcl.placeholder((1,), name="t_step", dtype=DTYPE)
    updated = hcl.placeholder(shape, name="updated", dtype=hcl.Int())

    def graph_create(V_new, V_prev, V_decomp, leaking, x0, x1, x2, x3, x4, x5, step, t_step, updated):
        x = (x0, x1, x2, x3, x4, x5)
        # A plain Python float in a kernel becomes a 32-bit constant, so build it as a 64-bit one
        thr = tvm.const(threshold, "float64")

        def corrected(idx):
            # True if the point idx was corrected at the previous time step
            return abs_same_type(V_prev[tuple(idx)] - V_decomp[(step[0] - 1,) + tuple(idx)]) > thr

        def mark_neighbours(in_region, idx):
            # Set in_region if any neighbour along any dimension was corrected (same as dilate in common/leaking.py)
            for dim in range(DIMS):
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
                        with hcl.if_(corrected(nb_idx)):
                            in_region[0] = 1

        with hcl.Stage("LocalUpdate"):
            with hcl.for_(0, shape[0], name="i0") as i0:
                with hcl.for_(0, shape[1], name="i1") as i1:
                    with hcl.for_(0, shape[2], name="i2") as i2:
                        with hcl.for_(0, shape[3], name="i3") as i3:
                            with hcl.for_(0, shape[4], name="i4") as i4:
                                with hcl.for_(0, shape[5], name="i5") as i5:
                                    idx = [i0, i1, i2, i3, i4, i5]
                                    # Update region of this time step
                                    in_region = hcl.scalar(0, "in_region", dtype=hcl.Int())
                                    in_region[0] = leaking[tuple(idx)]
                                    with hcl.if_(corrected(idx)):
                                        in_region[0] = 1
                                    mark_neighbours(in_region, idx)
                                    updated[tuple(idx)] = in_region[0]

                                    with hcl.if_(in_region[0] == 1):
                                        # Average of the left and right spatial derivatives (1st order)
                                        grad = []
                                        for dim in range(DIMS):
                                            dV_L, dV_R = spa_deriv_hcl(idx, V_prev, grid, dim, boundary="one_sided")
                                            g = hcl.scalar(0, f"grad{dim}")
                                            g[0] = (dV_L + dV_R) / 2
                                            grad.append(g[0])

                                        # Optimal control and dynamical rates of change (the dynamics do not depend on t)
                                        state = tuple(x[dim][idx[dim]] for dim in range(DIMS))
                                        uOpt = sys.opt_ctrl(0, state, grad)
                                        rates = sys.dynamics(0, state, uOpt, (0,) * DIMS)

                                        # Forward Euler step, same order of additions as compute/direct_6d.py
                                        change = hcl.scalar(0, "change")
                                        change[0] = rates[0] * grad[0]
                                        for dim in range(1, DIMS):
                                            change[0] = change[0] + rates[dim] * grad[dim]
                                        V_new[tuple(idx)] = V_prev[tuple(idx)] + change[0] * t_step[0]
                                    with hcl.else_():
                                        # Outside the update region keep the decomposition result
                                        V_new[tuple(idx)] = V_decomp[(step[0],) + tuple(idx)]

    s = hcl.create_schedule([V_new, V_prev, V_decomp, leaking] + xs + [step, t_step, updated], graph_create)
    # Thread parallelize the loop over the first dimension
    s_U = graph_create.LocalUpdate
    s[s_U].parallel(s_U.i0)
    return hcl.build(s)


def local_update_hcl_from_file(decomp_file, leaking, grid, sys, t_step, threshold=1e-6, cache_dir="kernels"):
    '''
        Used by main. Same growing-region local update as local_update_from_file in numpy_impl/local_update_6d.py,
        the update of each time step runs in a parallel HeteroCL kernel.
            decomp_file: npy file of the decomposition result, shape (x, y, vx, vy, theta, w, t), index 0 of the last axis is t = 0
            leaking: boolean mask of the initial leaking set, see get_leaking_set in common/leaking.py
            grid: odp Grid of the 6D system
            sys: dynamics with HeteroCL methods opt_ctrl and dynamics (see dynamics/__init__.py)
            t_step: time step used to compute decomp_file
            threshold: a point counts as corrected if it differs from the decomposition by more than this
            cache_dir: directory of the compiled kernels (see hcl_impl/kernel_cache.py)
        Returns the corrected value function, same shape as the decomposition result.
    '''
    # Step 1: load the decomposition result with time as the first axis
    result_decomp = np.ascontiguousarray(np.moveaxis(np.load(decomp_file).astype(np.float64, copy=False), -1, 0))
    num_steps = result_decomp.shape[0]
    shape = result_decomp.shape[1:]

    # Step 2: build (or load the cached) HeteroCL kernel and copy all inputs to HeteroCL arrays once
    hcl.init(DTYPE)
    total_start = time.time()
    key = (tuple(grid.pts_each_dim), tuple(grid.min), tuple(grid.max), tuple(grid.pDim), num_steps, threshold,
           type(sys).__name__, sorted(vars(sys).items()), inspect.getsource(type(sys)),
           inspect.getsource(build_local_update_kernel), inspect.getsource(spatial_deriv))
    kernel, how = load_or_build_kernel(lambda: build_local_update_kernel(sys, grid, num_steps, threshold),
                                       key, "local_update_6d", cache_dir)
    print(f"HeteroCL kernel {how} in {time.time() - total_start:.4f} seconds")

    xs = [hcl.asarray(np.reshape(grid.vs[d], grid.pts_each_dim[d]).astype(np.float64), dtype=DTYPE) for d in range(DIMS)]
    t_step_hcl = hcl.asarray(np.array([t_step], dtype=np.float64), dtype=DTYPE)
    V_decomp = hcl.asarray(result_decomp, dtype=DTYPE)
    leaking_hcl = hcl.asarray(leaking.astype(np.int32), dtype=hcl.Int())
    updated = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
    steps = [hcl.asarray(np.array([i], dtype=np.int32), dtype=hcl.Int()) for i in range(num_steps)]
    # Two value buffers, swapped after each time step
    V_prev = hcl.asarray(result_decomp[0], dtype=DTYPE)
    V_new = hcl.asarray(np.zeros(shape, dtype=np.float64), dtype=DTYPE)
    result_local = np.empty_like(result_decomp)
    result_local[0] = result_decomp[0]
    del result_decomp

    # Step 3: local updating loop over time steps
    total_updated = 0
    start = time.time()
    for i in range(1, num_steps):
        # Step 3a: find the update region and update the value function inside it, all in the kernel
        kernel(V_new, V_prev, V_decomp, leaking_hcl, *xs, steps[i], t_step_hcl, updated)

        # Step 3b: save the corrected value function of this time step, then swap the buffers
        result_local[i] = V_new.asnumpy()
        V_prev, V_new = V_new, V_prev
        total_updated += int(updated.asnumpy().sum())

    print(f"Total correction time: {time.time() - start:.4f} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")
    print(f"Total point updates: {total_updated} / {int(np.prod(shape)) * (num_steps - 1)}")

    return np.moveaxis(result_local, 0, -1)


def main():
    '''
        Read the decomposition (compute/decomposition_6d.py) and, if given, the direct computation
        (compute/direct_6d.py), run the HeteroCL local update, save it to results/result_local_hcl_6d_<combine>.npy
        and compare it with the direct computation.
    '''
    # Steps 1-4: inputs, dynamics, grid and initial leaking set (see compute/config_6d.py)
    parser = argparse.ArgumentParser()
    add_local_update_args_6d(parser)
    parser.add_argument("--cache_dir", default="kernels",
                        help="Directory of the compiled HeteroCL kernels. A kernel built before with the same "
                        "settings is loaded from here instead of being rebuilt.")
    args = parse_args_with_params(parser)
    sys, grid, leaking = setup_local_update_6d(parser, args)

    # Step 5: run the local update and save the corrected result
    result_local = local_update_hcl_from_file(args.decomp_file, leaking, grid, sys, args.t_step, cache_dir=args.cache_dir)
    os.makedirs("results", exist_ok=True)
    path = f"results/result_local_hcl_6d_{args.combine}.npy"
    np.save(path, result_local)
    print(f"Saved {path}")

    # Step 6: compare the decomposition and local update results with the direct computation (if given)
    if args.true_file is not None:
        printCorrectnessStatistics(np.load(args.true_file), np.load(args.decomp_file), result_local)
    else:
        print("No --true_file given, skipping the comparison with the direct computation")


if __name__ == "__main__":
    main()
