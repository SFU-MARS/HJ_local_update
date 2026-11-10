# Local update of the 3D Dubins car decomposition result, parallel HeteroCL version.
# Same update as numpy_impl/local_update_3d.py; the compiled kernel is cached in kernels/ to skip rebuilding.
import argparse
import hashlib
import inspect
import math
import os
import time

import heterocl as hcl
from heterocl.tvm import module as hcl_module
import numpy as np

from odp.Grid import Grid
from odp.dynamics import DubinsCar
from odp.computeGraphs.CustomGraphFunctions import my_abs, my_max
from odp.spatialDerivatives.firstOrderENO.first_orderENO3D import spa_derivX, spa_derivY, spa_derivT

from numpy_impl.local_update_3d import dilate, get_leaking_set, parse_args_with_params, printCorrectnessStatistics


def build_local_update_kernel(sys, grid, num_steps, threshold):
    '''
        Used by load_or_build_kernel. Returns the compiled HeteroCL kernel of one local update time step s:
            region = leaking corners + points (and their neighbours) where V_prev differs from the
                     decomposition result at step s-1 by more than threshold
            V_new = V_prev + t_step * (H + dissipation)   inside the region
            V_new = decomposition result at step s        outside the region
        Same first order scheme as numpy_impl/local_update_3d.py (1st order ENO + Lax-Friedrichs + forward Euler).
        The whole decomposition result stays in one HeteroCL array (shape (t, x, y, theta)), so no
        numpy <-> HeteroCL conversion is needed between time steps. The loop over x is parallelized.
    '''
    shape = tuple(grid.pts_each_dim)
    V_new = hcl.placeholder(shape, name="V_new", dtype=hcl.Float())
    V_prev = hcl.placeholder(shape, name="V_prev", dtype=hcl.Float())
    V_decomp = hcl.placeholder((num_steps,) + shape, name="V_decomp", dtype=hcl.Float())
    leaking = hcl.placeholder(shape, name="leaking", dtype=hcl.Int())
    x1 = hcl.placeholder((shape[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((shape[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((shape[2],), name="x3", dtype=hcl.Float())
    step = hcl.placeholder((1,), name="step", dtype=hcl.Int())
    t_step = hcl.placeholder((1,), name="t_step", dtype=hcl.Float())

    def graph_create(V_new, V_prev, V_decomp, leaking, x1, x2, x3, step, t_step):
        # Computation graph of the kernel, one pass over the grid
        def corrected(a, b, c):
            # True if point (a, b, c) was corrected at the previous time step
            return my_abs(V_prev[a, b, c] - V_decomp[step[0] - 1, a, b, c]) > threshold

        def mark_neighbours(in_region, i, j, k):
            # Set in_region if any neighbour along any dimension was corrected (same as dilate in numpy_impl/local_update_3d.py)
            idx = (i, j, k)
            for dim in range(3):
                n = shape[dim]
                for offset in (-1, 1):
                    nb = hcl.scalar(0, "nb", dtype=hcl.Int())
                    nb[0] = idx[dim] + offset
                    inside = hcl.scalar(1, "inside", dtype=hcl.Int())
                    if dim in grid.pDim:
                        # Periodic dimension wraps around
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
                        nb_idx = [i, j, k]
                        nb_idx[dim] = nb[0]
                        with hcl.if_(corrected(*nb_idx)):
                            in_region[0] = 1

        with hcl.Stage("LocalUpdate"):
            with hcl.for_(0, shape[0], name="i") as i:
                with hcl.for_(0, shape[1], name="j") as j:
                    with hcl.for_(0, shape[2], name="k") as k:
                        # Update region of this time step
                        in_region = hcl.scalar(0, "in_region", dtype=hcl.Int())
                        in_region[0] = leaking[i, j, k]
                        with hcl.if_(corrected(i, j, k)):
                            in_region[0] = 1
                        mark_neighbours(in_region, i, j, k)

                        with hcl.if_(in_region[0] == 1):
                            # Left and right spatial derivatives (1st order)
                            dV_dx_L = hcl.scalar(0, "dV_dx_L")
                            dV_dx_R = hcl.scalar(0, "dV_dx_R")
                            dV_dy_L = hcl.scalar(0, "dV_dy_L")
                            dV_dy_R = hcl.scalar(0, "dV_dy_R")
                            dV_dT_L = hcl.scalar(0, "dV_dT_L")
                            dV_dT_R = hcl.scalar(0, "dV_dT_R")
                            dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k, V_prev, grid)
                            dV_dy_L[0], dV_dy_R[0] = spa_derivY(i, j, k, V_prev, grid)
                            dV_dT_L[0], dV_dT_R[0] = spa_derivT(i, j, k, V_prev, grid)

                            # Average gradient
                            dV_dx = hcl.scalar(0, "dV_dx")
                            dV_dy = hcl.scalar(0, "dV_dy")
                            dV_dT = hcl.scalar(0, "dV_dT")
                            dV_dx[0] = (dV_dx_L[0] + dV_dx_R[0]) / 2
                            dV_dy[0] = (dV_dy_L[0] + dV_dy_R[0]) / 2
                            dV_dT[0] = (dV_dT_L[0] + dV_dT_R[0]) / 2

                            # Optimal control and dynamical rates of change (the dynamics do not depend on t)
                            state = (x1[i], x2[j], x3[k])
                            uOpt = sys.opt_ctrl(0, state, (dV_dx[0], dV_dy[0], dV_dT[0]))
                            dx_dt, dy_dt, dT_dt = sys.dynamics(0, state, uOpt, (0, 0, 0))

                            # Lax-Friedrichs dissipation, alpha is the max rate of change over the control set
                            r1_x, r1_y, r1_T = sys.dynamics(0, state, (sys.wMax, 0, 0), (0, 0, 0))
                            r2_x, r2_y, r2_T = sys.dynamics(0, state, (-sys.wMax, 0, 0), (0, 0, 0))
                            alpha_x = hcl.scalar(0, "alpha_x")
                            alpha_y = hcl.scalar(0, "alpha_y")
                            alpha_T = hcl.scalar(0, "alpha_T")
                            alpha_x[0] = my_max(my_abs(r1_x), my_abs(r2_x))
                            alpha_y[0] = my_max(my_abs(r1_y), my_abs(r2_y))
                            alpha_T[0] = my_max(my_abs(r1_T), my_abs(r2_T))
                            diss = hcl.scalar(0, "diss")
                            diss[0] = 0.5 * ((dV_dx_R[0] - dV_dx_L[0]) * alpha_x[0]
                                             + (dV_dy_R[0] - dV_dy_L[0]) * alpha_y[0]
                                             + (dV_dT_R[0] - dV_dT_L[0]) * alpha_T[0])

                            # Forward Euler step
                            V_new[i, j, k] = V_prev[i, j, k] + t_step[0] * (
                                dx_dt * dV_dx[0] + dy_dt * dV_dy[0] + dT_dt * dV_dT[0] + diss[0])
                        with hcl.else_():
                            # Outside the update region keep the decomposition result
                            V_new[i, j, k] = V_decomp[step[0], i, j, k]

    s = hcl.create_schedule([V_new, V_prev, V_decomp, leaking, x1, x2, x3, step, t_step], graph_create)
    # Thread parallelize the loop over x
    s_U = graph_create.LocalUpdate
    s[s_U].parallel(s_U.i)
    return hcl.build(s)


def load_or_build_kernel(sys, grid, num_steps, threshold, cache_dir="kernels"):
    '''
        Used by local_update_hcl_from_file. Load the compiled kernel from cache_dir if it was built
        before with the same settings, otherwise build it and save it there. The file name includes
        the grid size, number of time steps, threshold, dynamics parameters and a hash of the kernel
        source code, so any change builds a new kernel.
        Returns (kernel, "loaded" or "built").
    '''
    key = repr((tuple(grid.pts_each_dim), tuple(grid.min), tuple(grid.max), tuple(grid.pDim), num_steps, threshold,
                type(sys).__name__, sys.wMax, sys.speed, sys.uMode, inspect.getsource(build_local_update_kernel)))
    path = os.path.join(cache_dir, f"local_update_{hashlib.md5(key.encode()).hexdigest()[:12]}.so")
    if os.path.isfile(path):
        return hcl_module.load(path), "loaded"
    kernel = build_local_update_kernel(sys, grid, num_steps, threshold)
    os.makedirs(cache_dir, exist_ok=True)
    kernel.export_library(path)
    return kernel, "built"


def local_update_hcl_from_file(decomp_file, leaking, grid, t_step, threshold=1e-6, cache_dir="kernels"):
    '''
        Used by main. Same growing-region local update as local_update_from_file in numpy_impl/local_update_3d.py,
        the update of each time step runs in a parallel HeteroCL kernel.
            decomp_file: npy file of the decomposition result, shape (x, y, theta, t), index 0 of the last axis is t = 0
            leaking: boolean mask of the initial leaking set, see get_leaking_set in numpy_impl/local_update_3d.py
            grid: odp Grid of the 3D system
            t_step: time step used to compute decomp_file
            threshold: a point counts as corrected if it differs from the decomposition by more than this
            cache_dir: directory of the compiled kernels, a kernel built before with the same settings
                       is loaded from here instead of being rebuilt (see load_or_build_kernel)
        Returns the corrected value function, same shape as the decomposition result.
    '''
    # Step 1: load the decomposition result, the initial value function is the decomposition at t = 0
    result_decomp = np.load(decomp_file).astype(np.float32)
    data_init = np.ascontiguousarray(result_decomp[..., 0])
    num_steps = result_decomp.shape[-1]

    # Step 2: build (or load the cached) HeteroCL kernel and copy all inputs to HeteroCL arrays once
    hcl.init(hcl.Float(32))
    sys = DubinsCar(wMax=1, speed=1, uMode="max", dMode="min")
    total_start = time.time()
    kernel, how = load_or_build_kernel(sys, grid, num_steps, threshold, cache_dir)
    print(f"HeteroCL kernel {how} in {time.time() - total_start:.4f} seconds")

    x1 = hcl.asarray(np.reshape(grid.vs[0], grid.pts_each_dim[0]))
    x2 = hcl.asarray(np.reshape(grid.vs[1], grid.pts_each_dim[1]))
    x3 = hcl.asarray(np.reshape(grid.vs[2], grid.pts_each_dim[2]))
    t_step_hcl = hcl.asarray(np.array([t_step], dtype=np.float32))
    # Decomposition result with time as the first axis, (t, x, y, theta)
    V_decomp = hcl.asarray(np.ascontiguousarray(np.moveaxis(result_decomp, -1, 0)))
    leaking_hcl = hcl.asarray(leaking.astype(np.int32), dtype=hcl.Int())
    steps = [hcl.asarray(np.array([i], dtype=np.int32), dtype=hcl.Int()) for i in range(num_steps)]
    # Two value buffers, swapped after each time step
    V_prev = hcl.asarray(data_init)
    V_new = hcl.asarray(np.zeros(data_init.shape, dtype=np.float32))
    result_local = np.empty((num_steps,) + data_init.shape, dtype=np.float32)
    result_local[0] = data_init

    # Step 3: local updating loop over time steps
    start = time.time()

    for i in range(1, num_steps):
        # Step 3a: find the update region and update the value function inside it, all in the kernel
        kernel(V_new, V_prev, V_decomp, leaking_hcl, x1, x2, x3, steps[i], t_step_hcl)

        # Step 3b: save the corrected value function of this time step, then swap the buffers
        result_local[i] = V_new.asnumpy()
        V_prev, V_new = V_new, V_prev

    execution_time = time.time() - start
    print(f"Total correction time: {execution_time:.4f} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")

    result_local = np.moveaxis(result_local, 0, -1)

    # Number of updated points, recomputed from the results (not part of the timed loop)
    total_updated = 0
    for i in range(1, num_steps):
        corrected = abs(result_local[..., i-1] - result_decomp[..., i-1]) > threshold
        total_updated += int((dilate(corrected, grid) | leaking).sum())
    print(f"Total point updates: {total_updated} / {data_init.size * (num_steps - 1)}")

    return result_local


def main():
    '''
        Read the decomposition and direct results, run the HeteroCL local update, save the result
        to results/result_local_hcl_3d.npy and compare it with the direct computation.
    '''
    # Step 1: parse and check the command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="JSON parameter file (see hcl_impl/params/), its keys are the argument names below "
                        "without '--'. Arguments given on the command line override the file.", default=None)
    parser.add_argument("--decomp_file", help="npy file of the decomposition result, shape (x, y, theta, t). "
                        "Required, on the command line or in --params.", default=None)
    parser.add_argument("--true_file", help="Optional npy file of the direct computation result, same shape as --decomp_file. "
                        "If not given, the local update runs without the comparison.", default=None)
    parser.add_argument("--init_1", help="npy file of the subsystem 1 initial value V1, shape (x, y, theta), "
                        "saved by compute/decomposition_3d.py.", default="results/init_1.npy")
    parser.add_argument("--init_2", help="npy file of the subsystem 2 initial value V2, shape (x, y, theta), "
                        "saved by compute/decomposition_3d.py.", default="results/init_2.npy")
    parser.add_argument("--leaking_mode", choices=["smallest", "delta"], default="smallest",
                        help="Initial leaking set: 'smallest' = switching surface V1 = V2 plus its neighbours (default), "
                        "'delta' = |V1 - V2| < delta.")
    parser.add_argument("--t_step", help="Time step used to compute both files.", default=0.04, type=float)
    parser.add_argument("--cache_dir", help="Directory of the compiled HeteroCL kernels. A kernel built before with the "
                        "same settings is loaded from here instead of being rebuilt.", default="kernels")
    # Values from --params replace the defaults, so command-line arguments still override them
    args = parse_args_with_params(parser)
    if args.decomp_file is None:
        parser.error("--decomp_file is required (on the command line or in --params)")
    for path in (args.decomp_file, args.true_file, args.init_1, args.init_2):
        if path is not None and not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    result_decomp = np.load(args.decomp_file)
    result_true = None
    if args.true_file is not None:
        result_true = np.load(args.true_file)
        if result_decomp.shape != result_true.shape:
            parser.error(f"shape mismatch: {args.decomp_file} {result_decomp.shape} vs {args.true_file} {result_true.shape}, "
                         "generate both files with the same t_step and accuracy (see README)")
    data_init_upper = np.load(args.init_1)
    data_init_lower = np.load(args.init_2)
    for path, data in ((args.init_1, data_init_upper), (args.init_2, data_init_lower)):
        if data.shape != result_decomp.shape[:-1]:
            parser.error(f"shape mismatch: {path} {data.shape} vs {args.decomp_file} {result_decomp.shape[:-1]} (without t), "
                         "generate them with the same compute/decomposition_3d.py run")

    # Step 2: build the 3D grid with the grid size read from the decomposition file, theta is periodic
    num = result_decomp.shape[0]
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([num, num, num]), [2])

    # Step 3: initial leaking set from the subsystem initial values (--init_1, --init_2)
    leaking = get_leaking_set(data_init_upper, data_init_lower, result_decomp, grid, args.leaking_mode)

    # Step 4: run the local update and save the corrected result
    result_local = local_update_hcl_from_file(args.decomp_file, leaking, grid, args.t_step, cache_dir=args.cache_dir)
    np.save('results/result_local_hcl_3d.npy', result_local)

    # Step 5: compare the decomposition and local update results with the direct computation (if given)
    if result_true is not None:
        printCorrectnessStatistics(result_true, result_decomp, result_local)
    else:
        print("No --true_file given, skipping the comparison with the direct computation")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
