# Local update of the 2D coupled control decomposition result, HeteroCL version.
# Same update as numpy_impl/local_update_2d.py (CorrectionBasedOnLocalUpdate); the compiled kernel is cached
# in kernels/ to skip rebuilding.
import argparse
import copy
import inspect
import os
import time

import heterocl as hcl
from heterocl import tvm
import numpy as np

from compute.config_2d import add_local_update_args_2d, setup_local_update_2d
from compute.decomposition_2d import DecompositionResult
from common.combine import check_combine
from common.params import parse_args_with_params
from common.stats import printCorrectnessStatistics2d
from hcl_impl.kernel_cache import load_or_build_kernel
from common import hj_update, spatial_deriv
from common.hj_update import deriv_bounds_hcl, value_change_hcl

DTYPE = hcl.Float(64)


def abs_same_type(x):
    # |x| with the type of x. odp's my_abs stores the result as a 32-bit float, which rounds 64-bit values.
    return hcl.select(x > 0, x, -x)


def build_local_update_kernel(sys, grid, num_steps, threshold):
    '''
        Used by local_update_hcl. Returns the compiled HeteroCL kernel of one local update time step s:
            1. Identify: in_region = |V1 - V2| < big_delta at step s (the leaking corners)
            2. Grow: repeat until nothing changes: every point in the region that is not done yet gets
               V_new = V_prev + t_step * H; if V_new differs from the decomposition result at step s by
               more than threshold, its 4 neighbours (wrapping around the grid edge) join the region
            3. Every point outside the region keeps the decomposition result at step s
        Every update reads only V_prev, so the order of the updates does not change the result, and it is
        the same region and values as CorrectionBasedOnLocalUpdate.doCorrection in numpy_impl/local_update_2d.py.
        Same scheme as the numpy version and HJSolver (common/hj_update.py): 1st order ENO derivatives,
        Lax-Friedrichs dissipation with the global derivative bounds of V_prev (stage "Bounds") if the dynamics
        need them, forward Euler, 64-bit floats.
            in_region, done: work buffers (region, and points updated at this time step)
            counts (output): [number of leaking corners found in step 1, number of points updated], counted
                             in the kernel so the integer masks never need to be copied out (slow in HeteroCL)
    '''
    shape = tuple(int(n) for n in grid.pts_each_dim)
    nx, ny = shape
    V_new = hcl.placeholder(shape, name="V_new", dtype=DTYPE)
    V_prev = hcl.placeholder(shape, name="V_prev", dtype=DTYPE)
    V_decomp = hcl.placeholder((num_steps,) + shape, name="V_decomp", dtype=DTYPE)
    V_1 = hcl.placeholder((num_steps,) + shape, name="V_1", dtype=DTYPE)
    V_2 = hcl.placeholder((num_steps,) + shape, name="V_2", dtype=DTYPE)
    x1 = hcl.placeholder((nx,), name="x1", dtype=DTYPE)
    x2 = hcl.placeholder((ny,), name="x2", dtype=DTYPE)
    step = hcl.placeholder((1,), name="step", dtype=hcl.Int())
    t_step = hcl.placeholder((1,), name="t_step", dtype=DTYPE)
    big_delta = hcl.placeholder((1,), name="big_delta", dtype=DTYPE)
    in_region = hcl.placeholder(shape, name="in_region", dtype=hcl.Int())
    done = hcl.placeholder(shape, name="done", dtype=hcl.Int())
    counts = hcl.placeholder((2,), name="counts", dtype=hcl.Int())

    def graph_create(V_new, V_prev, V_decomp, V_1, V_2, x1, x2, step, t_step, big_delta, in_region, done, counts):
        s = step[0]
        # A plain Python float in a kernel becomes a 32-bit constant, so build it as a 64-bit one
        thr = tvm.const(threshold, "float64")

        # Step 1: start from the decomposition result, the region is the leaking corners
        with hcl.Stage("Identify"):
            counts[0] = 0
            counts[1] = 0
            with hcl.for_(0, nx, name="i") as i:
                with hcl.for_(0, ny, name="j") as j:
                    V_new[i, j] = V_decomp[s, i, j]
                    done[i, j] = 0
                    in_region[i, j] = 0
                    with hcl.if_(abs_same_type(V_1[s, i, j] - V_2[s, i, j]) < big_delta[0]):
                        in_region[i, j] = 1
                        counts[0] = counts[0] + 1

        # Global derivative bounds of V_prev, for the dissipation like HJSolver
        deriv_min = deriv_max = None
        if sys.DISSIPATION_USES_DERIV_BOUNDS:
            deriv_min, deriv_max = deriv_bounds_hcl(V_prev, grid)

        # Step 2: update the region and grow it until no new point joins
        with hcl.Stage("Grow"):
            again = hcl.scalar(1, "again", dtype=hcl.Int())
            with hcl.while_(again[0] == 1):
                again[0] = 0
                with hcl.for_(0, nx, name="i") as i:
                    with hcl.for_(0, ny, name="j") as j:
                        with hcl.if_(hcl.and_(in_region[i, j] == 1, done[i, j] == 0)):
                            # One HJSolver step (the dynamics do not depend on t)
                            V_new[i, j] = V_prev[i, j] + value_change_hcl(V_prev, [i, j], grid, sys, (x1[i], x2[j]),
                                                                           t_step[0], deriv_min, deriv_max)
                            done[i, j] = 1
                            counts[1] = counts[1] + 1
                            again[0] = 1

                            # A point that changed spreads the region to its neighbours, wrapping around
                            # the grid edge like prevIndexOnX ... nextIndexOnY in the numpy version
                            with hcl.if_(abs_same_type(V_new[i, j] - V_decomp[s, i, j]) > thr):
                                with hcl.if_(i == 0):
                                    in_region[nx - 1, j] = 1
                                with hcl.else_():
                                    in_region[i - 1, j] = 1
                                with hcl.if_(i == nx - 1):
                                    in_region[0, j] = 1
                                with hcl.else_():
                                    in_region[i + 1, j] = 1
                                with hcl.if_(j == 0):
                                    in_region[i, ny - 1] = 1
                                with hcl.else_():
                                    in_region[i, j - 1] = 1
                                with hcl.if_(j == ny - 1):
                                    in_region[i, 0] = 1
                                with hcl.else_():
                                    in_region[i, j + 1] = 1

    s = hcl.create_schedule([V_new, V_prev, V_decomp, V_1, V_2, x1, x2, step, t_step, big_delta, in_region, done, counts],
                            graph_create)
    return hcl.build(s)


def get_big_delta(config, decomp_curr, corrected_prev, decomp_init, decomp_final):
    '''
        Same as CorrectionBasedOnLocalUpdate.getBigDelta in numpy_impl/local_update_2d.py.
            0: constant config._big_delta
            1: max |final - initial decomposition result|
            2: max |decomposition at this step - corrected result at the previous step|
    '''
    if config._generate_big_delta == 0:
        return config._big_delta
    elif config._generate_big_delta == 1:
        return np.max(abs(decomp_final - decomp_init))
    elif config._generate_big_delta == 2:
        return np.max(abs(decomp_curr - corrected_prev))
    raise ValueError(f"unknown generate_big_delta: {config._generate_big_delta}")


def local_update_hcl(decomposition_result: DecompositionResult, config, cache_dir="kernels", debug=False):
    '''
        Used by main. Same local update as CorrectionBasedOnLocalUpdate in numpy_impl/local_update_2d.py,
        the update of each time step runs in a HeteroCL kernel.
            decomposition_result: from compute/decomposition_2d.py, arrays have time as the first axis
            config: from compute/config_2d.initConfig
        Returns (corrected value function with time as the first axis, points identified, points corrected).
        decomposition_result is not changed.
    '''
    # Step 1: inputs, time as the first axis
    result_decomp = np.asarray(decomposition_result.combined(), dtype=np.float64)
    result_1 = np.asarray(decomposition_result.subsystem1(), dtype=np.float64)
    result_2 = np.asarray(decomposition_result.subsystem2(), dtype=np.float64)
    num_steps = result_decomp.shape[0]
    grid = config.grid_2D()
    sys = config._sys_2d
    threshold = config._threshold2

    # Step 2: build (or load the cached) HeteroCL kernel and copy all inputs to HeteroCL arrays once
    hcl.init(DTYPE)
    total_start = time.time()
    key = (tuple(grid.pts_each_dim), tuple(grid.min), tuple(grid.max), tuple(grid.pDim), num_steps, threshold,
           type(sys).__name__, sorted(vars(sys).items()), inspect.getsource(type(sys)),
           inspect.getsource(build_local_update_kernel), inspect.getsource(spatial_deriv), inspect.getsource(hj_update))
    kernel, how = load_or_build_kernel(lambda: build_local_update_kernel(sys, grid, num_steps, threshold),
                                       key, "local_update_2d", cache_dir)
    print(f"HeteroCL kernel {how} in {time.time() - total_start:.4f} seconds")

    x1 = hcl.asarray(np.reshape(grid.vs[0], grid.pts_each_dim[0]).astype(np.float64), dtype=DTYPE)
    x2 = hcl.asarray(np.reshape(grid.vs[1], grid.pts_each_dim[1]).astype(np.float64), dtype=DTYPE)
    t_step = hcl.asarray(np.array([config._time_step], dtype=np.float64), dtype=DTYPE)
    V_decomp = hcl.asarray(np.ascontiguousarray(result_decomp), dtype=DTYPE)
    V_1 = hcl.asarray(np.ascontiguousarray(result_1), dtype=DTYPE)
    V_2 = hcl.asarray(np.ascontiguousarray(result_2), dtype=DTYPE)
    steps = [hcl.asarray(np.array([i], dtype=np.int32), dtype=hcl.Int()) for i in range(num_steps)]
    shape = result_decomp.shape[1:]
    in_region = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
    done = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
    counts = hcl.asarray(np.zeros(2, dtype=np.int32), dtype=hcl.Int())
    # Two value buffers, swapped after each time step
    V_prev = hcl.asarray(result_decomp[0], dtype=DTYPE)
    V_new = hcl.asarray(np.zeros(shape, dtype=np.float64), dtype=DTYPE)
    result_local = result_decomp.copy()

    # Step 3: local updating loop over time steps
    if debug:
        print("iteration, num_of_indices_to_correct, total_indices_corrected, total_time")
    total_identified = 0
    total_corrected = 0
    start = time.time()

    for i in range(1, num_steps):
        step_start = time.time()
        # Step 3a: threshold big delta in Algorithm 1, from the corrected result of the previous step
        big_delta_value = get_big_delta(config, result_decomp[i], result_local[i - 1], result_decomp[0], result_decomp[-1])
        big_delta = hcl.asarray(np.array([big_delta_value], dtype=np.float64), dtype=DTYPE)

        # Step 3b: identify, update and grow the region, all in the kernel
        kernel(V_new, V_prev, V_decomp, V_1, V_2, x1, x2, steps[i], t_step, big_delta, in_region, done, counts)

        # Step 3c: save the corrected value function of this time step, then swap the buffers
        result_local[i] = V_new.asnumpy()
        V_prev, V_new = V_new, V_prev

        num_identified, num_corrected = (int(c) for c in counts.asnumpy())
        total_identified += num_identified
        total_corrected += num_corrected
        if debug:
            print(f"{i}, {num_identified}, {num_corrected}, {round(time.time() - step_start, ndigits=4)}")

    print(f"Total correction time: {round(time.time() - start, ndigits=4)} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")

    return result_local, total_identified, total_corrected


def main():
    '''
        Read the saved 2d results (compute/direct_2d.py, compute/decomposition_2d.py), run the HeteroCL local
        update, save it to results/result_local_hcl_2d_<combine>.npy (shape (x, y, t)) and compare it with the
        direct computation if --true_file is given.
    '''
    # Step 1: inputs, from the command line and/or the --params file
    parser = argparse.ArgumentParser()
    add_local_update_args_2d(parser)
    parser.add_argument("--cache_dir", default="kernels",
                        help="Directory of the compiled HeteroCL kernels. A kernel built before with the same "
                        "settings is loaded from here instead of being rebuilt.")
    args = parse_args_with_params(parser)
    check_combine(args.combine, ("union", "intersection"), "2d")
    config, decomposition_result, result_true = setup_local_update_2d(parser, args)

    # Step 2: HeteroCL local update
    result_local, total_identified, total_corrected = local_update_hcl(
        decomposition_result, config, cache_dir=args.cache_dir, debug=bool(args.debug))
    print(f"Total points identified: {total_identified}")
    print(f"Total points corrected: {total_corrected}")

    # Step 3: save the corrected result, time as the last axis
    os.makedirs("results", exist_ok=True)
    path = f"results/result_local_hcl_2d_{args.combine}.npy"
    np.save(path, np.moveaxis(result_local, 0, -1))
    print(f"Saved {path}")

    # Step 4: compare with the direct computation (if given), same printout as the numpy version
    print("")
    if result_true is not None:
        corrected_result = copy.copy(decomposition_result)
        corrected_result._combined_results = result_local
        printCorrectnessStatistics2d(
            direct_computation_results=result_true,
            decomposition_results=decomposition_result,
            corrected_decomposition_results=corrected_result,
        )
    else:
        print("No --true_file given, skipping the comparison with the direct computation")


if __name__ == "__main__":
    main()
