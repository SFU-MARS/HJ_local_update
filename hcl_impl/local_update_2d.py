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

from compute.config_2d import initConfig
from compute.decomposition_2d import decomposition, DecompositionResult
from compute.direct_2d import direct_computation
from dynamics.coupled_2d import couple_u
from common.combine import check_combine
from common.params import parse_args_with_params
from common.stats import printCorrectnessStatistics2d
from hcl_impl.kernel_cache import load_or_build_kernel
from common import spatial_deriv
from common.spatial_deriv import spa_deriv_hcl

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
        Same scheme as the numpy version: average of the one-sided first order derivatives, no dissipation,
        forward Euler, 64-bit floats.
            identified (output): 1 for every leaking corner found in step 1
            done (output): 1 for every point updated at this time step
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
    identified = hcl.placeholder(shape, name="identified", dtype=hcl.Int())
    done = hcl.placeholder(shape, name="done", dtype=hcl.Int())

    def graph_create(V_new, V_prev, V_decomp, V_1, V_2, x1, x2, step, t_step, big_delta, in_region, identified, done):
        s = step[0]
        # A plain Python float in a kernel becomes a 32-bit constant, so build it as a 64-bit one
        thr = tvm.const(threshold, "float64")

        # Step 1: start from the decomposition result, the region is the leaking corners
        with hcl.Stage("Identify"):
            with hcl.for_(0, nx, name="i") as i:
                with hcl.for_(0, ny, name="j") as j:
                    V_new[i, j] = V_decomp[s, i, j]
                    done[i, j] = 0
                    in_region[i, j] = 0
                    with hcl.if_(abs_same_type(V_1[s, i, j] - V_2[s, i, j]) < big_delta[0]):
                        in_region[i, j] = 1
                    identified[i, j] = in_region[i, j]

        # Step 2: update the region and grow it until no new point joins
        with hcl.Stage("Grow"):
            again = hcl.scalar(1, "again", dtype=hcl.Int())
            with hcl.while_(again[0] == 1):
                again[0] = 0
                with hcl.for_(0, nx, name="i") as i:
                    with hcl.for_(0, ny, name="j") as j:
                        with hcl.if_(hcl.and_(in_region[i, j] == 1, done[i, j] == 0)):
                            dV_dx_L, dV_dx_R = spa_deriv_hcl([i, j], V_prev, grid, 0, boundary="one_sided")
                            dV_dy_L, dV_dy_R = spa_deriv_hcl([i, j], V_prev, grid, 1, boundary="one_sided")
                            dV_dx = hcl.scalar(0, "dV_dx")
                            dV_dy = hcl.scalar(0, "dV_dy")
                            dV_dx[0] = (dV_dx_L + dV_dx_R) / 2
                            dV_dy[0] = (dV_dy_L + dV_dy_R) / 2

                            # Optimal control and dynamical rates of change (the dynamics do not depend on t)
                            state = (x1[i], x2[j])
                            uOpt = sys.opt_ctrl(0, state, (dV_dx[0], dV_dy[0]))
                            dx_dt, dy_dt = sys.dynamics(0, state, uOpt, (0, 0))

                            # Forward Euler step
                            V_new[i, j] = V_prev[i, j] + (dx_dt * dV_dx[0] + dy_dt * dV_dy[0]) * t_step[0]
                            done[i, j] = 1
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

    s = hcl.create_schedule([V_new, V_prev, V_decomp, V_1, V_2, x1, x2, step, t_step, big_delta, in_region, identified, done],
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
           type(sys).__name__, sys.uMax, sys.uMode, inspect.getsource(couple_u),
           inspect.getsource(build_local_update_kernel), inspect.getsource(spatial_deriv))
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
    identified = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
    done = hcl.asarray(np.zeros(shape, dtype=np.int32), dtype=hcl.Int())
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
        kernel(V_new, V_prev, V_decomp, V_1, V_2, x1, x2, steps[i], t_step, big_delta, in_region, identified, done)

        # Step 3c: save the corrected value function of this time step, then swap the buffers
        result_local[i] = V_new.asnumpy()
        V_prev, V_new = V_new, V_prev

        num_identified = int(identified.asnumpy().sum())
        num_corrected = int(done.asnumpy().sum())
        total_identified += num_identified
        total_corrected += num_corrected
        if debug:
            print(f"{i}, {num_identified}, {num_corrected}, {round(time.time() - step_start, ndigits=4)}")

    print(f"Total correction time: {round(time.time() - start, ndigits=4)} seconds")
    print(f"Total local update time (kernel build/load + input copy + correction): {time.time() - total_start:.4f} seconds")

    return result_local, total_identified, total_corrected


def main():
    # Step 1: parse the inputs (from the command line and/or the --params file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="JSON parameter file (see hcl_impl/params/), its keys are the argument names below "
                        "without '--'. Arguments given on the command line override the file.", default=None)
    parser.add_argument("--combine", choices=["union", "intersection"], default="intersection",
                        help="How the subsystems are combined: 'union' = min(V1, V2) with uMode='max' (avoid), "
                        "'intersection' = max(V1, V2) with uMode='min' (reach). Default intersection.")
    parser.add_argument("--grid_size", type=int, default=101,
                        help="Size of the 2D grid. For example, use --grid_size 101 for a 101x101 grid.")
    parser.add_argument("--generate_big_delta", type=int, choices=[0, 1, 2], default=2,
                        help="0: constant --big_delta. 1: from the initial and final decomposition result. "
                        "2: a new big_delta at each time step (as described in the paper).")
    parser.add_argument("--big_delta", type=float, default=0.2,
                        help="big_delta used with --generate_big_delta 0.")
    parser.add_argument("--debug", type=int, choices=[0, 1], default=0, help="1 prints the stats of each time step.")
    parser.add_argument("--cache_dir", default="kernels",
                        help="Directory of the compiled HeteroCL kernels. A kernel built before with the same "
                        "settings is loaded from here instead of being rebuilt.")
    args = parse_args_with_params(parser)
    check_combine(args.combine, ("union", "intersection"), "2d")
    print(f"combine: {args.combine}")
    print(f"generate_big_delta: {args.generate_big_delta}")
    print(f"Grid size: {args.grid_size} x {args.grid_size}")

    # Step 2: direct computation and decomposition (numpy, same as the numpy version)
    config = initConfig(use_union=args.combine == "union", grid_size=args.grid_size,
                        generate_big_delta=args.generate_big_delta, big_delta=args.big_delta)
    print(f"Number of time_steps: {int(config._lookback_length/config._time_step)}")
    print("")
    grid, result_true = direct_computation(config=config, saveAllTimeStep=True)
    print("")
    decomposition_result = decomposition(config=config, saveAllTimeStep=True)

    # Step 3: HeteroCL local update
    result_local, total_identified, total_corrected = local_update_hcl(
        decomposition_result, config, cache_dir=args.cache_dir, debug=bool(args.debug))
    print(f"Total points identified: {total_identified}")
    print(f"Total points corrected: {total_corrected}")

    # Step 4: compare with the direct computation, same printout as the numpy version
    corrected_result = copy.copy(decomposition_result)
    corrected_result._combined_results = result_local
    print("")
    printCorrectnessStatistics2d(
        direct_computation_results=result_true,
        decomposition_results=decomposition_result,
        corrected_decomposition_results=corrected_result,
    )


if __name__ == "__main__":
    main()
