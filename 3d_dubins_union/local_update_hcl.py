import argparse
import math
import os
import time

import heterocl as hcl
import numpy as np

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle
from odp.dynamics import DubinsCar
from odp.computeGraphs.CustomGraphFunctions import my_abs, my_max
from odp.spatialDerivatives.firstOrderENO.first_orderENO3D import spa_derivX, spa_derivY, spa_derivT

from local_update_numpy import dilate, printCorrectnessStatistics


def build_local_update_kernel(sys, grid):
    '''
        HeteroCL kernel of one local update time step:
            V_new = V_prev + t_step * (H + dissipation)   inside the update region (mask == 1)
            V_new = V_decomp                              outside the update region
        Same first order scheme as local_update_numpy.py (1st order ENO + Lax-Friedrichs + forward Euler),
        the loop over x is parallelized across threads.
    '''
    shape = tuple(grid.pts_each_dim)
    V_new = hcl.placeholder(shape, name="V_new", dtype=hcl.Float())
    V_prev = hcl.placeholder(shape, name="V_prev", dtype=hcl.Float())
    V_decomp = hcl.placeholder(shape, name="V_decomp", dtype=hcl.Float())
    mask = hcl.placeholder(shape, name="mask", dtype=hcl.Int())
    x1 = hcl.placeholder((shape[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((shape[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((shape[2],), name="x3", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())
    t_step = hcl.placeholder((1,), name="t_step", dtype=hcl.Float())

    def graph_create(V_new, V_prev, V_decomp, mask, x1, x2, x3, t, t_step):
        with hcl.Stage("LocalUpdate"):
            with hcl.for_(0, shape[0], name="i") as i:
                with hcl.for_(0, shape[1], name="j") as j:
                    with hcl.for_(0, shape[2], name="k") as k:
                        with hcl.if_(mask[i, j, k] == 1):
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

                            # Optimal control and dynamical rates of change
                            state = (x1[i], x2[j], x3[k])
                            uOpt = sys.opt_ctrl(t, state, (dV_dx[0], dV_dy[0], dV_dT[0]))
                            dx_dt, dy_dt, dT_dt = sys.dynamics(t, state, uOpt, (0, 0, 0))

                            # Lax-Friedrichs dissipation, alpha is the max rate of change over the control set
                            r1_x, r1_y, r1_T = sys.dynamics(t, state, (sys.wMax, 0, 0), (0, 0, 0))
                            r2_x, r2_y, r2_T = sys.dynamics(t, state, (-sys.wMax, 0, 0), (0, 0, 0))
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
                            V_new[i, j, k] = V_decomp[i, j, k]

    s = hcl.create_schedule([V_new, V_prev, V_decomp, mask, x1, x2, x3, t, t_step], graph_create)
    # Thread parallelize the loop over x
    s_U = graph_create.LocalUpdate
    s[s_U].parallel(s_U.i)
    return hcl.build(s)


def local_update_hcl_from_file(decomp_file, data_init_upper, data_init_lower, true_file, grid, t_step,
                               delta=None, threshold=1e-6):
    '''
        Same growing-region local update as local_update_from_file in local_update_numpy.py,
        the update of each time step runs in a parallel HeteroCL kernel.
    '''
    # Step 1: load the decomposition and direct results, build the full initial value function
    result_decomp = np.load(decomp_file).astype(np.float32)
    result_true = np.load(true_file)
    data_init = np.minimum(data_init_upper, data_init_lower).astype(np.float32)

    # Step 2: find the leaking corners, where the two subsystem values are close (|V1 - V2| < delta)
    if delta is None:
        delta = np.max(abs(data_init - result_decomp[..., -1]))
    print('The delta value is', delta)
    leaking = abs(data_init_upper - data_init_lower) < delta
    print(f"The number of leaking corner points: {leaking.sum()} / {data_init.size}")

    # Step 3: build the HeteroCL kernel and its inputs
    hcl.init(hcl.Float(32))
    sys = DubinsCar(wMax=1, speed=1, uMode="max", dMode="min")
    start = time.time()
    kernel = build_local_update_kernel(sys, grid)
    print(f"HeteroCL build time: {time.time() - start:.4f} seconds")

    x1 = hcl.asarray(np.reshape(grid.vs[0], grid.pts_each_dim[0]))
    x2 = hcl.asarray(np.reshape(grid.vs[1], grid.pts_each_dim[1]))
    x3 = hcl.asarray(np.reshape(grid.vs[2], grid.pts_each_dim[2]))
    t_step_hcl = hcl.asarray(np.array([t_step], dtype=np.float32))
    V_new = hcl.asarray(np.zeros(data_init.shape, dtype=np.float32))

    # Step 4: local updating loop over time steps
    result_local = result_decomp.copy()
    data = data_init
    total_updated = 0

    start = time.time()

    for i in range(1, result_decomp.shape[-1]):
        t = hcl.asarray(np.array([(i-1)*t_step, i*t_step], dtype=np.float32))

        # Step 4a: update region = leaking corners + corrected points at the previous step and their neighbours
        corrected = abs(data - result_decomp[..., i-1]) > threshold
        region = dilate(corrected, grid) | leaking

        # Step 4b: run the parallel kernel, decomposition result outside the region
        kernel(V_new, hcl.asarray(data), hcl.asarray(np.ascontiguousarray(result_decomp[..., i])),
               hcl.asarray(region.astype(np.int32), dtype=hcl.Int()), x1, x2, x3, t, t_step_hcl)
        data = V_new.asnumpy()

        # Step 4c: save the corrected value function of this time step
        result_local[..., i] = data
        total_updated += int(region.sum())

    execution_time = time.time() - start
    print(f"Total correction time: {execution_time:.4f} seconds")
    print(f"Total point updates: {total_updated} / {data_init.size * (result_decomp.shape[-1] - 1)}")

    # Step 5: compare the decomposition and local update results with the direct computation
    printCorrectnessStatistics(result_true, result_decomp, result_local)

    return result_local


def main():
    # Step 1: parse and check the command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp_file", help="npy file of the decomposition result, shape (x, y, theta, t).", required=True)
    parser.add_argument("--true_file", help="npy file of the direct computation result, same shape as --decomp_file.", required=True)
    parser.add_argument("--t_step", help="Time step used to compute both files.", default=0.04, type=float)
    args = parser.parse_args()
    for path in (args.decomp_file, args.true_file):
        if not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    decomp_shape = np.load(args.decomp_file, mmap_mode='r').shape
    true_shape = np.load(args.true_file, mmap_mode='r').shape
    if decomp_shape != true_shape:
        parser.error(f"shape mismatch: {args.decomp_file} {decomp_shape} vs {args.true_file} {true_shape}, "
                     "generate both files with the same t_step and accuracy (see README)")

    # Step 2: build the 3D grid with the grid size read from the decomposition file, theta is periodic
    num = np.load(args.decomp_file, mmap_mode='r').shape[0]
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([num, num, num]), [2])

    # Step 3: initial value function of the subsystems, same obstacle as decomposition.py
    data_init_upper = ShapeRectangle(grid, np.array([-0.5, -np.inf, -np.inf]), np.array([0.5, np.inf, np.inf]))
    data_init_lower = ShapeRectangle(grid, np.array([-np.inf, -0.5, -np.inf]), np.array([np.inf, 0.5, np.inf]))

    # Step 4: run the local update and save the corrected result
    result_local = local_update_hcl_from_file(args.decomp_file, data_init_upper, data_init_lower, args.true_file,
                                              grid, args.t_step)
    np.save('results/result_local_hcl_3d.npy', result_local)


if __name__ == "__main__":
    if os.path.exists("results") == False:
        os.mkdir("results")
    main()
