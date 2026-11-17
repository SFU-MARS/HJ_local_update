# Decomposition of the 4D Dubins car into two 3D subsystems solved with HJSolver,
# combined as V = min(V1, V2) (union) or max(V1, V2) (intersection).
# Saves results/result_decomp_4d_<combine>.npy, shape (t, x, y, v, theta) (time as the first axis, index 0 is t = 0),
# and the subsystem initial values results/init_1_4d.npy, results/init_2_4d.npy, shape (x, y, v, theta).
import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *
from odp.solver import HJSolver

from compute.hj_solver_wrapper import solve_hj

from odp.Plots import PlotOptions
from odp.Plots import visualize_plots

from subdynamics.dubins_4d import subsys_1, subsys_2

import time
import os

from common.combine import check_combine


def decomposition(num, num_v=21, saveAllTimeStep=True, lookback_length=0.5, t_step=0.02,
                  obstacle_min=(-0.5, -0.5), obstacle_max=(0.5, 0.5), accuracy="low", plot=True,
                  combine="union", aMax=1, wMax=1, uMode="max", results_dir="results"):
    r'''
        Decompose the 4D Dubins car with speed limits (v stays in the grid's v range [0, 2])
            \dot{x} = v*cos(theta), \dot{y} = v*sin(theta), \dot{v} = a, \dot{theta} = w
        into
            subsystem 1: (x, v, theta)
            subsystem 2: (y, v, theta)
        Each subsystem is solved with the optimized_dp HJSolver, and the 4D value function is
        reconstructed from V1(x, v, theta) and V2(y, v, theta), with the rectangles x_min <= x <= x_max and
        y_min <= y <= y_max (v and theta unbounded):
            combine="union": V = min(V1, V2), the union of the rectangles
            combine="intersection": V = max(V1, V2), the intersection of the rectangles (a square)
            num, num_v: grid points of x, y and theta, and of v
            aMax, wMax: acceleration and turn rate bounds, |a| <= aMax, |w| <= wMax
            uMode: "max" (control avoids the set) or "min" (control reaches it)
        Saves <results_dir>/result_decomp_4d_<combine>.npy and <results_dir>/init_1_4d.npy, init_2_4d.npy
        (the initial values do not depend on combine).
    '''
    check_combine(combine, ("union", "intersection"), "4d")

    # Create 3D grids for the subsystems (x or y, v, theta), theta is periodic
    g_1 = Grid(np.array([-4.0, 0.0, -math.pi]), np.array([4.0, 2.0, math.pi]), 3, np.array([num, num_v, num]), [2])
    g_2 = Grid(np.array([-4.0, 0.0, -math.pi]), np.array([4.0, 2.0, math.pi]), 3, np.array([num, num_v, num]), [2])

    # Initialize value function (negative inside the set)
    data_sub_1 = ShapeRectangle(g_1, np.array([obstacle_min[0], -np.inf, -np.inf]), np.array([obstacle_max[0], np.inf, np.inf]))
    data_sub_2 = ShapeRectangle(g_2, np.array([obstacle_min[1], -np.inf, -np.inf]), np.array([obstacle_max[1], np.inf, np.inf]))

    # Save the subsystem initial values on the 4D grid (x, y, v, theta), used for the leaking set
    # in the local update: init_1 = V1(x, v, theta), init_2 = V2(y, v, theta)
    os.makedirs(results_dir, exist_ok=True)
    np.save(f'{results_dir}/init_1_4d.npy', np.broadcast_to(data_sub_1[:, np.newaxis, :, :], (num, num, num_v, num)).astype(np.float32))
    np.save(f'{results_dir}/init_2_4d.npy', np.broadcast_to(data_sub_2[np.newaxis, :, :, :], (num, num, num_v, num)).astype(np.float32))

    ## Look-back length and time step of computation
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics: uMode max avoids the set, uMode min reaches it; lowest speed = lower end of the v grid
    sys_1 = subsys_1(aMax=aMax, wMax=wMax, vMin=0.0, vMax=2.0, dMax=0.0, uMode=uMode, dMode='min')
    sys_2 = subsys_2(aMax=aMax, wMax=wMax, vMin=0.0, vMax=2.0, dMax=0.0, uMode=uMode, dMode='min')

    # Backward reachable tube of the obstacle
    compMethods = {"TargetSetMode": "None"}

    start = time.time()

    '''
    Computation in subsystem 1
    '''
    if saveAllTimeStep:
        # Index 0 is t = 0, and one HJSolver step per time step (checked)
        result_1 = solve_hj(sys_1, g_1, data_sub_1, tau, accuracy=accuracy)
    else:
        result_1 = HJSolver(sys_1, g_1, data_sub_1, tau, compMethods, saveAllTimeSteps=False, accuracy=accuracy)

    '''
    Computation in subsystem 2
    '''
    if saveAllTimeStep:
        # Index 0 is t = 0, and one HJSolver step per time step (checked)
        result_2 = solve_hj(sys_2, g_2, data_sub_2, tau, accuracy=accuracy)
    else:
        result_2 = HJSolver(sys_2, g_2, data_sub_2, tau, compMethods, saveAllTimeSteps=False, accuracy=accuracy)

    '''
    Combine the results from 2 subsystems
    '''
    grid_min = np.array([-4.0, -4.0, 0.0, -math.pi])
    grid_max = np.array([4.0, 4.0, 2.0, math.pi])
    dims = grid_min.shape[0]
    N = np.array([num, num, num_v, num])
    g = Grid(grid_min, grid_max, dims, N, periodicDims=[3])

    if not saveAllTimeStep:
        # result_1: (x, v, theta), result_2: (y, v, theta) -> (x, y, v, theta)
        result_upper = result_1[:, np.newaxis, :, :]
        result_lower = result_2[np.newaxis, :, :, :]
    else:
        # solve_hj already put t = 0 at index 0
        data_list_1 = result_1
        data_list_2 = result_2

        # (x, v, theta, t), (y, v, theta, t) -> (x, y, v, theta, t)
        result_upper = data_list_1[:, np.newaxis, :, :, :]
        result_lower = data_list_2[np.newaxis, :, :, :, :]

    result_full = np.minimum(result_upper, result_lower) if combine == "union" else np.maximum(result_upper, result_lower)
    result_upper = np.broadcast_to(result_upper, result_full.shape)
    result_lower = np.broadcast_to(result_lower, result_full.shape)

    execution_time = time.time() - start

    print("Total kernel time decomposition: ", execution_time)
    print("Finished updating the value function")

    # Saved with time as the first axis, as the local updates read it
    np.save(f'{results_dir}/result_decomp_4d_{combine}.npy', np.ascontiguousarray(np.moveaxis(result_full, -1, 0)))
    for name in (f"result_decomp_4d_{combine}.npy", "init_1_4d.npy", "init_2_4d.npy"):
        print(f"Saved {results_dir}/{name}")

    # Visualization of the (x, y, theta) slice at the middle v
    if plot:
        os.makedirs('plots', exist_ok=True)
        po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,3], slicesCut=[num_v // 2], colorscale="Bluered",
                        save_fig=True, filename=f"plots/decomposition_result_4d_{combine}", interactive_html=True)
        visualize_plots(result_full, g, po)

    return result_full, result_upper, result_lower, g


def main():
    '''
        Command-line version, same arguments as direct_4d.py: --dynamics_params takes aMax, wMax and uMode.
    '''
    import argparse
    from common.params import json_object
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=41, help="Grid points of x, y and theta (default 41).")
    parser.add_argument("--grid_size_v", type=int, default=21, help="Grid points of v (default 21).")
    parser.add_argument("--combine", choices=["union", "intersection"], default="union",
                        help="union = min(V1, V2), intersection = max(V1, V2). The control mode is uMode in --dynamics_params.")
    parser.add_argument("--t_step", type=float, default=0.02, help="Time step (default 0.02, below HJSolver's CFL bound with 21 v points).")
    parser.add_argument("--lookback_length", type=float, default=0.5,
                        help="Look-back length, lookback_length / t_step time steps (default 0.5: 25 steps).")
    parser.add_argument("--results_dir", default="results", help="Directory of the saved results.")
    parser.add_argument("--dynamics_params", type=json_object, default={},
                        help="aMax, wMax and uMode as a JSON object, e.g. '{\"aMax\": 1, \"wMax\": 1, \"uMode\": \"min\"}' "
                        "(default 1, 1 and 'max'). uMode is the control mode, 'max' (avoid) or 'min' (reach).")
    parser.add_argument("--accuracy", default="low", help="HJSolver accuracy, 'low' matches the local update scheme.")
    parser.add_argument("--plot", action="store_true", help="Also save an interactive plot to plots/.")
    args = parser.parse_args()
    unknown = sorted(set(args.dynamics_params) - {"aMax", "wMax", "uMode"})
    if unknown:
        parser.error(f"--dynamics_params: unknown parameters {unknown}, it accepts ['aMax', 'wMax', 'uMode']")
    params = {"aMax": 1, "wMax": 1, "uMode": "max", **args.dynamics_params}
    if params["uMode"] not in ("min", "max"):
        parser.error(f"--dynamics_params: uMode must be 'min' or 'max', got {params['uMode']!r}")
    print(f"Subsystem dynamics: dubins_4d {params}")
    try:
        decomposition(args.grid_size, num_v=args.grid_size_v, saveAllTimeStep=True, lookback_length=args.lookback_length, t_step=args.t_step,
             accuracy=args.accuracy, plot=args.plot, combine=args.combine, aMax=params["aMax"], wMax=params["wMax"],
             uMode=params["uMode"], results_dir=args.results_dir)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))


if __name__ == "__main__":
    main()
