# Direct computation of the full 4D Dubins car with HJSolver, used as the ground truth.
# Saves results/result_direct_4d_<combine>.npy, shape (x, y, v, theta, t), index 0 of the last axis is t = 0.
import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *
from odp.solver import HJSolver

from compute.hj_solver_wrapper import solve_hj
from odp.dynamics import DubinsCar4D

from odp.Plots import PlotOptions
from odp.Plots import visualize_plots

import time
import os

from common.combine import check_combine


def direct_comp(num, saveAllTimeStep=True, lookback_length=1.0, t_step=0.01,
                obstacle_min=(-0.5, -0.5), obstacle_max=(0.5, 0.5), accuracy="low", plot=True,
                combine="union", aMax=1, wMax=1, uMode="max", results_dir="results"):
    r'''
        Directly solve the 4D Dubins car (DubinsCar4D in optimized_dp, no disturbance)
            \dot{x} = v*cos(theta), \dot{y} = v*sin(theta), \dot{v} = a, \dot{theta} = w
        with the optimized_dp HJSolver. The set is the same as in direct_3d.py, built from the
        rectangles x_min <= x <= x_max and y_min <= y <= y_max (v and theta unbounded):
            combine="union": union of the rectangles, V = min(V1, V2)
            combine="intersection": intersection of the rectangles (a square), V = max(V1, V2)
            aMax, wMax: acceleration and turn rate bounds, |a| <= aMax, |w| <= wMax
            uMode: "max" (control avoids the set) or "min" (control reaches it)
        Saves <results_dir>/result_direct_4d_<combine>.npy.
    '''
    check_combine(combine, ("union", "intersection"), "4d")

    # Create 4D grid (x, y, v, theta), theta is periodic
    grid_min = np.array([-4.0, -4.0, 0.0, -math.pi])
    grid_max = np.array([4.0, 4.0, 2.0, math.pi])
    dims = grid_min.shape[0]
    N = np.array([num, num, num, num])
    g = Grid(grid_min, grid_max, dims, N, periodicDims=[3])

    # Initialize value function (negative inside the set)
    data_1 = ShapeRectangle(g, np.array([obstacle_min[0], -np.inf, -np.inf, -np.inf]),
                            np.array([obstacle_max[0], np.inf, np.inf, np.inf]))
    data_2 = ShapeRectangle(g, np.array([-np.inf, obstacle_min[1], -np.inf, -np.inf]),
                            np.array([np.inf, obstacle_max[1], np.inf, np.inf]))
    data = np.minimum(data_1, data_2) if combine == "union" else np.maximum(data_1, data_2)

    ## Look-back length and time step of computation
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics: uMode max avoids the set, uMode min reaches it.
    # DubinsCar4D needs dMode opposite to uMode; the disturbance bounds are zero.
    sys = DubinsCar4D(uMin=[-aMax, -wMax], uMax=[aMax, wMax], dMin=[0.0, 0.0], dMax=[0.0, 0.0],
                      uMode=uMode, dMode="min" if uMode == "max" else "max")

    # Same computation method as direct_3d.py
    compMethods = {"TargetSetMode": "None"}

    start = time.time()

    if saveAllTimeStep:
        # Index 0 is t = 0, and one HJSolver step per time step (checked)
        result = solve_hj(sys, g, data, tau, accuracy=accuracy)
    else:
        result = HJSolver(sys, g, data, tau, compMethods, saveAllTimeSteps=False, accuracy=accuracy)

    execution_time = time.time() - start

    print("Total kernel time direct computation: ", execution_time)
    print("Finished updating the value function")

    os.makedirs(results_dir, exist_ok=True)
    np.save(f'{results_dir}/result_direct_4d_{combine}.npy', result)
    print(f"Saved {results_dir}/result_direct_4d_{combine}.npy")

    # Visualization of the (x, y, theta) slice at the middle v
    if plot:
        os.makedirs('plots', exist_ok=True)
        po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,3], slicesCut=[num // 2], colorscale="Bluered",
                        save_fig=True, filename=f"plots/direct_result_4d_{combine}", interactive_html=True)
        visualize_plots(result, g, po)

    return g, result


def main():
    '''
        Command-line version, same arguments as direct_3d.py except --dynamics (always DubinsCar4D):
        --dynamics_params takes aMax, wMax and uMode.
    '''
    import argparse
    from common.params import json_object
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=41, help="Grid points per dimension (default 41).")
    parser.add_argument("--combine", choices=["union", "intersection"], default="union",
                        help="union = min(V1, V2), intersection = max(V1, V2). The control mode is uMode in --dynamics_params.")
    parser.add_argument("--t_step", type=float, default=0.01, help="Time step.")
    parser.add_argument("--lookback_length", type=float, default=1.0,
                        help="Look-back length, lookback_length / t_step time steps.")
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
    print(f"Dynamics: DubinsCar4D {params}")
    try:
        direct_comp(args.grid_size, saveAllTimeStep=True, lookback_length=args.lookback_length, t_step=args.t_step,
             accuracy=args.accuracy, plot=args.plot, combine=args.combine, aMax=params["aMax"], wMax=params["wMax"],
             uMode=params["uMode"], results_dir=args.results_dir)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))


if __name__ == "__main__":
    main()
