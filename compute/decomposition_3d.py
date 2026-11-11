# Decomposition of the 3D Dubins car into two 2D subsystems solved with HJSolver,
# combined as V = min(V1, V2) (union) or max(V1, V2) (intersection).
# Saves results/result_decomp_3d_<combine>.npy, shape (t, x, y, theta) (time as the first axis, index 0 is t = 0),
# and the subsystem initial values results/init_1_3d.npy, results/init_2_3d.npy, shape (x, y, theta).
import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *
from odp.solver import HJSolver

from compute.hj_solver import solve_hj

from odp.Plots import PlotOptions
from odp.Plots import visualize_plots

from subdynamics.dubins_3d import subsys_1, subsys_2

import time
import os

from common.combine import check_combine
from dynamics import U_MODE


def decomposition(num, saveAllTimeStep=True, lookback_length=1.0, t_step=0.04,
                  obstacle_min=(-0.5, -0.5), obstacle_max=(0.5, 0.5), accuracy="low", plot=True,
                  combine="union", v=1, wMax=1, results_dir="results"):
    r'''
        Decompose the 3D Dubins car
            \dot{x} = v*cos(theta), \dot{y} = v*sin(theta), \dot{theta} = w
        into
            subsystem 1: (x, theta)
            subsystem 2: (y, theta)
        Each subsystem is solved with the optimized_dp HJSolver, and the 3D value function is
        reconstructed from V1(x, theta) and V2(y, theta), with the rectangles x_min <= x <= x_max and
        y_min <= y <= y_max (theta unbounded):
            combine="union": V = min(V1, V2), the obstacle is the union of the rectangles, control
                avoids it (uMode='max')
            combine="intersection": V = max(V1, V2), the target is the intersection of the rectangles
                (a square), control reaches it (uMode='min')
            v, wMax: speed and turn rate bound of the Dubins car
        Saves <results_dir>/result_decomp_3d_<combine>.npy and <results_dir>/init_1_3d.npy, init_2_3d.npy
        (the initial values do not depend on combine).
    '''
    check_combine(combine, ("union", "intersection"), "3d")

    # Create 2D grids for the subsystems, theta is periodic
    g_1 = Grid(np.array([-4.0, -math.pi]), np.array([4.0, math.pi]), 2, np.array([num, num]), [1])
    g_2 = Grid(np.array([-4.0, -math.pi]), np.array([4.0, math.pi]), 2, np.array([num, num]), [1])

    # Initialize value function (negative inside the set)
    data_sub_1 = ShapeRectangle(g_1, np.array([obstacle_min[0], -np.inf]), np.array([obstacle_max[0], np.inf]))
    data_sub_2 = ShapeRectangle(g_2, np.array([obstacle_min[1], -np.inf]), np.array([obstacle_max[1], np.inf]))

    # Save the subsystem initial values on the 3D grid (x, y, theta), used for the leaking set
    # in the local update: init_1 = V1(x, theta), init_2 = V2(y, theta)
    os.makedirs(results_dir, exist_ok=True)
    np.save(f'{results_dir}/init_1_3d.npy', np.broadcast_to(data_sub_1[:, np.newaxis, :], (num, num, num)).astype(np.float32))
    np.save(f'{results_dir}/init_2_3d.npy', np.broadcast_to(data_sub_2[np.newaxis, :, :], (num, num, num)).astype(np.float32))

    ## Look-back length and time step of computation
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics: union avoids the obstacle (uMode max), intersection reaches the target (uMode min)
    sys_1 = subsys_1(v=v, wMax=wMax, dMax=0.0, uMode=U_MODE[combine], dMode='min')
    sys_2 = subsys_2(v=v, wMax=wMax, dMax=0.0, uMode=U_MODE[combine], dMode='min')

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
    grid_min = np.array([-4.0, -4.0, -math.pi])
    grid_max = np.array([4.0, 4.0, math.pi])
    dims = grid_min.shape[0]
    N = np.array([num, num, num])
    g = Grid(grid_min, grid_max, dims, N, periodicDims=[2])

    if not saveAllTimeStep:
        # result_1: (x, theta), result_2: (y, theta) -> (x, y, theta)
        result_upper = result_1[:, np.newaxis, :]
        result_lower = result_2[np.newaxis, :, :]
    else:
        # solve_hj already put t = 0 at index 0
        data_list_1 = result_1
        data_list_2 = result_2

        # (x, theta, t), (y, theta, t) -> (x, y, theta, t)
        result_upper = data_list_1[:, np.newaxis, :, :]
        result_lower = data_list_2[np.newaxis, :, :, :]

    result_full = np.minimum(result_upper, result_lower) if combine == "union" else np.maximum(result_upper, result_lower)
    result_upper = np.broadcast_to(result_upper, result_full.shape)
    result_lower = np.broadcast_to(result_lower, result_full.shape)

    execution_time = time.time() - start

    print("Total kernel time decomposition: ", execution_time)
    print("Finished updating the value function")

    # Saved with time as the first axis, as the local updates read it
    np.save(f'{results_dir}/result_decomp_3d_{combine}.npy', np.ascontiguousarray(np.moveaxis(result_full, -1, 0)))
    for name in (f"result_decomp_3d_{combine}.npy", "init_1_3d.npy", "init_2_3d.npy"):
        print(f"Saved {results_dir}/{name}")

    # Visualization of 3D value function
    if plot:
        os.makedirs('plots', exist_ok=True)
        po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2], slicesCut=[],colorscale="Bluered",
                        save_fig=True, filename=f"plots/decomposition_result_3d_{combine}", interactive_html=True)
        visualize_plots(result_full, g, po)

    return result_full, result_upper, result_lower, g


def main():
    '''
        Command-line version, same arguments as the 2d and 6d compute scripts (common/params.add_compute_args).
    '''
    import argparse
    from common.params import add_compute_args
    from dynamics import make_dynamics
    parser = argparse.ArgumentParser()
    add_compute_args(parser, grid_size=41, combine="union", t_step=0.04, lookback_length=1.0, dynamics="dubins_3d")
    parser.add_argument("--accuracy", default="low", help="HJSolver accuracy, 'low' matches the local update scheme.")
    parser.add_argument("--plot", action="store_true", help="Also save an interactive plot to plots/.")
    args = parser.parse_args()
    try:
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=3)
    except (ValueError, TypeError) as e:
        parser.error(f"--dynamics {args.dynamics} --dynamics_params {args.dynamics_params}: {e}")
    print(f"Dynamics: {args.dynamics} {vars(sys)}")
    try:
        decomposition(args.grid_size, saveAllTimeStep=True, lookback_length=args.lookback_length, t_step=args.t_step,
             accuracy=args.accuracy, plot=args.plot, combine=args.combine, v=sys.v, wMax=sys.wMax,
             results_dir=args.results_dir)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))


if __name__ == "__main__":
    main()
