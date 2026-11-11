# Direct computation of the 6D quadrotor with optimized_dp's HJSolver, used as the ground truth.
import argparse
import os
import time

import numpy as np

from compute.config_6d import add_compute_args_6d, grid_6d, init_values_6d, result_files_6d
from compute.hj_solver import solve_hj
from common.combine import check_combine
from dynamics import U_MODE
from dynamics.quadrotor_6d import quadrotor


def direct_comp(num, saveAllTimeStep=True, lookback_length=0.1, combine="union", t_step=0.01, sys=None):
    '''
        Solve the 6D quadrotor with HJSolver (accuracy "low", one step per time step).
            combine: "union" (obstacle x <= 0 or y <= 0, V = min, avoid) or "intersection" (target
                x <= 0 and y <= 0, V = max, reach)
            sys: full system dynamics, e.g. from dynamics.make_dynamics; None uses quadrotor with
                 uTMax=1, utMax=1, uMode from combine (union: max, intersection: min)
        Returns (grid, value function with time as the last axis).
    '''
    check_combine(combine, ("union", "intersection"), "6d")
    g = grid_6d(num)
    data1, data2 = init_values_6d(num)
    data = np.minimum(data1, data2) if combine == "union" else np.maximum(data1, data2)
    tau = np.arange(start=0, stop=lookback_length + 1e-5, step=t_step)
    if sys is None:
        sys = quadrotor(uTMax=1, utMax=1, dMax=0.0, uMode=U_MODE[combine], dMode='min')
    start = time.time()
    result = solve_hj(sys, g, data, tau)
    print(f"Direct computation time: {time.time() - start:.4f} seconds")
    return g, result


def main():
    '''
        Run the 6d direct computation and save it to results/result_direct_6d_<combine>.npy,
        shape (x, y, vx, vy, theta, w, t).
    '''
    from dynamics import make_dynamics
    parser = argparse.ArgumentParser()
    add_compute_args_6d(parser)
    args = parser.parse_args()
    try:
        check_combine(args.combine, ("union", "intersection"), "6d")
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=6)
    except (ValueError, TypeError, NotImplementedError) as e:
        parser.error(str(e))
    print(f"Dynamics: {args.dynamics} {vars(sys)}")
    try:
        _, result = direct_comp(args.grid_size, saveAllTimeStep=True, lookback_length=args.lookback_length,
                                combine=args.combine, t_step=args.t_step, sys=sys)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))
    os.makedirs(args.results_dir, exist_ok=True)
    path = result_files_6d(args.combine, args.results_dir)["direct"]
    np.save(path, result)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

    
    

