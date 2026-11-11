# Direct computation of the 2d coupled control system with optimized_dp's HJSolver, used as the ground truth.
import argparse
import os
import time

import numpy as np

from compute.config_2d import add_compute_args_2d, grid_2d, init_values_2d, result_files_2d
from compute.hj_solver import solve_hj
from common.combine import check_combine


def direct_comp(num, sys, lookback_length=0.2, combine="intersection", t_step=0.02):
    '''
        Solve the 2D coupled control system with HJSolver (accuracy "low", one step per time step).
            sys: full system dynamics, e.g. from dynamics.make_dynamics (uMode from combine)
            combine: "union" (V = min(V1, V2), avoid) or "intersection" (V = max(V1, V2), reach)
        Returns (grid, value function with time as the last axis).
    '''
    check_combine(combine, ("union", "intersection"), "2d")
    g = grid_2d(num)
    data1, data2 = init_values_2d(num)
    data = np.minimum(data1, data2) if combine == "union" else np.maximum(data1, data2)
    tau = np.arange(start=0, stop=lookback_length + 1e-5, step=t_step)
    start = time.time()
    result = solve_hj(sys, g, data, tau)
    print(f"Direct computation time: {time.time() - start:.4f} seconds")
    return g, result


def main():
    '''
        Run the 2d direct computation and save it to results/result_direct_2d_<combine>.npy, shape (x, y, t).
    '''
    from dynamics import make_dynamics
    parser = argparse.ArgumentParser()
    add_compute_args_2d(parser)
    args = parser.parse_args()
    try:
        check_combine(args.combine, ("union", "intersection"), "2d")
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=2)
    except (ValueError, TypeError, NotImplementedError) as e:
        parser.error(str(e))
    print(f"Dynamics: {args.dynamics} {vars(sys)}")
    try:
        _, result = direct_comp(args.grid_size, sys, lookback_length=args.lookback_length, combine=args.combine,
                                t_step=args.t_step)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))
    os.makedirs(args.results_dir, exist_ok=True)
    path = result_files_2d(args.combine, args.results_dir)["direct"]
    np.save(path, result)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
