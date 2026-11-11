# Decomposition of the 2d coupled control system: the 1D subsystem is solved with optimized_dp's HJSolver,
# and V1(x), V2(y) are combined as min (union) or max (intersection).
import argparse
import os
import time

import numpy as np

from compute.config_2d import add_compute_args_2d, grid_sub_2d, init_value_sub_2d, result_files_2d
from compute.hj_solver import solve_hj
from common.combine import check_combine


def decomposition(num, subsystem, lookback_length=0.2, combine="intersection", t_step=0.02):
    '''
        Solve the 1D subsystem with HJSolver (accuracy "low", one step per time step). Both subsystems
        x and y have the same dynamics and initial value, so it is solved once and used for V1(x) and V2(y).
            subsystem: 1D subsystem dynamics, e.g. from dynamics.make_subdynamics (uMode from combine)
            combine: "union" (V = min(V1, V2), avoid) or "intersection" (V = max(V1, V2), reach)
        Returns (V, V1, V2) on the 2D grid, time as the first axis. V1 and V2 are broadcast views.
    '''
    check_combine(combine, ("union", "intersection"), "2d")
    g = grid_sub_2d(num)
    tau = np.arange(start=0, stop=lookback_length + 1e-5, step=t_step)
    start = time.time()
    # (s, t) -> (t, s)
    V_sub = np.moveaxis(solve_hj(subsystem, g, init_value_sub_2d(g), tau), -1, 0)

    # Subsystem 1 (x) and subsystem 2 (y) on the 2D grid (t, x, y)
    result_upper = V_sub[:, :, np.newaxis]
    result_lower = V_sub[:, np.newaxis, :]
    result_full = np.minimum(result_upper, result_lower) if combine == "union" else np.maximum(result_upper, result_lower)
    result_upper = np.broadcast_to(result_upper, result_full.shape)
    result_lower = np.broadcast_to(result_lower, result_full.shape)
    print(f"Decomposition time: {time.time() - start:.4f} seconds")
    return result_full, result_upper, result_lower


def main():
    '''
        Run the 2d decomposition and save
            results/result_decomp_2d_<combine>.npy: V = min(V1, V2) or max(V1, V2), shape (x, y, t)
            results/init_1_2d.npy, results/init_2_2d.npy: V1, V2 at t = 0 on the 2D grid, used for the leaking set
    '''
    from dynamics import make_subdynamics
    parser = argparse.ArgumentParser()
    add_compute_args_2d(parser)
    args = parser.parse_args()
    try:
        check_combine(args.combine, ("union", "intersection"), "2d")
        subsystem, = make_subdynamics(args.dynamics, args.dynamics_params, args.combine, dim=2)
    except (ValueError, TypeError, NotImplementedError) as e:
        parser.error(str(e))
    print(f"Subsystem dynamics: {args.dynamics} {vars(subsystem)}")
    try:
        result_full, result_upper, result_lower = decomposition(
            args.grid_size, subsystem, lookback_length=args.lookback_length, combine=args.combine, t_step=args.t_step)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))
    os.makedirs(args.results_dir, exist_ok=True)
    files = result_files_2d(args.combine, args.results_dir)
    np.save(files["decomp"], np.moveaxis(result_full, 0, -1))
    np.save(files["init_1"], result_upper[0])
    np.save(files["init_2"], result_lower[0])
    for key in ("decomp", "init_1", "init_2"):
        print(f"Saved {files[key]}")


if __name__ == "__main__":
    main()
