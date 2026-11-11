# Decomposition of the 6D quadrotor into two 4D subsystems solved with optimized_dp's HJSolver:
# subsystem 1 (x, vx, theta, w) and subsystem 2 (y, vy, theta, w), combined as min (union) or max (intersection).
import argparse
import os
import time

import numpy as np
from odp.Shapes import Lower_Half_Space

from compute.config_6d import add_compute_args_6d, grid_sub_6d, result_files_6d
from compute.hj_solver import solve_hj
from common.combine import check_combine
from dynamics import U_MODE
from subdynamics.quadrotor_6d import subsys_1, subsys_2


def decomposition(num, saveAllTimeStep=True, lookback_length=0.1, combine="union", subsystems=None, t_step=0.01):
    '''
        Solve both 4D subsystems with HJSolver (accuracy "low", one step per time step) and combine them on
        the 6D grid (x, y, vx, vy, theta, w).
            combine: "union" (V = min(V1, V2), avoid, uMode max) or "intersection" (V = max(V1, V2), reach, uMode min)
            subsystems: (subsystem 1, subsystem 2) dynamics, e.g. from dynamics.make_subdynamics;
                None uses subsys_1, subsys_2 with uTMax=1, utMax=1, uMode from combine
        Returns (V, V1, V2) on the 6D grid, time as the first axis. V1 and V2 are broadcast views.
    '''
    check_combine(combine, ("union", "intersection"), "6d")
    g = grid_sub_6d(num)
    data_sub = Lower_Half_Space(g, 0, 0)  # x <= 0 for subsystem 1, y <= 0 for subsystem 2
    tau = np.arange(start=0, stop=lookback_length + 1e-5, step=t_step)
    if subsystems is None:
        subsystems = (subsys_1(uTMax=1, utMax=1, dMax=0.0, uMode=U_MODE[combine], dMode='min'),
                      subsys_2(uTMax=1, utMax=1, dMax=0.0, uMode=U_MODE[combine], dMode='min'))

    start = time.time()
    # (a, b, theta, w, t) -> (t, a, b, theta, w)
    V1 = np.moveaxis(solve_hj(subsystems[0], g, data_sub, tau), -1, 0)
    V2 = np.moveaxis(solve_hj(subsystems[1], g, data_sub, tau), -1, 0)

    # Subsystem 1 (x, vx, theta, w) and subsystem 2 (y, vy, theta, w) on the 6D grid (t, x, y, vx, vy, theta, w)
    result_upper = V1[:, :, np.newaxis, :, np.newaxis, :, :]
    result_lower = V2[:, np.newaxis, :, np.newaxis, :, :, :]
    result_full = np.minimum(result_upper, result_lower) if combine == "union" else np.maximum(result_upper, result_lower)
    result_upper = np.broadcast_to(result_upper, result_full.shape)
    result_lower = np.broadcast_to(result_lower, result_full.shape)
    print(f"Decomposition time: {time.time() - start:.4f} seconds")
    return result_full, result_upper, result_lower


def main():
    '''
        Run the 6d decomposition and save
            results/result_decomp_6d_<combine>.npy: V = min(V1, V2) or max(V1, V2), shape (t, x, y, vx, vy, theta, w)
            results/init_1_6d.npy, results/init_2_6d.npy: V1, V2 at t = 0 on the 6D grid, used for the leaking set
    '''
    from dynamics import make_subdynamics
    parser = argparse.ArgumentParser()
    add_compute_args_6d(parser)
    args = parser.parse_args()
    try:
        check_combine(args.combine, ("union", "intersection"), "6d")
        subsystems = make_subdynamics(args.dynamics, args.dynamics_params, args.combine, dim=6)
    except (ValueError, TypeError, NotImplementedError) as e:
        parser.error(str(e))
    print(f"Subsystem dynamics: {args.dynamics} {vars(subsystems[0])}")
    try:
        result_full, result_upper, result_lower = decomposition(
            args.grid_size, saveAllTimeStep=True, lookback_length=args.lookback_length, combine=args.combine,
            subsystems=subsystems, t_step=args.t_step)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))
    os.makedirs(args.results_dir, exist_ok=True)
    files = result_files_6d(args.combine, args.results_dir)
    np.save(files["decomp"], result_full)
    np.save(files["init_1"], result_upper[0])
    np.save(files["init_2"], result_lower[0])
    for key in ("decomp", "init_1", "init_2"):
        print(f"Saved {files[key]}")


if __name__ == "__main__":
    main()