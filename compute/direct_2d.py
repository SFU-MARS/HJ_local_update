# Direct computation of the 2d coupled control system with optimized_dp's HJSolver, used as the ground truth.
import argparse
import os
import time

import numpy as np

from compute.config_2d import Config, construct2DGrid, add_compute_args_2d, config_from_compute_args_2d, result_files_2d
from compute.hj_solver import solve_hj


def direct_computation(config: Config, saveAllTimeStep=True, debug=False):
    '''
        Solve the full 2D system config._sys_2d with HJSolver (accuracy "low", one step per time step).
        Returns (grid, value function with time as the first axis).
    '''
    print("direct_computation")
    print("================================================")
    g = construct2DGrid(number_of_grid_points=config._number_of_grid_points)
    data = config.value_function_2d(grid=g)
    start = time.time()
    result = solve_hj(config._sys_2d, g, data, config._tau)
    print(f"Number of timesteps: {result.shape[-1]}")
    print(f"Direct computation time: {time.time() - start:.4f} seconds")
    return g, np.moveaxis(result, -1, 0)


def main():
    '''
        Run the 2d direct computation and save it to results/result_direct_2d_<combine>.npy, shape (x, y, t).
    '''
    parser = argparse.ArgumentParser()
    add_compute_args_2d(parser)
    args = parser.parse_args()
    config = config_from_compute_args_2d(parser, args)
    try:
        _, data_list = direct_computation(config=config, saveAllTimeStep=True)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))
    os.makedirs(args.results_dir, exist_ok=True)
    path = result_files_2d(args.combine, args.results_dir)["direct"]
    np.save(path, np.moveaxis(np.asarray(data_list), 0, -1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
