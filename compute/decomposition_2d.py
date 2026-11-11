# Decomposition of the 2d coupled control system: the 1D subsystem is solved with optimized_dp's HJSolver,
# and V1(x), V2(y) are combined as min (union) or max (intersection).
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from compute.hj_solver import solve_hj

# Plot Options
from odp.Plots import *
import argparse
import os

from compute.config_2d import Config, construct1DGrid, construct2DGrid, add_compute_args_2d, config_from_compute_args_2d, result_files_2d
import time


class DecompositionResult:
    def __init__(
        self,
        upper_subsystem_results,
        lower_subsystem_results,
        combined_results,
    ) -> None:
        self._upper_subsystem_results = upper_subsystem_results
        self._lower_subsystem_results = lower_subsystem_results
        self._combined_results = combined_results

    def combined(self):
        return self._combined_results

    def subsystem1(self):
        return self._upper_subsystem_results

    def subsystem2(self):
        return self._lower_subsystem_results


def decomposition(config: Config, saveAllTimeStep=True, debug=False):
    '''
        Solve the 1D subsystem config._subsys_1d with HJSolver (accuracy "low", one step per time step) and
        combine V1(x) and V2(y) (the same 1D result) into the 2D grid with min (union) or max (intersection).
        Returns a DecompositionResult, time as the first axis.
    '''
    print("decomposition")
    print("================================================")

    # Create 1D Grid
    g = construct1DGrid(
        number_of_grid_points=config._number_of_grid_points,
        min_val=-4.0,
        max_val=4.0,
    )

    # Initialize value function for 1D and solve it
    data_sub = config.value_function_1d(grid=g)
    start = time.time()
    result_1d = solve_hj(config._subsys_1d, g, data_sub, config._tau)
    data_list = list(np.moveaxis(result_1d, -1, 0))
    data_sub = data_list[-1]
    print(f"Decomposition time: {time.time() - start:.4f} seconds")

    """
    Combine the results from 2 subsystems
    """

    if not saveAllTimeStep:
        result_full, result_upper, result_lower = combine_subsystem_results(
            result_upper=data_sub,
            result_lower=data_sub,
            number_of_grid_points=config._number_of_grid_points,
            use_union=config._use_union,
        )
    else:
        result_full, result_upper, result_lower = (
            combine_subsystem_results_all_time_steps(
                result_upper_list=data_list,
                result_lower_list=data_list,
                number_of_grid_points=config._number_of_grid_points,
                use_union=config._use_union,
            )
        )
        pass

    decomposition_result = DecompositionResult(
        upper_subsystem_results=result_upper,
        lower_subsystem_results=result_lower,
        combined_results=result_full,
    )

    return decomposition_result


def combine_subsystem_results(result_upper, result_lower, number_of_grid_points, use_union):
    result_upper_flip = np.flip(result_upper, axis=0)
    result_upper_expand = np.tile(result_upper_flip, (number_of_grid_points, 1))
    print(result_upper_expand.shape)
    result_upper = np.transpose(result_upper_expand, (1, 0))

    result_lower_flip = np.flip(result_lower, axis=0)
    result_lower_expand = np.tile(result_lower_flip, (number_of_grid_points, 1))
    print(result_lower_expand.shape)
    result_lower = np.transpose(result_lower_expand, (0, 1))

    if use_union:
        result_full = np.minimum(result_upper, result_lower)
    else:
        result_full = np.maximum(result_upper, result_lower)
    return result_full, result_upper, result_lower


def combine_subsystem_results_all_time_steps(
    result_upper_list, result_lower_list, number_of_grid_points, use_union,
):

    result_upper_flip = np.flip(result_upper_list, axis=0)
    result_upper_expand = np.tile(result_upper_list, (number_of_grid_points, 1, 1))
    # print(result_upper_expand.shape)
    result_upper = np.transpose(result_upper_expand, (1, 2, 0))
    # print(result_upper.shape)

    result_lower_flip = np.flip(result_lower_list, axis=0)
    result_lower_expand = np.tile(result_lower_list, (number_of_grid_points, 1, 1))
    # print(result_lower_expand.shape)
    result_lower = np.transpose(result_lower_expand, (1, 0, 2))
    # print(result_lower.shape)
    if use_union:
        result_full = np.minimum(result_upper, result_lower)
    else:
        result_full = np.maximum(result_upper, result_lower)
    return result_full, result_upper, result_lower


def main():
    '''
        Run the 2d decomposition and save, all with shape (x, y, t):
            results/result_decomp_2d_<combine>.npy: V = min (union) or max (intersection) of V1, V2
            results/result_sub1_2d_<combine>.npy, results/result_sub2_2d_<combine>.npy: V1(x), V2(y) on the 2D grid
    '''
    parser = argparse.ArgumentParser()
    add_compute_args_2d(parser)
    args = parser.parse_args()
    config = config_from_compute_args_2d(parser, args)
    try:
        result = decomposition(config=config, saveAllTimeStep=True)
    except ValueError as e:  # HJSolver needed more than one step per time step
        parser.error(str(e))
    os.makedirs(args.results_dir, exist_ok=True)
    files = result_files_2d(args.combine, args.results_dir)
    for key, data in (("decomp", result.combined()), ("sub1", result.subsystem1()), ("sub2", result.subsystem2())):
        np.save(files[key], np.moveaxis(np.asarray(data), 0, -1))
        print(f"Saved {files[key]}")


if __name__ == "__main__":
    main()
