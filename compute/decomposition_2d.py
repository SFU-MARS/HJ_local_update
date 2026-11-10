from pprint import pprint
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from subdynamics.coupled_2d import subsys
from common.spatial_deriv import spa_deriv

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
    print("decomposition")
    print("================================================")

    # Create 1D Grid
    g = construct1DGrid(
        number_of_grid_points=config._number_of_grid_points,
        min_val=-4.0,
        max_val=4.0,
    )

    # Initialize value function for 1D
    data_sub = config.value_function_1d(grid=g)
    # Set 1D system dynamics
    sys = config._subsys_1d
    t_step = config._time_step
    small_number = config._small_number
    tau = config._tau

    # temp arrays for processing
    list_x = np.reshape(g.vs[0], g.pts_each_dim[0])
    data_change = np.zeros(tuple(g.pts_each_dim))

    # Set computation task and compute HJ PDE
    """
    Computation in subsystems
    """
    if saveAllTimeStep:
        data_list = []
        data_list.append(data_sub)
        # print('The shape of data list is: ', data_list)

    tNow = tau[0]
    start = time.time()
    count = 0

    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        count += 1

        # while tNow < tau[i]:
        # Update the value function
        for x in range(len(list_x)):
            # Get spatial derivative
            dV_dx_L, dV_dx_R = spa_deriv((x,), data_sub, g, 0, boundary="one_sided")
            # Get the average gradient
            dV_dx = (dV_dx_L + dV_dx_R) / 2
            # Get the dynamical rates of change
            uOpt = sys.opt_ctrl_numpy(t, [list_x[x]], dV_dx)
            dx_dt = sys.dynamics_numpy(t, [list_x[x]], uOpt, 0)

            # Updating the value function
            data_change[x] = dx_dt * dV_dx

        data_sub = data_sub + t_step * data_change
        if saveAllTimeStep:
            data_list.append(data_sub)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow = tNow + t_step

    execution_time = time.time() - start

    # print(f"Total kernel time: {execution_time:.4f} seconds")
    print(f"Decomposition time: {execution_time:.4f} seconds")

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
    result = decomposition(config=config, saveAllTimeStep=True)
    os.makedirs(args.results_dir, exist_ok=True)
    files = result_files_2d(args.combine, args.results_dir)
    for key, data in (("decomp", result.combined()), ("sub1", result.subsystem1()), ("sub2", result.subsystem2())):
        np.save(files[key], np.moveaxis(np.asarray(data), 0, -1))
        print(f"Saved {files[key]}")


if __name__ == "__main__":
    main()
