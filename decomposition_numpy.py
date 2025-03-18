import imp
from pprint import pprint
import numpy as np

from odp.Grid import Grid
from odp.Shapes import *

from subsystem import subsys
from update_V_numpy import spa_deriv

# Plot Options
from odp.Plots import *
from config import Config, construct1DGrid, construct2DGrid
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


def decomposition_old(num, saveAllTimeStep=True, lookback_length=0.02, debug=False):

    print("decomposition_old")
    print("================================================")
    # num = 51

    # Create Grid
    grid_min = np.array([-4.0])
    grid_max = np.array([4.0])
    dims = grid_min.shape[0]
    N = np.array([num])
    g = Grid(grid_min, grid_max, dims, N)

    # Initialize value function
    data_sub = ShapeRectangle(g, [-1.0], [1.0])

    # print(f"grid: {g.vs}")
    # print(f"data_sub: {data_sub}")

    ## Look-back length and time step of computation
    # lookback_length = 0.02
    t_step = 0.02
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    # Set system dynamics
    sys = subsys(x=[0], uMax=1, dMax=0.0, uMode="min", dMode="min")

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
        print(f"Time step: {count}, {t}")
        count += 1

        # while tNow < tau[i]:
        # Update the value function
        for x in range(len(list_x)):
            # Get spatial derivative
            dV_dx_L, dV_dx_R = spa_deriv(x, data_sub, g)
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

    # print("Total kernel time: ", execution_time)
    print(f"Total kernel time: {execution_time:.4f} seconds")

    """
    Combine the results from 2 subsystems
    """

    grid_min = np.array([-4.0, -4.0])
    grid_max = np.array([4.0, 4.0])
    dims = grid_min.shape[0]
    N = np.array([num, num])
    g = Grid(grid_min, grid_max, dims, N)

    if not saveAllTimeStep:
        result_upper_flip = np.flip(data_sub, axis=0)
        result_upper_expand = np.tile(result_upper_flip, (num, 1))
        print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (1, 0))

        result_lower_flip = np.flip(data_sub, axis=0)
        result_lower_expand = np.tile(result_lower_flip, (num, 1))
        print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (0, 1))

        result_full = np.maximum(result_upper, result_lower)

    else:

        result_upper_flip = np.flip(data_list, axis=0)
        result_upper_expand = np.tile(data_list, (num, 1, 1))
        # print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (1, 2, 0))
        # print(result_upper.shape)

        result_lower_flip = np.flip(data_list, axis=0)
        result_lower_expand = np.tile(data_list, (num, 1, 1))
        # print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (1, 0, 2))
        # print(result_lower.shape)

        result_full = np.maximum(result_upper, result_lower)

        # print('Decomposition Correct')

    return result_full


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

    # print(f"grid: {g.vs}")
    # print(f"data_sub: {data_sub}")
    # print(f"list_x: {list_x}")
    # print(f"data_change: {data_change}")

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
        print(f"Time step: {count}, {t}")
        count += 1

        # while tNow < tau[i]:
        # Update the value function
        for x in range(len(list_x)):
            # Get spatial derivative
            dV_dx_L, dV_dx_R = spa_deriv(x, data_sub, g)
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

    print(f"Total kernel time: {execution_time:.4f} seconds")

    """
    Combine the results from 2 subsystems
    """

    if not saveAllTimeStep:
        result_full, result_upper, result_lower = combine_subsystem_results(
            result_upper=data_sub,
            result_lower=data_sub,
            number_of_grid_points=config._number_of_grid_points,
        )
    else:
        result_full, result_upper, result_lower = (
            combine_subsystem_results_all_time_steps(
                result_upper_list=data_list,
                result_lower_list=data_list,
                number_of_grid_points=config._number_of_grid_points,
            )
        )
        pass

    decomposition_result = DecompositionResult(
        upper_subsystem_results=result_upper,
        lower_subsystem_results=result_lower,
        combined_results=result_full,
    )

    return decomposition_result


def combine_subsystem_results(result_upper, result_lower, number_of_grid_points):
    result_upper_flip = np.flip(result_upper, axis=0)
    result_upper_expand = np.tile(result_upper_flip, (number_of_grid_points, 1))
    print(result_upper_expand.shape)
    result_upper = np.transpose(result_upper_expand, (1, 0))

    result_lower_flip = np.flip(result_lower, axis=0)
    result_lower_expand = np.tile(result_lower_flip, (number_of_grid_points, 1))
    print(result_lower_expand.shape)
    result_lower = np.transpose(result_lower_expand, (0, 1))

    result_full = np.maximum(result_upper, result_lower)
    return result_full, result_upper, result_lower


def combine_subsystem_results_all_time_steps(
    result_upper_list, result_lower_list, number_of_grid_points
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
    result_full = np.maximum(result_upper, result_lower)
    return result_full, result_upper, result_lower
