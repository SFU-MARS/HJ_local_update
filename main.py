from direct_numpy import direct_computation_old, direct_computation
from decomposition_numpy import decomposition, decomposition_old, DecompositionResult

import numpy as np
import heterocl as hcl

# import cv2
import sys as py_sys

# from set_2plot import *
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

import plotly.express as px

from odp.Grid import Grid
from odp.Shapes import *
from config import Config, construct2DGrid
from system import couple_u
from subsystem import subsys

# The function of computing first order spatial derivative
from update_V_numpy import spa_derivX, spa_derivY

import time

np.set_printoptions(threshold=py_sys.maxsize)
np.set_printoptions(precision=4)


def initConfig() -> Config:
    return Config(
        # number_of_grid_points=101,
        number_of_grid_points=5,
        lookback_length=0.2,
        time_steps=0.02,
        small_number=1e-5,
        sys_2d=couple_u(
            x=[0, 0],
            uMax=1,
            dMax=0.0,
            uMode="min",
            dMode="min",
        ),
        subsys_1d=subsys(
            x=[0],
            uMax=1,
            dMax=0.0,
            uMode="min",
            dMode="min",
        ),
    )


def plotArray(array, show=True):
    fig = px.imshow(array)
    if show:
        fig.show()


def correctionBasedOnDirectComputation(true_final, result_decomp, config: Config):

    print("correctionBasedOnDirectComputation")
    print("================================================")
    decomp_final = result_decomp[-1]
    result_diff = decomp_final - true_final

    print("true_final:")
    print(true_final)
    print("decomp_final:")
    print(decomp_final)

    indice = np.argwhere(abs(result_diff) > 1e-6)
    indice_transpose = indice.transpose()
    indices_ref = np.ravel_multi_index(indice_transpose, decomp_final.shape)

    # Initialize from config
    grid = construct2DGrid(number_of_grid_points=config._number_of_grid_points)
    data_init = config.value_function_2d(grid=grid)
    t_step = config._time_steps
    tau = config._tau
    sys = config._sys_2d

    data = data_init.copy()

    # Plotting Option
    po = PlotOptions(
        do_plot=True,
        plot_type="value",
        plotDims=[0, 1],
        colorscale="Bluered",
        save_fig=True,
        filename="local_update_3",
        interactive_html=True,
    )

    """
    Below is the local updating loop of the value function with respect to the indices
    pure numpy version
    """

    list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
    list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])

    data_change = np.zeros(tuple(grid.pts_each_dim))

    tNow = tau[0]

    start = time.time()

    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)

        # while tNow < tau[i]:
        # Update the value function

        for ind in range(len(indice)):
            # print(indices[i])
            # get indices
            x = indice[ind][0]
            y = indice[ind][1]
            # get spatial derivative
            dV_dx_L, dV_dx_R = spa_derivX(x, y, data, grid)
            dV_dy_L, dV_dy_R = spa_derivY(x, y, data, grid)
            # Get the average gradient
            dV_dx = (dV_dx_L + dV_dx_R) / 2
            dV_dy = (dV_dy_L + dV_dy_R) / 2
            # get the dynamical rates of change
            uOpt = sys.opt_ctrl_numpy(t, [list_x1[x], list_x2[y]], [dV_dx, dV_dy])
            dx_dt, dy_dt = sys.dynamics_numpy(t, [list_x1[x], list_x2[y]], uOpt, 0)
            # Updating the value function
            data_change[x, y] = dx_dt * dV_dx + dy_dt * dV_dy

        data = data + data_change * t_step
        tNow += t_step
        data_change = np.zeros(tuple(grid.pts_each_dim))

        data_ref = result_decomp[i].copy()
        np.put(data_ref, indices_ref, data[indice_transpose[0], indice_transpose[1]])

        data = data_ref.copy()

        execution_time = time.time() - start
    print("Total kernel time local: ", execution_time)
    print("Finished updating the value function")

    """
    Combination process: local updating result, and decomposition result
    """
    result_combine = decomp_final.copy()

    np.put(result_combine, indices_ref, data[indice_transpose[0], indice_transpose[1]])
    print("Corrected result:")
    print(result_combine)
    print("Im here")

    compareArrays(result_combine, true_final)

    """
    Comparision with direct computation
    """
    # plot_overlay_set(grid, true_final, decomp_final, result_combine, po)

    print("The total number of points: ", true_final.shape[0] * true_final.shape[1])
    print("The number of points getting locally updated: ", indice.shape)
    print(
        "Original number of points with different values: ",
        np.argwhere(abs(decomp_final - true_final) > 1e-6).shape,
    )
    print(
        "The number of points with different values after local updating: ",
        np.argwhere(abs(result_combine - true_final) > 1e-6).shape,
    )

    count = true_final.shape[0] * true_final.shape[1]

    diff_decomp = np.sum(result_diff) / count
    diff_corrected = np.sum(result_combine - true_final) / count
    print(
        "The average error between direct computation and decomposition: ", diff_decomp
    )
    print(
        "The average error between direct computation and local updates: ",
        diff_corrected,
    )

    diff_decomp = np.sum(abs(result_diff)) / count
    diff_corrected = np.sum(abs(result_combine - true_final)) / count
    print(
        "The average absolute error between direct computation and decomposition: ",
        diff_decomp,
    )
    print(
        "The average absolute error between direct computation and local updates: ",
        diff_corrected,
    )

    diff_decomp_max = np.max(abs(result_diff))
    diff_corrected_max = np.max(abs(result_combine - true_final))
    print(
        "The maximum absolute error between direct computation and decomposition: ",
        diff_decomp_max,
    )
    print(
        "The maximum absolute error between direct computation and local updates: ",
        diff_corrected_max,
    )


def correctionBasedOnLocalUpdate(
    decomposition_result: DecompositionResult, config: Config
):
    """

    def recompute_value(point, prev_combined_result):
        # use prev_combined result to recompute point
        old_val = point.value()
        new_val = point.recompute(prev_combined_result)
        flag = false
        if new_value - old_value > threshold2:
            flag = true

        return point, flag





    cumulative_points_to_correct = array of 0s based on the combined_result array.
    new_points_to_correct = array of 0s based on the combined_result array.
    for each time_step:

        # rest frontier, new_points_to correct

        # find the points to correct from lower and upper subsystem results
        for each point:
            if lower[point] - upper[point] < threshold:
                new_points_to_correct[point] = 1
                # add to cumulative_points_to_correct
                # Optimize this stuff
                cumulative_points_to_correct[point] = 1

        # for all points to be recomputed, recompute them.
        for point in cumulative_points_to_correct:
            flag, point = recompute_value(point, prev_combined_result)
            # Optimize this stuff: can we maintain the border and only check frontier for the border vertices?
            if flag:
                # should recompute the neighbors also.
                frontier[point] = 1

        #
        while frontier is not empty:
            # should check neighbors of all points in fronier
            for point in frontier:
                for neighbor of point:
                    if cumulative_points_to_correct[neighbor] is False:
                        new_points_to_correct[neighbor] = 1

            # reset frontier for next iteration
            for point in new_points_to_correct:
                new_points_to_correct[point] = 0
                flag, point = recompute_value(point, prev_combined_result)

                if flag:
                    # mark this point for corrections in all future time steps
                    cumulative_points_to_correct[point] = 1

                    # add it to frontier and check its neighbors
                    frontier[point] = 1

    """
    pass


def printResults(result_list):
    # a list of result arrays
    i = 0
    print(f"Printing results array of length {len(result_list)}")
    for result in result_list:
        print(i)
        print(result)
        i += 1


def compareArrays(array1, array2, number_of_precision_points=8, debug=False):
    diffArray = abs(array1 - array2)
    if debug:
        print("array1")
        print(array1)
        print("---------------------")
        print("array2")
        print(array2)
        print("---------------------")
        print("diff_array")
        print(diffArray)

    for precision in range(1, number_of_precision_points):
        precision_value = 1 / (10**precision)
        number_of_entires_with_error = len(
            np.argwhere(abs(diffArray) > precision_value)
        )
        print(f"1e-{precision}: {precision_value} {number_of_entires_with_error}")


def main():
    # Initializtion
    config = initConfig()

    # Perform direct computation and decomposition
    grid, result_true = direct_computation(
        config=config,
        saveAllTimeStep=True,
        # lookback_length=config._lookback_length,
    )
    printResults(result_list=result_true)
    # direct_comp_old(
    #     num=config._number_of_grid_points,
    #     saveAllTimeStep=True,
    #     lookback_length=config._lookback_length,
    # )
    result_decomp_old = decomposition_old(
        config._number_of_grid_points,
        saveAllTimeStep=True,
        lookback_length=config._lookback_length,
    )
    decomposition_result: DecompositionResult = decomposition(
        config=config,
        saveAllTimeStep=True,
    )
    # Initialize the value function
    # data_init = ShapeRectangle(grid, [-1.0, -1.0], [1.0, 1.0])
    data_init = config.value_function_2d(grid=grid)

    true_final = result_true[-1]
    decomp_final = decomposition_result.combined()[-1]
    decomp_final_old = result_decomp_old[-1]
    result_diff = decomp_final - true_final

    # for x, y in zip(result_decomp_old, decomposition_result.combined()):
    #     # result_diff2 = decomp_final - decomp_final_old
    #     compareArrays(x, y)
    #     # exit(1)
    #     # plotArray(result_diff2)

    print("comparing decomp_final and true_final")
    compareArrays(decomp_final, true_final)

    # exit(1)

    correctionBasedOnDirectComputation(
        true_final=true_final,
        result_decomp=decomposition_result.combined(),
        config=config,
    )


if __name__ == "__main__":
    main()
