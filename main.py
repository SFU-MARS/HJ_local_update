import argparse
from direct_numpy import direct_computation_old, direct_computation
from decomposition_numpy import decomposition, decomposition_old, DecompositionResult

import numpy as np
import heterocl as hcl
import copy
# import cv2
import sys as py_sys

# from set_2plot import *
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

import plotly.express as px

from odp.Grid import Grid
from odp.Shapes import *
from config import Config, construct2DGrid, Frontier
from system import couple_u
from subsystem import subsys

# The function of computing first order spatial derivative
from update_V_numpy import spa_derivX, spa_derivY

import time

np.set_printoptions(threshold=py_sys.maxsize)
np.set_printoptions(precision=5)


def initConfig(use_union: bool, 
               grid_size: int, 
               generate_big_delta: int, 
               big_delta: float,) -> Config:
    if use_union:
        sys_2d=couple_u(
            x=[0, 0],
            uMax=1,
            dMax=0.0,
            uMode="max",
            dMode="min",
        )
        subsys_1d=subsys(
            x=[0],
            uMax=1,
            dMax=0.0,
            uMode="max",
            dMode="min",
        )
    else:
        sys_2d=couple_u(
            x=[0, 0],
            uMax=1,
            dMax=0.0,
            uMode="min",
            dMode="min",
        )
        subsys_1d=subsys(
            x=[0],
            uMax=1,
            dMax=0.0,
            uMode="min",
            dMode="min",
        )

    return Config(
        number_of_grid_points=grid_size,
        lookback_length=0.20,
        time_step=0.02,
        small_number=1e-5,
        sys_2d=sys_2d,
        subsys_1d=subsys_1d,
        use_union=use_union,
        threshold2=1e-6,
        use_optimization1=True,
        generate_big_delta=generate_big_delta,
        big_delta=big_delta,
    )

def plotArray(array, show=True):
    fig = px.imshow(array)
    if show:
        fig.show()

def correctionBasedOnDirectComputation(
    true_final, result_decomp, config: Config, debug=False
):

    print("correctionBasedOnDirectComputation")
    print("================================================")
    decomp_final = result_decomp[-1]
    result_diff = decomp_final - true_final

    if debug:
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
    t_step = config._time_step
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
    print("Correcting decomposition results...")

    start = time.time()
    count = 0

    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print(f"Time step: {count}, {t}")
        count += 1

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
    print(f"Total kernel time: {execution_time:.4f} seconds")

    """
    Combination process: local updating result, and decomposition result
    """
    result_combine = decomp_final.copy()

    np.put(result_combine, indices_ref, data[indice_transpose[0], indice_transpose[1]])
    if debug:
        print("Corrected result:")
        print(result_combine)
        compareArrays(result_combine, true_final)

def printCorrectnessStatistics(
        direct_computation_results, decomposition_results:DecompositionResult, corrected_decomposition_results:DecompositionResult, 
        ):
    print("Processing stats")
    print("================================================")

    """
    Comparision with direct computation
    """
    # plot_overlay_set(grid, true_final, decomp_final, result_combine, po)
    direct_computation_final = direct_computation_results[-1]
    decomp_final = decomposition_results.combined()[-1]
    decomp_final_correct = corrected_decomposition_results.combined()[-1]

    print(f"The total number of points: {direct_computation_final.shape[0] * direct_computation_final.shape[1]}")
    print(
        f"Original number of points with different values: {np.argwhere(abs(decomp_final - direct_computation_final) > 1e-6).shape}"
    )
    print(
        f"Decomposition method: Number of points with different values (1e-6): {np.argwhere(abs(decomp_final - direct_computation_final) > 1e-6).shape}"
    )
    print(
        f"After local update, Number of points with different values (1e-6): {np.argwhere(abs(decomp_final_correct - direct_computation_final) > 1e-6).shape}"
    )

    total_num_of_points = direct_computation_final.shape[0] * direct_computation_final.shape[1]
    diff_decomp = np.sum(abs(decomp_final - direct_computation_final)) / total_num_of_points
    diff_corrected = np.sum(abs(decomp_final_correct - direct_computation_final)) / total_num_of_points
    print(
        f"The average absolute error between direct computation and decomposition: {diff_decomp:0.10f}"
    )
    print(
        f"The average absolute error between direct computation and local updates: {diff_corrected:0.10f}"
    )

    diff_decomp_max = np.max(abs(decomp_final - direct_computation_final))
    diff_corrected_max = np.max(abs(decomp_final_correct - direct_computation_final))
    print(
        f"The maximum absolute error between direct computation and decomposition: {diff_decomp_max:0.10f}"
    )
    print(
        f"The maximum absolute error between direct computation and local updates: {diff_corrected_max:0.10f}"
    )

    print("Number of values with error:")
    compareArrays(array1=direct_computation_final,
                  array2=decomp_final_correct)

class IterStats:
    def __init__(self, iteration,
                 num_of_indices_to_correct, 
                 total_indices_corrected,
                 init_time,
                 identification_time,
                 correction1_time,
                 correction2_time,
                 ):
        self._iteration = iteration
        self._num_of_indices_to_correct = num_of_indices_to_correct
        self._total_indices_corrected = total_indices_corrected
        self._init_time = init_time
        self._identification_time = identification_time
        self._correction1_time = correction1_time
        self._correction2_time = correction2_time
        self._total_time = init_time+identification_time+correction1_time+correction2_time

    def printHeader(self):
        print("iteration, "
        "num_of_indices_to_correct, "
        "total_indices_corrected,"
        # "init_time,"
        # "identification_time, "
        # "correction1_time, "
        # "correction2_time, "
        "total_time"
        )
    
    def printStats(self):
        print(f"{self._iteration}, "
        f"{self._num_of_indices_to_correct}, "
        f"{self._total_indices_corrected},"
        # f"{round(self._init_time, ndigits=4)},"
        # f"{round(self._identification_time, ndigits=4)}, "
        # f"{round(self._correction1_time, ndigits=4)}, "
        # f"{round(self._correction2_time, ndigits=4)}, "
        f"{round(self._total_time, ndigits=4)}"
        )

class PerfStats:
    def __init__(
            self, 
            decomposition_result: DecompositionResult,
            true_result, 
            config: Config,):
        self._decomposition_result = decomposition_result
        self._true_result = true_result
        self._config = config
        self._iter_stats:list[IterStats] = []

    def addIterStat(self, iter_stat: IterStats):
        self._iter_stats.append(iter_stat)

    def printStats(self):
        self.printIterStats()

    def printIterStats(self):
        self._iter_stats[0].printHeader();
        for iter_stat in self._iter_stats:
            iter_stat.printStats()

    def totalPointsCorrected(self):
        points_corrected = 0
        for iter_stat in self._iter_stats:
            points_corrected += iter_stat._total_indices_corrected            
        return points_corrected
    
    def totalPointsIdentified(self):
        points_identified = 0
        for iter_stat in self._iter_stats:
            points_identified += iter_stat._num_of_indices_to_correct
        return points_identified


class CorrectionBasedOnLocalUpdate:
    def __init__(
        self,
        decomposition_result: DecompositionResult,
        true_result,
        config: Config,
    ):
        self._decomposition_result = decomposition_result
        self._true_result = true_result
        self._config = config
        # grid data for current and previous timesteps
        self.prev_data = None
        self.current_data = None
        self._grid = self._config.grid_2D()
        self._sys = self._config._sys_2d

        self._list_x1 = np.reshape(self._grid.vs[0], self._grid.pts_each_dim[0])
        self._list_x2 = np.reshape(self._grid.vs[1], self._grid.pts_each_dim[1])

        self._perf_stats = PerfStats(decomposition_result=decomposition_result,
                                     true_result=true_result,
                                     config=config,
                                     )

    def recomputeValueChange(self, data, x, y, t):
        # recompute value at index data[x][y] using the grid at timestep t
        # Note: t is not being used in both opt_ctrl_numpy() and dynamics_numpy().

        # Get spatial derivative
        dV_dx_L, dV_dx_R = spa_derivX(x, y, data, self._grid)
        dV_dy_L, dV_dy_R = spa_derivY(x, y, data, self._grid)

        # Get the average gradient
        dV_dx = (dV_dx_L + dV_dx_R) / 2
        dV_dy = (dV_dy_L + dV_dy_R) / 2

        # Get the dynamical rates of change
        uOpt = self._sys.opt_ctrl_numpy(t, [self._list_x1[x], self._list_x2[y]], [dV_dx, dV_dy])
        dx_dt, dy_dt = self._sys.dynamics_numpy(t, [self._list_x1[x], self._list_x2[y]], uOpt, 0)

        # Updating the value function
        data_change = dx_dt * dV_dx + dy_dt * dV_dy
        data_change = data_change*self._config._time_step
        return data_change
    
    def updateValue(self, curr_result, prev_result, index, frontier:Frontier, updated_points_curr: Frontier, time_step) -> bool:
        x = index[0]
        y = index[1]
        new_value_change = self.recomputeValueChange(data=prev_result, x=x, y=y, t=time_step,)
        old_value = curr_result[x][y]
        new_value = new_value_change + prev_result[x][y]
        curr_result[x][y] = new_value

        if self._config._use_optimization1:
            # Avoid correcting indices which are already corrected in the current iteration
            updated_points_curr.add(index)

        flag = False
        if abs(old_value - new_value) > self.threshold2():
            flag = True
            prev_index_on_x = self.prevIndexOnX(index=index)
            next_index_on_x = self.nextIndexOnX(index=index)
            prev_index_on_y = self.prevIndexOnY(index=index)
            next_index_on_y = self.nextIndexOnY(index=index)
            if self._config._use_optimization1:
                if updated_points_curr.hasIndex(index=prev_index_on_x) is False:
                    frontier.add(prev_index_on_x)
                if updated_points_curr.hasIndex(index=next_index_on_x) is False:
                    frontier.add(next_index_on_x)
                if updated_points_curr.hasIndex(index=prev_index_on_y) is False:
                    frontier.add(prev_index_on_y)
                if updated_points_curr.hasIndex(index=next_index_on_y) is False:
                    frontier.add(next_index_on_y)
            else:
                frontier.add(prev_index_on_x)
                frontier.add(next_index_on_x)
                frontier.add(prev_index_on_y)
                frontier.add(next_index_on_y)
        return flag

    """
    cumulative_points_to_correct = array of 0s based on the combined_result array.
    new_points_to_correct = array of 0s based on the combined_result array.
    for each time_step:

        # reset frontier, new_points_to_correct to all false

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
                    # if cumulative_points_to_correct[neighbor] is False:
                        new_points_to_correct[neighbor] = 1

            # reset frontier for next iteration
            for point in new_points_to_correct:
                new_points_to_correct[point] = 0
                frontier[point] = 0
                flag, point = recompute_value(point, prev_combined_result)

                if flag:
                    # mark this point for corrections in all future time steps
                    cumulative_points_to_correct[point] = 1

                    # add it to frontier and check its neighbors
                    frontier[point] = 1

    """
    def doCorrection(self, debug=False):
        if debug:
            print("================================================")
            print("START doCorrection")
        result_combined_all_timesteps = self._decomposition_result.combined()
        subsystem1_data_all_timesteps = self._decomposition_result.subsystem1()
        subsystem2_data_all_timesteps = self._decomposition_result.subsystem2()

        combined_data_prev = result_combined_all_timesteps[0]

        frontier = Frontier(x=self._config._number_of_grid_points,
                            y=self._config._number_of_grid_points)
        next_frontier = Frontier(x=self._config._number_of_grid_points,
                            y=self._config._number_of_grid_points)
        updated_points_curr = Frontier(x=self._config._number_of_grid_points,
                            y=self._config._number_of_grid_points)
        
        updated_points_prev = Frontier(x=self._config._number_of_grid_points,
                            y=self._config._number_of_grid_points)
        
        # cumulative_Frontier = Frontier(x=combined_data_prev.shape[0],
        #                     y=combined_data_prev.shape[1])
        time_step = self._config._time_step
        small_number = self._config._small_number
        tau = self._config._tau

        curr_tau = tau[0]
        start_time = time.time()

        for i in range(1, len(tau)):
            ###### INIT FOR EACH TIME STEP ######
            
            t = np.array([curr_tau, tau[i]])
            curr_result_upper = subsystem1_data_all_timesteps[i]
            curr_result_lower = subsystem2_data_all_timesteps[i]
            curr_result = result_combined_all_timesteps[i]
            prev_result = result_combined_all_timesteps[i - 1]
            curr_result_true = self._true_result[i]
            total_indices_corrected = 0

            # Compute threshold big delta in Algorithm 1
            bigDelta = self.getBigDelta(
                curr_result_combined = curr_result, 
                prev_result_combined = prev_result, 
                decomp_result_init= result_combined_all_timesteps[0], 
                decomp_result_final = result_combined_all_timesteps[-1],)
            print(f"big_delta:{bigDelta}")

            curr_time = time.time()
            init_time = time.time() - start_time
            start_time = curr_time
            
            ###### IDENTIFY INDICES TO CORRECT ######

            new_indices_to_correct = self.getNewPointsToCorrect(
                curr_result_upper, curr_result_lower, bigDelta
            )

            curr_time = time.time()
            identification_time = time.time() - start_time
            start_time = curr_time

            ###### CORRECTION 1: CORRECT THE IDENTIFIED INDICES  ######

            for index in new_indices_to_correct:
                self.updateValue(curr_result=curr_result, prev_result=prev_result, index=index, frontier=frontier, updated_points_curr=updated_points_curr, time_step=i)
                total_indices_corrected += 1

            if debug:
                print(f"new_indices_to_correct: {len(new_indices_to_correct)}")
                print(f"Frontier: {frontier.count()}")

            curr_time = time.time()
            correction1_time = time.time() - start_time
            start_time = curr_time

            ###### CORRECTION 2: RECURSIVELY CORRECT THE NGHS OF IDENTIFIED INDICES  ######

            while frontier.isEmpty() == False:
                for index in frontier.activeIndices():
                    # Update frontier vertex and check if its nghs need to be added to the next_frontier
                    self.updateValue(curr_result=curr_result, prev_result=prev_result, index=index, frontier=next_frontier, updated_points_curr=updated_points_curr, time_step=i)
                    total_indices_corrected += 1
                
                frontier, next_frontier = next_frontier, frontier
                next_frontier.reset()
                if debug:
                    print(f"Frontier: {frontier.count()}")            

            result_combined_all_timesteps[i] = curr_result

            # reset updated_points_curr for next iteration
            if self._config._use_optimization1:
                updated_points_curr.reset()

            curr_time = time.time()
            correction2_time = time.time() - start_time
            start_time = curr_time

            self._perf_stats.addIterStat(IterStats(
                iteration=i,
                num_of_indices_to_correct=len(new_indices_to_correct),
                total_indices_corrected=total_indices_corrected,
                init_time=init_time,
                identification_time=identification_time,
                correction1_time=correction1_time,
                correction2_time=correction2_time,
            ))
            
            if debug:
                print("----------------------------------------------------")

        if debug:
            print("================================================")
            print("END doCorrection")

    def getBigDelta(self, curr_result_combined, prev_result_combined, decomp_result_init, decomp_result_final):
        if self._config._generate_big_delta == 0:
            # Use a constant big_delta defined in the command-line
            # Useful for testing
            return self._config._big_delta
        elif self._config._generate_big_delta == 1:
            # Use new theory to generate big_delta
            # here, we use the initial and final decomposition result to generate the data
            max_diff = np.max(abs(decomp_result_final - decomp_result_init))
            return max_diff
        elif self._config._generate_big_delta == 2:
            # Use old theory to generate big_delta
            # Here, we use the previous and current decomposition result.
            # We will have different big_deltas for each timestep
            max_diff = np.max(abs(curr_result_combined - prev_result_combined))
            return max_diff

    def threshold2(self):
        return self._config._threshold2
    
    def prevIndexOnX(self, index):
        x = index[0]
        y = index[1]
        max_index = self._config._number_of_grid_points
        y = (y-1) % max_index
        prev_index = (x, y)
        return prev_index
        
    def nextIndexOnX(self, index):
        x = index[0]
        y = index[1]
        max_index = self._config._number_of_grid_points
        y = (y+1) % max_index
        next_index = (x, y)
        return next_index

    def prevIndexOnY(self, index):
        x = index[0]
        y = index[1]
        max_index = self._config._number_of_grid_points
        x = (x-1) % max_index
        prev_index = (x, y)
        return prev_index
        
    def nextIndexOnY(self, index):
        x = index[0]
        y = index[1]
        max_index = self._config._number_of_grid_points
        x = (x+1) % max_index
        next_index = (x, y)
        return next_index
    
    def getNewPointsToCorrect(self, result_upper, result_lower, bigDelta):
        result_diff = abs(result_upper - result_lower)

        # Get new_points_to_correct =
        indices_to_correct = np.argwhere(result_diff < bigDelta)
        # print(f"result_upper: \n{result_upper}")
        # print(f"result_lower: \n{result_lower}")
        # print(f"result_diff: \n{result_diff}")

        # print(f"indices_to_correct: \n{indices_to_correct}")
        return indices_to_correct
    
    def printStats(self):
        self._perf_stats.printStats()
    
def printResults(result_list):
    # a list of result arrays
    i = 0
    print(f"Printing results array of length {len(result_list)}")
    for result in result_list:
        print(i)
        print(result)
        i += 1

def compareArrays(array1, array2, number_of_precision_points=8, debug=False):
    diff_array = abs(array1 - array2)
    if debug:
        print(f"array1: \n{array1}")
        print(f"array2: \n{array2}")
        print(f"diff_array: \n{diff_array}")

    for precision in range(1, number_of_precision_points):
        precision_value = 1 / (10**precision)
        number_of_entires_with_error = len(
            np.argwhere(abs(diff_array) > precision_value)
        )
        # print(f"1e-{precision}: {precision_value} {number_of_entires_with_error}")
        print(f"{precision_value}: {number_of_entires_with_error}")

def main():
    # Initializtion
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_union", 
                        help = "Set this argument 0 if you do not want to use union. Otherwise, set it to 1", 
                        nargs="?",
                        default="0",
                        const="",
                        type=int,
                        )
    parser.add_argument("--grid_size", 
                        help = "Size of the 2D grid", 
                        nargs="?",
                        default=101,
                        const="",
                        type=int,
                        )    
    parser.add_argument("--generate_big_delta", 
                        help = "Set this argument to "
                        "0 if you want to use the constant big_delta specified by the --big_delta arguments."
                        "1 if you want to use new_theory for generating big_delta (described in paper) ."
                        "2 if you want to generate big_delta for each time_step (old logic).", 
                        nargs="?",
                        default=0,
                        const="",
                        type=int,
                        )
    parser.add_argument("--big_delta", 
                        help = "The value of big_delta to be used for finding the approximated leaking corners. ", 
                        nargs="?",
                        default=0.2,
                        type=float,
                        const="",
                        )    
    args = parser.parse_args()
    use_union: bool = bool(args.use_union)
    grid_size: int = int(args.grid_size)
    generate_big_delta: int = int(args.generate_big_delta)
    big_delta: float = float(args.big_delta)
    print(f"use_union: {use_union}")
    print(f"generate_big_delta: {generate_big_delta}")
    print(f"big_delta: {big_delta}")

    config = initConfig(use_union=use_union, grid_size=grid_size, generate_big_delta=generate_big_delta, big_delta=big_delta)
    print(f"Grid size: {grid_size} x {grid_size}")
    print(f"Number of time_steps: {int(config._lookback_length/config._time_step)}")
    print(f"Threshold2: {config._threshold2}")
    print(f"use_optimization1: {config._use_optimization1}")

    # Perform direct computation and decomposition
    grid, result_true = direct_computation(
        config=config,
        saveAllTimeStep=True,
        # lookback_length=config._lookback_length,
    )
    # printResults(result_list=result_true)
    # direct_comp_old(
    #     num=config._number_of_grid_points,
    #     saveAllTimeStep=True,
    #     lookback_length=config._lookback_length,
    # )
    # result_decomp_old = decomposition_old(
    #     config._number_of_grid_points,
    #     saveAllTimeStep=True,
    #     lookback_length=config._lookback_length,
    # )
    decomposition_result: DecompositionResult = decomposition(
        config=config,
        saveAllTimeStep=True,
    )
    # Initialize the value function
    # data_init = ShapeRectangle(grid, [-1.0, -1.0], [1.0, 1.0])
    data_init = config.value_function_2d(grid=grid)

    direct_computation_final = result_true[-1]
    decomp_final = decomposition_result.combined()[-1]
    # decomp_final_old = result_decomp_old[-1]

    print("comparing decomposition_old and decomposition")
    count = 0
    # for x, y in zip(result_true, decomposition_result.combined()):
    #     # result_diff2 = decomp_final - decomp_final_old
    #     print(f"Timestep: {count}")
    #     count += 1
    #     compareArrays(x, y)
    #     print("-----------------------------------")
    #     # exit(1)
    #     # plotArray(result_diff2)

    # print("comparing decomp_final and true_final")
    # compareArrays(decomp_final, true_final, debug=True)

    # correctionBasedOnDirectComputation(
    #     true_final=true_final,
    #     result_decomp=decomposition_result.combined(),
    #     config=config,
    # )
    decomposition_result_copy = copy.deepcopy(decomposition_result)

    local_update = CorrectionBasedOnLocalUpdate(
        decomposition_result=decomposition_result, 
        true_result=result_true,
        config=config,
    )
    start = time.time()
    local_update.doCorrection()
    correction_time = time.time() - start
    local_update.printStats()
    print(f"Total correction time: {round(correction_time,ndigits=4)} seconds")
    print(f"Total points identified: {local_update._perf_stats.totalPointsIdentified()}")
    print(f"Total points corrected: {local_update._perf_stats.totalPointsCorrected()}")

    printCorrectnessStatistics(
        direct_computation_results=result_true,
        decomposition_results=decomposition_result_copy,
        corrected_decomposition_results=decomposition_result,
    )

    # compareArrays(array1=decomposition_result.combined()[-1],
    #               array2=result_true[-1])
    
if __name__ == "__main__":
    main()
