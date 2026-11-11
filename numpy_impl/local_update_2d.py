import argparse
from compute.decomposition_2d import DecompositionResult

import numpy as np
import heterocl as hcl
import copy
import os
# import cv2
import sys as py_sys

# from common.plotting import *
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

import plotly.express as px

from odp.Grid import Grid
from odp.Shapes import *
from compute.config_2d import Config, construct2DGrid, Frontier, add_local_update_args_2d, setup_local_update_2d
from common.params import parse_args_with_params
from common.combine import check_combine
from common.stats import compareArrays, printCorrectnessStatistics2d

# The function of computing first order spatial derivative
from common.hj_update import deriv_bounds_numpy, value_change_numpy

import time

np.set_printoptions(threshold=py_sys.maxsize)
np.set_printoptions(precision=5)


def plotArray(array, show=True):
    fig = px.imshow(array)
    if show:
        fig.show()

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
        "total_indices_corrected, "
        # "init_time,"
        # "identification_time, "
        # "correction1_time, "
        # "correction2_time, "
        "total_time"
        )
    
    def printStats(self):
        print(f"{self._iteration}, "
        f"{self._num_of_indices_to_correct}, "
        f"{self._total_indices_corrected}, "
        # f"{round(self._init_time, ndigits=4)}, "
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
        self._deriv_min = self._deriv_max = None

        self._perf_stats = PerfStats(decomposition_result=decomposition_result,
                                     true_result=true_result,
                                     config=config,
                                     )

    def recomputeValueChange(self, data, x, y, t):
        # Value change of one HJSolver step at data[x][y] (see common/hj_update.py), from the value function
        # data of the previous time step. self._deriv_min / _deriv_max are the global derivative bounds of
        # data, set in doCorrection, used for the dissipation like HJSolver.
        return value_change_numpy(data, (x, y), self._grid, self._sys, t, self._config._time_step,
                                  (self._list_x1, self._list_x2), self._deriv_min, self._deriv_max)
    
    def updateValue(self, curr_result, prev_result, index, frontier:Frontier, visited: Frontier, time_step) -> bool:
        x = index[0]
        y = index[1]
        new_value_change = self.recomputeValueChange(data=prev_result, x=x, y=y, t=time_step,)
        old_value = curr_result[x][y]
        new_value = new_value_change + prev_result[x][y]
        curr_result[x][y] = new_value

        # mark all updated vertices as visited
        visited.add(index=index)

        flag = False
        if abs(old_value - new_value) > self.threshold2():
            flag = True
            prev_index_on_x = self.prevIndexOnX(index=index)
            next_index_on_x = self.nextIndexOnX(index=index)
            prev_index_on_y = self.prevIndexOnY(index=index)
            next_index_on_y = self.nextIndexOnY(index=index)
            if visited.hasIndex(index=prev_index_on_x) is False:
                frontier.add(prev_index_on_x)
            if visited.hasIndex(index=next_index_on_x) is False:
                frontier.add(next_index_on_x)
            if visited.hasIndex(index=prev_index_on_y) is False:
                frontier.add(prev_index_on_y)
            if visited.hasIndex(index=next_index_on_y) is False:
                frontier.add(next_index_on_y)
        return flag

    def doCorrection(self, debug=False):
        if debug:
            print("------------------------------------------------")
            print("START doCorrection")
        result_combined_all_timesteps = self._decomposition_result.combined()
        subsystem1_data_all_timesteps = self._decomposition_result.subsystem1()
        subsystem2_data_all_timesteps = self._decomposition_result.subsystem2()

        frontier = Frontier(x=self._config._number_of_grid_points,
                            y=self._config._number_of_grid_points)
        next_frontier = Frontier(x=self._config._number_of_grid_points,
                            y=self._config._number_of_grid_points)
        visited = Frontier(x=self._config._number_of_grid_points,
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
            # Global derivative bounds of the previous time step, for the dissipation like HJSolver
            self._deriv_min, self._deriv_max = (deriv_bounds_numpy(np.asarray(prev_result), self._grid)
                                                if self._sys.DISSIPATION_USES_DERIV_BOUNDS else (None, None))
            curr_result_true = self._true_result[i]
            total_indices_corrected = 0
            visited.reset()

            # Compute threshold big delta in Algorithm 1
            bigDelta = self.getBigDelta(
                curr_result_combined = curr_result, 
                prev_result_combined = prev_result, 
                decomp_result_init= result_combined_all_timesteps[0], 
                decomp_result_final = result_combined_all_timesteps[-1],)
            if debug:
                print(f"For iteration:{i}, big_delta:{bigDelta}")

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
                self.updateValue(curr_result=curr_result, prev_result=prev_result, index=index, frontier=frontier, visited=visited, time_step=i)
                total_indices_corrected += 1

            curr_time = time.time()
            correction1_time = time.time() - start_time
            start_time = curr_time

            ###### CORRECTION 2: RECURSIVELY CORRECT THE NGHS OF IDENTIFIED INDICES  ######

            while frontier.isEmpty() is False:
                for index in frontier.activeIndices():
                    # Update frontier vertex and check if its nghs need to be added to the next_frontier
                    if visited.hasIndex(index) is False:
                        self.updateValue(curr_result=curr_result, prev_result=prev_result, index=index, frontier=next_frontier, visited=visited, time_step=i)
                        total_indices_corrected += 1
                
                frontier, next_frontier = next_frontier, frontier
                next_frontier.reset()

            result_combined_all_timesteps[i] = curr_result

            # reset updated_points_curr for next iteration
            if self._config._use_optimization1:
                visited.reset()

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
            print("------------------------------------------------")
            print("END doCorrection")

    def getBigDelta(self, curr_result_combined, prev_result_combined, decomp_result_init, decomp_result_final):
        if self._config._generate_big_delta == 0:
            # Use a constant big_delta defined in the command-line
            # Useful for testing
            return self._config._big_delta
        elif self._config._generate_big_delta == 1:
            # here, we use the initial and final decomposition result to generate the data
            max_diff = np.max(abs(decomp_result_final - decomp_result_init))
            return max_diff
        elif self._config._generate_big_delta == 2:
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
        indices_to_correct = np.argwhere(result_diff < bigDelta)
        return indices_to_correct
    
    def printPerfStats(self):
        print("")
        print("Printing performance stats per iteration")
        self._perf_stats.printStats()
    
def printResults(result_list):
    # a list of result arrays
    i = 0
    print(f"Printing results array of length {len(result_list)}")
    for result in result_list:
        print(i)
        print(result)
        i += 1

def main():
    '''
        Read the saved 2d results (compute/direct_2d.py, compute/decomposition_2d.py), run the numpy local
        update, save it to results/result_local_2d_<combine>.npy (shape (x, y, t)) and compare it with the
        direct computation if --true_file is given.
    '''
    # Step 1: inputs, from the command line and/or the --params file
    parser = argparse.ArgumentParser()
    add_local_update_args_2d(parser)
    args = parse_args_with_params(parser)
    check_combine(args.combine, ("union", "intersection"), "2d")
    use_debug = bool(args.debug)
    config, decomposition_result, result_true = setup_local_update_2d(parser, args)

    # Step 2: local update (changes decomposition_result in place, so keep a copy for the comparison)
    decomposition_result_copy = copy.deepcopy(decomposition_result)
    local_update = CorrectionBasedOnLocalUpdate(
        decomposition_result=decomposition_result,
        true_result=result_true,
        config=config,
    )
    start = time.time()
    local_update.doCorrection(debug=use_debug,)
    correction_time = time.time() - start
    if use_debug:
        local_update.printPerfStats()
    print(f"Total correction time: {round(correction_time,ndigits=4)} seconds")
    print(f"Total points identified: {local_update._perf_stats.totalPointsIdentified()}")
    print(f"Total points corrected: {local_update._perf_stats.totalPointsCorrected()}")

    # Step 3: save the corrected result, time as the last axis
    os.makedirs("results", exist_ok=True)
    path = f"results/result_local_2d_{args.combine}.npy"
    np.save(path, np.moveaxis(np.asarray(decomposition_result.combined()), 0, -1))
    print(f"Saved {path}")

    # Step 4: compare with the direct computation (if given)
    print("")
    if result_true is not None:
        printCorrectnessStatistics2d(
            direct_computation_results=result_true,
            decomposition_results=decomposition_result_copy,
            corrected_decomposition_results=decomposition_result,
        )
    else:
        print("No --true_file given, skipping the comparison with the direct computation")


if __name__ == "__main__":
    main()
