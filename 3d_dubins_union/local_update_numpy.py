import argparse

import numpy as np
import math
import os
import sys as py_sys

from odp.Grid import Grid
from odp.Shapes import ShapeRectangle

from system import dubins_car

# The function of computing first order spatial derivative
from update_V_numpy import spa_deriv3X1, spa_deriv3X2, spa_deriv3X3

import time

np.set_printoptions(threshold=py_sys.maxsize)
np.set_printoptions(precision=5)


class Frontier:
    '''
        Boolean mask over the 3D grid, used for the frontier and visited sets
    '''
    def __init__(self, shape):
        self._shape = tuple(shape)
        self.grid = np.zeros(self._shape, dtype=bool)

    def add(self, index):
        self.grid[tuple(index)] = True

    def reset(self):
        self.grid = np.zeros(self._shape, dtype=bool)

    def hasIndex(self, index):
        return bool(self.grid[tuple(index)])

    def isEmpty(self):
        return not self.grid.any()

    def activeIndices(self):
        return np.argwhere(self.grid)

    def count(self):
        return int(self.grid.sum())


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
        "total_time"
        )

    def printStats(self):
        print(f"{self._iteration}, "
        f"{self._num_of_indices_to_correct}, "
        f"{self._total_indices_corrected}, "
        f"{round(self._total_time, ndigits=4)}"
        )


class PerfStats:
    def __init__(self):
        self._iter_stats: list[IterStats] = []

    def addIterStat(self, iter_stat: IterStats):
        self._iter_stats.append(iter_stat)

    def printStats(self):
        self._iter_stats[0].printHeader()
        for iter_stat in self._iter_stats:
            iter_stat.printStats()

    def totalPointsCorrected(self):
        return sum(iter_stat._total_indices_corrected for iter_stat in self._iter_stats)

    def totalPointsIdentified(self):
        return sum(iter_stat._num_of_indices_to_correct for iter_stat in self._iter_stats)


def recomputeValueChange(data, x, y, z, grid, sys, t, t_step):
    # recompute value change at index data[x, y, z] using the value function data

    # Get the left and right spatial derivatives (1st order)
    dV_dx_L, dV_dx_R = spa_deriv3X1(x, y, z, data, grid)
    dV_dy_L, dV_dy_R = spa_deriv3X2(x, y, z, data, grid)
    dV_dT_L, dV_dT_R = spa_deriv3X3(x, y, z, data, grid)

    # Get the average gradient
    dV_dx = (dV_dx_L + dV_dx_R) / 2
    dV_dy = (dV_dy_L + dV_dy_R) / 2
    dV_dT = (dV_dT_L + dV_dT_R) / 2

    # Get the dynamical rates of change
    state = [grid.grid_points[0][x], grid.grid_points[1][y], grid.grid_points[2][z]]
    uOpt = sys.opt_ctrl_numpy(t, state, [dV_dx, dV_dy, dV_dT])
    dx_dt, dy_dt, dT_dt = sys.dynamics_numpy(t, state, uOpt, 0)

    # Lax-Friedrichs dissipation, alpha is the max rate of change over the control set,
    # same as optimized_dp
    rates_1 = sys.dynamics_numpy(t, state, (sys.wMax,), 0)
    rates_2 = sys.dynamics_numpy(t, state, (-sys.wMax,), 0)
    alpha_x, alpha_y, alpha_T = np.maximum(np.abs(rates_1), np.abs(rates_2))
    diss = 0.5 * ((dV_dx_R - dV_dx_L) * alpha_x + (dV_dy_R - dV_dy_L) * alpha_y + (dV_dT_R - dV_dT_L) * alpha_T)

    # Updating the value function
    data_change = dx_dt * dV_dx + dy_dt * dV_dy + dT_dt * dV_dT + diss
    return data_change * t_step


class CorrectionBasedOnLocalUpdate:
    def __init__(
        self,
        result_decomp,
        result_upper,
        result_lower,
        grid,
        sys,
        tau,
        t_step,
        generate_big_delta=2,
        big_delta=0.2,
        threshold2=1e-6,
    ):
        '''
            result_decomp, result_upper, result_lower: arrays of shape (x, y, theta, t),
            index 0 of the last axis is t = 0. result_decomp is corrected in place.
        '''
        self._result_decomp = result_decomp
        self._result_upper = result_upper
        self._result_lower = result_lower
        self._grid = grid
        self._sys = sys
        self._tau = tau
        self._t_step = t_step
        self._generate_big_delta = generate_big_delta
        self._big_delta = big_delta
        self._threshold2 = threshold2

        self._perf_stats = PerfStats()

    def recomputeValueChange(self, data, x, y, z, t):
        return recomputeValueChange(data, x, y, z, self._grid, self._sys, t, self._t_step)

    def updateValue(self, curr_result, prev_result, index, frontier: Frontier, visited: Frontier, t) -> bool:
        x, y, z = index
        new_value = prev_result[x, y, z] + self.recomputeValueChange(data=prev_result, x=x, y=y, z=z, t=t)
        old_value = curr_result[x, y, z]
        curr_result[x, y, z] = new_value

        # mark all updated vertices as visited
        visited.add(index)

        flag = False
        if abs(old_value - new_value) > self._threshold2:
            flag = True
            for ngh in self.neighbours(index):
                if visited.hasIndex(ngh) is False:
                    frontier.add(ngh)
        return flag

    def neighbours(self, index):
        # Neighbours along each dimension, periodic dimensions wrap around
        nghs = []
        for dim in range(self._grid.dims):
            n = self._grid.pts_each_dim[dim]
            for step in (-1, 1):
                k = index[dim] + step
                if dim in self._grid.pDim:
                    k = k % n
                elif k < 0 or k >= n:
                    continue
                ngh = list(index)
                ngh[dim] = k
                nghs.append(tuple(ngh))
        return nghs

    def getBigDelta(self, curr_result_combined, prev_result_combined):
        if self._generate_big_delta == 0:
            # Use a constant big_delta defined in the command-line
            return self._big_delta
        elif self._generate_big_delta == 1:
            # Use the initial and final decomposition result
            return np.max(abs(self._result_decomp[..., -1] - self._result_decomp[..., 0]))
        elif self._generate_big_delta == 2:
            # Use the previous and current decomposition result, different big_delta for each time step
            return np.max(abs(curr_result_combined - prev_result_combined))

    def getNewPointsToCorrect(self, result_upper, result_lower, bigDelta):
        result_diff = abs(result_upper - result_lower)
        return np.argwhere(result_diff < bigDelta)

    def doCorrection(self, debug=False):
        shape = self._grid.pts_each_dim
        frontier = Frontier(shape)
        next_frontier = Frontier(shape)
        visited = Frontier(shape)

        start_time = time.time()

        for i in range(1, len(self._tau)):
            ###### INIT FOR EACH TIME STEP ######
            t = np.array([self._tau[i-1], self._tau[i]])
            curr_result_upper = self._result_upper[..., i]
            curr_result_lower = self._result_lower[..., i]
            curr_result = self._result_decomp[..., i].copy()
            prev_result = self._result_decomp[..., i-1]
            total_indices_corrected = 0
            visited.reset()

            # Compute threshold big delta in Algorithm 1
            bigDelta = self.getBigDelta(curr_result, prev_result)
            if debug:
                print(f"For iteration:{i}, big_delta:{bigDelta}")

            curr_time = time.time()
            init_time = curr_time - start_time
            start_time = curr_time

            ###### IDENTIFY INDICES TO CORRECT ######
            new_indices_to_correct = self.getNewPointsToCorrect(curr_result_upper, curr_result_lower, bigDelta)

            curr_time = time.time()
            identification_time = curr_time - start_time
            start_time = curr_time

            ###### CORRECTION 1: CORRECT THE IDENTIFIED INDICES ######
            for index in new_indices_to_correct:
                self.updateValue(curr_result, prev_result, tuple(index), frontier, visited, t)
                total_indices_corrected += 1

            curr_time = time.time()
            correction1_time = curr_time - start_time
            start_time = curr_time

            ###### CORRECTION 2: RECURSIVELY CORRECT THE NGHS OF IDENTIFIED INDICES ######
            while frontier.isEmpty() is False:
                for index in frontier.activeIndices():
                    index = tuple(index)
                    # Update frontier vertex and check if its nghs need to be added to the next_frontier
                    if visited.hasIndex(index) is False:
                        self.updateValue(curr_result, prev_result, index, next_frontier, visited, t)
                        total_indices_corrected += 1

                frontier, next_frontier = next_frontier, frontier
                next_frontier.reset()

            self._result_decomp[..., i] = curr_result

            curr_time = time.time()
            correction2_time = curr_time - start_time
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

    def printPerfStats(self):
        print("")
        print("Printing performance stats per iteration")
        self._perf_stats.printStats()


def dilate(mask, grid):
    '''
        Add the neighbours of every True point along each dimension (the stencil of the
        first order scheme). Periodic dimensions wrap around.
    '''
    out = mask.copy()
    for dim in range(mask.ndim):
        if dim in grid.pDim:
            out |= np.roll(mask, 1, axis=dim) | np.roll(mask, -1, axis=dim)
        else:
            lo = [slice(None)] * mask.ndim
            hi = [slice(None)] * mask.ndim
            lo[dim] = slice(None, -1)
            hi[dim] = slice(1, None)
            out[tuple(hi)] |= mask[tuple(lo)]
            out[tuple(lo)] |= mask[tuple(hi)]
    return out


def local_update_from_file(decomp_file, data_init_upper, data_init_lower, true_file, grid, sys, t_step,
                           delta=None, threshold=1e-6):
    '''
        Local updating based on a saved decomposition result
            decomp_file: npy file of the decomposition result, shape (x, y, theta, t), index 0 of the last axis is t = 0
            data_init_upper, data_init_lower: initial value function of subsystem 1 and 2 on the 3D grid,
                the initial value function is min(data_init_upper, data_init_lower)
            true_file: npy file of the direct computation result, same shape as decomp_file
            delta: threshold for finding the leaking corners, by default max|data_init - final decomposition|
            threshold: a point counts as corrected if it differs from the decomposition by more than this

        The update region grows over time: at each time step, the points updated are the leaking
        corners plus every point (and its neighbours) whose value at the previous time step differs
        from the decomposition result. Every other point keeps the decomposition value.
    '''
    # Step 1: load the decomposition and direct results, build the full initial value function
    result_decomp = np.load(decomp_file)
    result_true = np.load(true_file)
    data_init = np.minimum(data_init_upper, data_init_lower)

    # Step 2: find the leaking corners, where the two subsystem values are close (|V1 - V2| < delta)
    # and the decomposition min(V1, V2) can be wrong (new_theory_L in 6d_union)
    if delta is None:
        delta = np.max(abs(data_init - result_decomp[..., -1]))
    print('The delta value is', delta)
    leaking = abs(data_init_upper - data_init_lower) < delta
    print(f"The number of leaking corner points: {leaking.sum()} / {data_init.size}")

    # Step 3: local updating loop over time steps (pure numpy version)
    # data holds the corrected value function at the previous time step, starting from t = 0
    result_local = result_decomp.copy()
    data = data_init.astype(np.float64)
    total_updated = 0

    start = time.time()

    for i in range(1, result_decomp.shape[-1]):
        t = np.array([(i-1)*t_step, i*t_step])

        # Step 3a: find the points corrected at the previous time step (different from the decomposition)
        corrected = abs(data - result_decomp[..., i-1]) > threshold

        # Step 3b: the update region is the leaking corners plus the corrected points and their
        # neighbours, since the first order scheme at a point reads its neighbours' previous values
        region = dilate(corrected, grid) | leaking

        # Step 3c: start from the decomposition result, then recompute one Euler step inside the region
        new_data = result_decomp[..., i].astype(np.float64)
        for x, y, z in np.argwhere(region):
            new_data[x, y, z] = data[x, y, z] + recomputeValueChange(data, x, y, z, grid, sys, t, t_step)
        data = new_data

        # Step 3d: save the corrected value function of this time step
        result_local[..., i] = data
        total_updated += int(region.sum())

    execution_time = time.time() - start
    print(f"Total correction time: {execution_time:.4f} seconds")
    print(f"Total point updates: {total_updated} / {data_init.size * (result_decomp.shape[-1] - 1)}")

    # Step 4: compare the decomposition and local update results with the direct computation
    printCorrectnessStatistics(result_true, result_decomp, result_local)

    return result_local


def compareArrays(array1, array2, number_of_precision_points=8):
    diff_array = abs(array1 - array2)
    for precision in range(1, number_of_precision_points):
        precision_value = 1 / (10**precision)
        number_of_entires_with_error = len(np.argwhere(diff_array > precision_value))
        print(f"{precision_value}: {number_of_entires_with_error}")


def printCorrectnessStatistics(result_true, result_decomp, result_local):
    print("Processing stats")
    print("================================================")

    '''
    Comparision with direct computation at the final time step
    '''
    true_final = result_true[..., -1]
    decomp_final = result_decomp[..., -1]
    local_final = result_local[..., -1]

    total_num_of_points = true_final.size
    print(f"The total number of points: {total_num_of_points}")
    print(f"Decomposition method: Number of points with different values (1e-6): {np.argwhere(abs(decomp_final - true_final) > 1e-6).shape}")
    print(f"After local update, Number of points with different values (1e-6): {np.argwhere(abs(local_final - true_final) > 1e-6).shape}")

    print(f"The average absolute error between direct computation and decomposition: {np.sum(abs(decomp_final - true_final)) / total_num_of_points:0.10f}")
    print(f"The average absolute error between direct computation and local updates: {np.sum(abs(local_final - true_final)) / total_num_of_points:0.10f}")
    print(f"The maximum absolute error between direct computation and decomposition: {np.max(abs(decomp_final - true_final)):0.10f}")
    print(f"The maximum absolute error between direct computation and local updates: {np.max(abs(local_final - true_final)):0.10f}")

    # Zero level set comparison
    print(f"Points with different sign (set mismatch), decomposition: {np.sum((decomp_final < 0) != (true_final < 0))}")
    print(f"Points with different sign (set mismatch), local update: {np.sum((local_final < 0) != (true_final < 0))}")

    print("Number of values with error b/w direct computation and decomposition:")
    compareArrays(true_final, decomp_final)
    print("Number of values with error b/w direct computation and local update:")
    compareArrays(true_final, local_final)


def main():
    # Step 1: parse and check the command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp_file", help="npy file of the decomposition result, shape (x, y, theta, t).", default=None)
    parser.add_argument("--true_file", help="npy file of the direct computation result, same shape as --decomp_file.", default=None)
    parser.add_argument("--t_step", help="Time step used to compute both files. Keep it below the CFL bound of HJSolver "
                        "(about 0.048 for a 41^3 grid) so that HJSolver takes exactly one Euler step per time step.",
                        default=0.04, type=float)
    args = parser.parse_args()

    missing = [name for name, value in (("--decomp_file", args.decomp_file), ("--true_file", args.true_file)) if value is None]
    if missing:
        parser.error(f"need file inputs {' and '.join(missing)} (e.g. --decomp_file results/result_decomp_3d.npy "
                     "--true_file results/result_direct_3d.npy)")
    for path in (args.decomp_file, args.true_file):
        if not os.path.isfile(path):
            parser.error(f"file not found: {path}")

    # Step 2: set the dynamics, same as the direct computation, control tries to avoid the obstacle
    sys = dubins_car(v=1, wMax=1, dMax=0.0, uMode='max', dMode='min')

    # Step 3: build the 3D grid with the grid size read from the decomposition file, theta is periodic
    num = np.load(args.decomp_file, mmap_mode='r').shape[0]
    grid = Grid(np.array([-4.0, -4.0, -math.pi]), np.array([4.0, 4.0, math.pi]), 3, np.array([num, num, num]), [2])

    # Step 4: initial value function of the subsystems, same obstacle as decomposition.py
    data_init_upper = ShapeRectangle(grid, np.array([-0.5, -np.inf, -np.inf]), np.array([0.5, np.inf, np.inf]))
    data_init_lower = ShapeRectangle(grid, np.array([-np.inf, -0.5, -np.inf]), np.array([np.inf, 0.5, np.inf]))

    # Step 5: run the local update and save the corrected result
    result_local = local_update_from_file(args.decomp_file, data_init_upper, data_init_lower, args.true_file,
                                          grid, sys, args.t_step)
    np.save('results/result_local_3d.npy', result_local)


if __name__ == "__main__":
    if os.path.exists("results") == False:
        os.mkdir("results")
    main()
