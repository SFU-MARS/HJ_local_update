import argparse
import os

import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *

from dynamics.quadrotor_6d import quadrotor
from common.spatial_deriv import spa_deriv

# Plot Options
from odp.Plots import *
# Solver Core
from odp.solver import HJSolver

import time

from common.combine import check_combine
from compute.config_6d import add_compute_args_6d, grid_6d, init_values_6d, result_files_6d


def direct_comp(num, saveAllTimeStep=True, lookback_length=0.02, combine="union", t_step=0.02, sys=None):
    '''
        Direct computation of the 6D quadrotor, used as the ground truth. Pure Python loop over every
        grid point, so only small grids are practical (about 30 s for 9^6 with 5 time steps).
            combine: "union" (only union is implemented for 6d)
            sys: full system dynamics, e.g. from dynamics.make_dynamics; None uses quadrotor with
                 uTMax=1, utMax=1, uMode='max'
        Returns (grid, list of the value function at each time step).
    '''
    check_combine(combine, ("union",), "6d")

    # Create Grid
    g = grid_6d(num)

    ## Initialize value function: union of the half spaces x <= 0 and y <= 0
    data1, data2 = init_values_6d(num)
    data = np.minimum(data1, data2)

    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    if sys is None:
        sys = quadrotor(uTMax=1, utMax=1, dMax=0.0, uMode='max', dMode='min')


    '''
    Direct updating loop
    '''
    if saveAllTimeStep:
        data_list = []
        data_list.append(data)
        # print('The shape of data list is: ', len(data_list))

    list_x0 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x1 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x2 = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_x3 = np.reshape(g.vs[3], g.pts_each_dim[3])
    list_x4 = np.reshape(g.vs[4], g.pts_each_dim[4])
    list_x5 = np.reshape(g.vs[5], g.pts_each_dim[5])

    data_change = np.zeros(tuple(g.pts_each_dim))

    tNow = tau[0]

    start = time.time()

    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)

        # while tNow < tau[i]:
            # Update the value function

            
        for x0 in range(len(list_x0)):
            for x1 in range(len(list_x1)):
                for x2 in range(len(list_x2)):
                    for x3 in range(len(list_x3)):
                       for x4 in range(len(list_x4)):
                           for x5 in range(len(list_x5)): 

                                # Get spatial derivative
                                dV_d0_L, dV_d0_R = spa_deriv((x0, x1, x2, x3, x4, x5), data, g, 0, boundary="one_sided")
                                dV_d1_L, dV_d1_R = spa_deriv((x0, x1, x2, x3, x4, x5), data, g, 1, boundary="one_sided")
                                dV_d2_L, dV_d2_R = spa_deriv((x0, x1, x2, x3, x4, x5), data, g, 2, boundary="one_sided")
                                dV_d3_L, dV_d3_R = spa_deriv((x0, x1, x2, x3, x4, x5), data, g, 3, boundary="one_sided")
                                dV_d4_L, dV_d4_R = spa_deriv((x0, x1, x2, x3, x4, x5), data, g, 4, boundary="one_sided")
                                dV_d5_L, dV_d5_R = spa_deriv((x0, x1, x2, x3, x4, x5), data, g, 5, boundary="one_sided")
                                
                                # Get the average gradient
                                dV_d0 = (dV_d0_L + dV_d0_R)/2
                                dV_d1 = (dV_d1_L + dV_d1_R)/2
                                dV_d2 = (dV_d2_L + dV_d2_R)/2
                                dV_d3 = (dV_d3_L + dV_d3_R)/2
                                dV_d4 = (dV_d4_L + dV_d4_R)/2
                                dV_d5 = (dV_d5_L + dV_d5_R)/2
                                
                                # Get the dynamical rates of change
                                uTOpt, utOpt = sys.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3], list_x4[x4], list_x5[x5]], [dV_d0, dV_d1, dV_d2, dV_d3, dV_d4, dV_d5])
                                d0_dt, d1_dt, d2_dt, d3_dt, d4_dt, d5_dt = sys.dynamics_numpy(t,  [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3], list_x4[x4], list_x5[x5]], [uTOpt, utOpt], 0)
                                
                                # Updating the value function
                                data_change[x0, x1, x2, x3, x4, x5] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3 + d4_dt*dV_d4 + d5_dt*dV_d5
                
        data = data + data_change*t_step
        if saveAllTimeStep:
            data_list.append(data)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow += t_step
        # print('The shape of data list is: ', len(data_list))
    execution_time = time.time() - start

    # print('The shape of data list is: ', data_list.shape)
        
    print("Total kernel time direct: ", execution_time)
    print("Finished updating the value function")
    
    if saveAllTimeStep:
        return g, data_list

    return g, data

def main():
    '''
        Run the 6d direct computation and save it to results/result_direct_6d_<combine>.npy,
        shape (x, y, vx, vy, theta, w, t).
    '''
    from dynamics import make_dynamics
    parser = argparse.ArgumentParser()
    add_compute_args_6d(parser)
    args = parser.parse_args()
    try:
        check_combine(args.combine, ("union",), "6d")
        sys = make_dynamics(args.dynamics, args.dynamics_params, args.combine, dim=6)
    except (ValueError, TypeError, NotImplementedError) as e:
        parser.error(str(e))
    print(f"Dynamics: {args.dynamics} {vars(sys)}")
    _, data_list = direct_comp(args.grid_size, saveAllTimeStep=True, lookback_length=args.lookback_length,
                               combine=args.combine, t_step=args.t_step, sys=sys)
    os.makedirs(args.results_dir, exist_ok=True)
    path = result_files_6d(args.combine, args.results_dir)["direct"]
    np.save(path, np.moveaxis(np.asarray(data_list), 0, -1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

    
    

