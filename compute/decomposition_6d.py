import argparse

import numpy as np
import math

from odp.Grid import Grid
from odp.Shapes import *

from subdynamics.quadrotor_6d import subsys_1, subsys_2
from common.spatial_deriv import spa_deriv

# Plot Options
from odp.Plots import *
import time
import os

from common.combine import check_combine
from dynamics import U_MODE
from compute.config_6d import add_compute_args_6d, grid_sub_6d, result_files_6d

def decomposition(num, saveAllTimeStep=True, lookback_length=0.02, combine="union", subsystems=None, t_step=0.02):
    '''
        Decomposition of the 6D quadrotor into two 4D subsystems, subsystem 1 (x, vx, theta, w) and
        subsystem 2 (y, vy, theta, w), combined as min(V1, V2) (union) or max(V1, V2) (intersection).
            combine: "union" (avoid, uMode max) or "intersection" (reach, uMode min)
            subsystems: (subsystem 1, subsystem 2) dynamics, e.g. from dynamics.make_subdynamics;
                None uses subsys_1, subsys_2 with uTMax=1, utMax=1, uMode from combine
        Returns (V, V1, V2) on the 6D grid, time as the first axis.
    '''
    check_combine(combine, ("union", "intersection"), "6d")
    # Create Grid of the subsystems
    g = grid_sub_6d(num)
    
    # Initialize value function
    data_sub_1_1 = Lower_Half_Space(g, 0, 0)
    data_sub_1_2 = Lower_Half_Space(g, 2, -1)
    data_sub_1_3 = Upper_Half_Space(g, 2, 1)
    data_sub_1 = data_sub_1_1
    # data_sub_1 = np.minimum(data_sub_1, data_sub_1_2)
    # data_sub_1 = np.minimum(data_sub_1, data_sub_1_3)
    data_sub_2 = data_sub_1.copy()
    
    # Visualization
    po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0],
                    colorscale="Bluered", save_fig=True, filename="direct_2Int_leaking", interactive_html=True)
    
    
    ## Look-back length and time step of computation
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)
    
    # Set system dynamics
    if subsystems is None:
        sys_1 = subsys_1(uTMax=1, utMax=1, dMax=0.0, uMode=U_MODE[combine], dMode='min')
        sys_2 = subsys_2(uTMax=1, utMax=1, dMax=0.0, uMode=U_MODE[combine], dMode='min')
    else:
        sys_1, sys_2 = subsystems
    
    list_x0 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x1 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x2 = np.reshape(g.vs[2], g.pts_each_dim[2])
    list_x3 = np.reshape(g.vs[3], g.pts_each_dim[3])
    
    data_change = np.zeros(tuple(g.pts_each_dim))
    
    # Set computation task and compute HJ PDE
    '''
    Computation in subsystem 1
    '''
    if saveAllTimeStep:
        data_list_1 = []
        data_list_1.append(data_sub_1.copy())      
        # print('The shape of data list is: ', data_list)
        
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
                        # Get spatial derivative
                        dV_d0_L, dV_d0_R = spa_deriv((x0, x1, x2, x3), data_sub_1, g, 0, boundary="one_sided")
                        dV_d1_L, dV_d1_R = spa_deriv((x0, x1, x2, x3), data_sub_1, g, 1, boundary="one_sided")
                        dV_d2_L, dV_d2_R = spa_deriv((x0, x1, x2, x3), data_sub_1, g, 2, boundary="one_sided")
                        dV_d3_L, dV_d3_R = spa_deriv((x0, x1, x2, x3), data_sub_1, g, 3, boundary="one_sided")
                        
                        # Get the average gradient
                        dV_d0 = (dV_d0_L + dV_d0_R)/2
                        dV_d1 = (dV_d1_L + dV_d1_R)/2
                        dV_d2 = (dV_d2_L + dV_d2_R)/2
                        dV_d3 = (dV_d3_L + dV_d3_R)/2
                        
                        # Get the dynamical rates of change
                        uTOpt, utOpt = sys_1.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [dV_d0, dV_d1, dV_d2, dV_d3])
                        d0_dt, d1_dt, d2_dt, d3_dt = sys_1.dynamics_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [uTOpt, utOpt], 0)
                        
                        # Updating the value function
                        data_change[x0, x1, x2, x3] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3
                        
        data_sub_1 = data_sub_1 + t_step * data_change
        if saveAllTimeStep:
            data_list_1.append(data_sub_1)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow = tNow + t_step
        
    '''
    Computation in subsystem 2
    '''
    if saveAllTimeStep:
        data_list_2 = []
        data_list_2.append(data_sub_2.copy())
        # print('The shape of data list is: ', data_list)
        
    tNow = tau[0]
    # start = time.time()
    
    for i in range(1, len(tau)):
        t = np.array([tNow, tau[i]])
        print("Time step: ", t)
        
        # while tNow < tau[i]:
            # Update the value function
        for x0 in range(len(list_x0)):
            for x1 in range(len(list_x1)):
                for x2 in range(len(list_x2)):
                    for x3 in range(len(list_x3)):
                        
                        # Get spatial derivative
                        dV_d0_L, dV_d0_R = spa_deriv((x0, x1, x2, x3), data_sub_2, g, 0, boundary="one_sided")
                        dV_d1_L, dV_d1_R = spa_deriv((x0, x1, x2, x3), data_sub_2, g, 1, boundary="one_sided")
                        dV_d2_L, dV_d2_R = spa_deriv((x0, x1, x2, x3), data_sub_2, g, 2, boundary="one_sided")
                        dV_d3_L, dV_d3_R = spa_deriv((x0, x1, x2, x3), data_sub_2, g, 3, boundary="one_sided")
                        
                        # Get the average gradient
                        dV_d0 = (dV_d0_L + dV_d0_R)/2
                        dV_d1 = (dV_d1_L + dV_d1_R)/2
                        dV_d2 = (dV_d2_L + dV_d2_R)/2
                        dV_d3 = (dV_d3_L + dV_d3_R)/2
                        
                        # Get the dynamical rates of change
                        uTOpt, utOpt = sys_2.opt_ctrl_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [dV_d0, dV_d1, dV_d2, dV_d3])
                        d0_dt, d1_dt, d2_dt, d3_dt = sys_2.dynamics_numpy(t, [list_x0[x0], list_x1[x1], list_x2[x2], list_x3[x3]], [uTOpt, utOpt], 0)
                        
                        # Updating the value function
                        data_change[x0, x1, x2, x3] = d0_dt*dV_d0 + d1_dt*dV_d1 + d2_dt*dV_d2 + d3_dt*dV_d3
                        
        data_sub_2 = data_sub_2 + t_step * data_change
        if saveAllTimeStep:
            data_list_2.append(data_sub_2)
        data_change = np.zeros(tuple(g.pts_each_dim))
        tNow = tNow + t_step
            
    # execution_time = time.time() - start
    
    # print("Total kernel time: ", execution_time)
    # print("Finished updating the value function")
    
    
    '''
    Combine the results from 2 subsystems
    '''
        
    if not saveAllTimeStep:
        result_upper_flip = np.flip(data_sub_1, axis=0) 
        result_upper_expand = np.tile(result_upper_flip, (num, num, 1, 1, 1, 1))
        print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (2, 0, 3, 1, 4, 5))

        result_lower_flip = data_sub_2
        result_lower_expand = np.tile(result_lower_flip, (num, num, 1, 1, 1, 1))
        print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (0,2,1,3,4,5))

        result_full = np.minimum(result_upper, result_lower) if combine == "union" else np.maximum(result_upper, result_lower)
        
    else:
        
        # result_upper_flip = np.flip(data_list_1, axis=0) 
        result_upper_expand = np.tile(data_list_1, (num, num, 1, 1, 1, 1, 1))
        # print(result_upper_expand.shape)
        result_upper = np.transpose(result_upper_expand, (2, 3, 0, 4, 1, 5, 6))
        # print(result_upper.shape)

        # result_lower_flip = np.flip(data_list_2, axis=0)
        result_lower_expand = np.tile(data_list_2, (num, num, 1, 1, 1, 1, 1))
        # print(result_lower_expand.shape)
        result_lower = np.transpose(result_lower_expand, (2, 0, 3, 1, 4, 5, 6))
        # print(result_lower.shape)

        result_full = np.minimum(result_upper, result_lower) if combine == "union" else np.maximum(result_upper, result_lower)
        
        # print('Decomposition Correct')
        
    execution_time = time.time() - start
    
    print("Total kernel time decomposition: ", execution_time)
    print("Finished updating the value function")
    
    return result_full, result_upper, result_lower


def main():
    '''
        Run the 6d decomposition and save
            results/result_decomp_6d_<combine>.npy: V = min(V1, V2) or max(V1, V2), shape (x, y, vx, vy, theta, w, t)
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
    result_full, result_upper, result_lower = decomposition(
        args.grid_size, saveAllTimeStep=True, lookback_length=args.lookback_length, combine=args.combine,
        subsystems=subsystems, t_step=args.t_step)
    os.makedirs(args.results_dir, exist_ok=True)
    files = result_files_6d(args.combine, args.results_dir)
    np.save(files["decomp"], np.moveaxis(result_full, 0, -1))
    np.save(files["init_1"], result_upper[0])
    np.save(files["init_2"], result_lower[0])
    for key in ("decomp", "init_1", "init_2"):
        print(f"Saved {files[key]}")


if __name__ == "__main__":
    main()