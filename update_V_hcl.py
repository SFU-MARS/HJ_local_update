import heterocl as hcl
from odp.spatialDerivatives.firstOrderENO.first_orderENO2D import *
import numpy as np


def update_V(system, grid, ind):
    # hcl.init()

    V_f = hcl.placeholder(tuple(grid.pts_each_dim), name="V_f", dtype=hcl.Float())
    V_init = hcl.placeholder(tuple(grid.pts_each_dim), name="V_init", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())
    indices = hcl.placeholder((tuple(ind.shape)), name="indices")
    # Positions vector
    x1 = hcl.placeholder((grid.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((grid.pts_each_dim[1],), name="x2", dtype=hcl.Float())


    def graph_create(V_new, V_init, x1, x2, indices, t):
        # hcl.init()

        # Time interval
        delta_t = hcl.scalar(0, "delta_t")
        delta_t[0] = t[1] - t[0]
        
        with hcl.Stage("Hamiltonian"):
            with hcl.for_(1, indices.shape[0]) as i:
                    # Variables to calculate dV_dx
                    dV_dx_L = hcl.scalar(0, "dV_dx_L")
                    dV_dx_R = hcl.scalar(0, "dV_dx_R")
                    dV_dx = hcl.scalar(0, "dV_dx")
                    # Variables to calculate dV_dy
                    dV_dy_L = hcl.scalar(0, "dV_dy_L")
                    dV_dy_R = hcl.scalar(0, "dV_dy_R")
                    dV_dy = hcl.scalar(0, "dV_dy")

                    # Indice variables
                    i1 = hcl.scalar(0, "i1")
                    i1[0] = indices[i,0]
                    i2 = hcl.scalar(0, "i2")
                    i2[0] = indices[i,1]

                    # Get the spatial derivative
                    dV_dx_L[0], dV_dx_R[0] = spa_derivX(i1, i2, V_init, grid)
                    dV_dy_L[0], dV_dy_R[0] = spa_derivY(i1, i2, V_init, grid)
                    # Calculate average gradient
                    dV_dx[0] = (dV_dx_L + dV_dx_R) / 2
                    dV_dy[0] = (dV_dy_L + dV_dy_R) / 2
                    # Calculate dynamical rates of changes
                    uOpt = system.opt_ctrl(0, (x1[i1], x2[i2]), (dV_dx[0], dV_dy[0])) # For thie particular case, the state is not used
                    # Updating the value
                    dx_dt, dy_dt = system.dynamics(0, (x1[i1], x2[i2]), uOpt, 0)
                    V_new[i1, i2] = -(dx_dt * dV_dx[0] + dy_dt * dV_dy[0])

        # Ignore the dissipation term for now    
        result = hcl.update(V_new, lambda i, j: V_init[i, j] + V_new[i, j] * delta_t[0])
                    
        # Copy V_new to V_init
        hcl.update(V_init, lambda i, j: V_new[i, j])
        return result
    
    s = hcl.create_schedule([V_f, V_init, x1, x2, indices, t], graph_create)
    print('optimizing\n')
    s_H = graph_create.Hamiltonian
    s[s_H].parallel(s_H.i)
    return (hcl.build(s))