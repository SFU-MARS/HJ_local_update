# One step of optimized_dp's HJSolver scheme (accuracy "low") at a single grid point, for the local updates:
# 1st order ENO derivatives (extrapolated at the grid edge), Hamiltonian with the optimal control,
# Lax-Friedrichs dissipation, forward Euler. numpy and HeteroCL versions.
#
# HJSolver takes the dissipation coefficients from the optimal control at the global min and max spatial
# derivatives (over the whole grid). For dynamics whose coefficients depend on them
# (DISSIPATION_USES_DERIV_BOUNDS, e.g. couple_u), the local update computes these bounds from the full
# value function of the previous time step, with deriv_bounds_numpy or deriv_bounds_hcl.
import heterocl as hcl
import numpy as np

from common.spatial_deriv import spa_deriv, spa_deriv_arrays, spa_deriv_hcl


def deriv_bounds_numpy(data, grid):
    '''
        Global min and max of the left and right spatial derivatives in each dimension.
        Returns (deriv_min, deriv_max), one value per dimension.
    '''
    deriv_min, deriv_max = [], []
    for dim in range(data.ndim):
        left, right = spa_deriv_arrays(data, grid, dim)
        deriv_min.append(min(left.min(), right.min()))
        deriv_max.append(max(left.max(), right.max()))
    return deriv_min, deriv_max


def value_change_numpy(data, index, grid, sys, t, t_step, state_values, deriv_min=None, deriv_max=None):
    '''
        Value change of one HJSolver step at index (numpy).
            state_values: list of the grid coordinates of each dimension
            deriv_min, deriv_max: from deriv_bounds_numpy, needed if sys.DISSIPATION_USES_DERIV_BOUNDS
    '''
    dims = len(index)
    left, right, grad = [], [], []
    for dim in range(dims):
        dV_L, dV_R = spa_deriv(index, data, grid, dim)
        left.append(dV_L)
        right.append(dV_R)
        grad.append((dV_L + dV_R)/2)

    state = [state_values[dim][index[dim]] for dim in range(dims)]
    uOpt = sys.opt_ctrl_numpy(t, state, grad)
    rates = sys.dynamics_numpy(t, state, uOpt, 0)
    alpha = sys.dissipation_coeffs_numpy(t, state, deriv_min, deriv_max)

    diss = (right[0] - left[0])*alpha[0]
    hamiltonian = rates[0]*grad[0]
    for dim in range(1, dims):
        diss = diss + (right[dim] - left[dim])*alpha[dim]
        hamiltonian = hamiltonian + rates[dim]*grad[dim]
    return (hamiltonian + 0.5*diss)*t_step


def for_all_points(shape, body, prefix="p"):
    '''
        Nested HeteroCL loops over every grid point, body(index) is called with the list of loop variables.
    '''
    def loop(dim, index):
        if dim == len(shape):
            body(index)
            return
        with hcl.for_(0, shape[dim], name=f"{prefix}{dim}") as i:
            loop(dim + 1, index + [i])
    loop(0, [])


def deriv_bounds_hcl(V, grid):
    '''
        Inside a HeteroCL kernel: global min and max of the left and right spatial derivatives of V in each
        dimension (sequential loops over the grid, in a stage named "Bounds"). Call it at the top level of the
        kernel, not inside a stage, so the returned scalars can be used by the following stages.
        Returns (deriv_min, deriv_max), lists of HeteroCL scalars.
    '''
    shape = tuple(int(n) for n in grid.pts_each_dim)
    deriv_min = [hcl.scalar(1e9, f"deriv_min{d}", dtype=V.dtype) for d in range(len(shape))]
    deriv_max = [hcl.scalar(-1e9, f"deriv_max{d}", dtype=V.dtype) for d in range(len(shape))]

    def body(index):
        for dim in range(len(shape)):
            dV_L, dV_R = spa_deriv_hcl(index, V, grid, dim)
            for value in (dV_L, dV_R):
                with hcl.if_(value < deriv_min[dim][0]):
                    deriv_min[dim][0] = value
                with hcl.if_(value > deriv_max[dim][0]):
                    deriv_max[dim][0] = value
    with hcl.Stage("Bounds"):
        for_all_points(shape, body, prefix="b")
    return deriv_min, deriv_max


def value_change_hcl(V, index, grid, sys, state, t_step, deriv_min=None, deriv_max=None):
    '''
        Inside a HeteroCL kernel: value change of one HJSolver step at index, same as value_change_numpy.
            state: tuple of the state coordinates at index
            t_step: HeteroCL expression of the time step
            deriv_min, deriv_max: lists of HeteroCL scalars from deriv_bounds_hcl, or None
        Returns a HeteroCL expression.
    '''
    dims = len(index)
    left, right, grad = [], [], []
    for dim in range(dims):
        dV_L, dV_R = spa_deriv_hcl(index, V, grid, dim)
        L = hcl.scalar(0, f"dV_L{dim}")
        R = hcl.scalar(0, f"dV_R{dim}")
        g = hcl.scalar(0, f"grad{dim}")
        L[0] = dV_L
        R[0] = dV_R
        g[0] = (L[0] + R[0]) / 2
        left.append(L[0])
        right.append(R[0])
        grad.append(g[0])

    uOpt = sys.opt_ctrl(0, state, tuple(grad))
    rates = sys.dynamics(0, state, uOpt, (0,) * 4)
    bounds = (None, None) if deriv_min is None else (tuple(d[0] for d in deriv_min), tuple(d[0] for d in deriv_max))
    alpha = sys.dissipation_coeffs(0, state, *bounds)

    diss = hcl.scalar(0, "diss")
    hamiltonian = hcl.scalar(0, "hamiltonian")
    diss[0] = (right[0] - left[0]) * alpha[0]
    hamiltonian[0] = rates[0] * grad[0]
    for dim in range(1, dims):
        diss[0] = diss[0] + (right[dim] - left[dim]) * alpha[dim]
        hamiltonian[0] = hamiltonian[0] + rates[dim] * grad[dim]
    change = hcl.scalar(0, "change")
    change[0] = (hamiltonian[0] + 0.5 * diss[0]) * t_step
    return change[0]
