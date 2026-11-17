# One step of optimized_dp's HJSolver scheme (accuracy "low") at a single grid point, for the local updates:
# 1st order ENO derivatives (extrapolated at the grid edge), Hamiltonian with the optimal control,
# Lax-Friedrichs dissipation, forward Euler. numpy and HeteroCL versions.
#
# Like HJSolver, every local update works in 32-bit floats: DTYPE (numpy) and DTYPE_HCL (HeteroCL).
# The numpy version rounds every intermediate result to 32 bits, since numpy turns a 32-bit scalar
# combined with a Python float into a 64-bit one.
#
# HJSolver takes the dissipation coefficients from the optimal control at the global min and max spatial
# derivatives (over the whole grid). For dynamics whose coefficients depend on them
# (DISSIPATION_USES_DERIV_BOUNDS, e.g. couple_u), the local update computes these bounds from the full
# value function of the previous time step, with deriv_bounds_numpy or deriv_bounds_hcl.
import heterocl as hcl
import numpy as np

from common.spatial_deriv import spa_deriv, spa_deriv_arrays, spa_deriv_hcl

DTYPE = np.float32
DTYPE_HCL = hcl.Float(32)


def abs_hcl(x):
    '''
        |x| inside a HeteroCL kernel, with the type of x (odp's my_abs goes through a scalar of the default type).
    '''
    return hcl.select(x > 0, x, -x)


def deriv_bounds_numpy(data, grid, dims=None):
    '''
        Global min and max of the left and right spatial derivatives in each dimension.
            dims: the dimensions to compute (default all); the others get 0, for dynamics whose dissipation
                  coefficients only depend on some of them (DERIV_BOUNDS_DIMS)
        Returns (deriv_min, deriv_max), one value per dimension.
    '''
    deriv_min, deriv_max = [], []
    for dim in range(data.ndim):
        if dims is not None and dim not in dims:
            deriv_min.append(DTYPE(0))
            deriv_max.append(DTYPE(0))
            continue
        left, right = spa_deriv_arrays(data, grid, dim)
        deriv_min.append(min(left.min(), right.min()))
        deriv_max.append(max(left.max(), right.max()))
    return deriv_min, deriv_max


def value_change_numpy(data, index, grid, sys, t, t_step, state_values, deriv_min=None, deriv_max=None):
    '''
        Value change of one HJSolver step at index (numpy).
            data: DTYPE array
            state_values: list of the grid coordinates of each dimension
            deriv_min, deriv_max: from deriv_bounds_numpy, needed if sys.DISSIPATION_USES_DERIV_BOUNDS
        Returns a DTYPE scalar.
    '''
    f = DTYPE
    dims = len(index)
    left, right, grad = [], [], []
    for dim in range(dims):
        dV_L, dV_R = spa_deriv(index, data, grid, dim)
        left.append(f(dV_L))
        right.append(f(dV_R))
        grad.append(f((left[-1] + right[-1])/f(2)))

    state = [f(state_values[dim][index[dim]]) for dim in range(dims)]
    uOpt = sys.opt_ctrl_numpy(t, state, grad)
    rates = [f(r) for r in sys.dynamics_numpy(t, state, uOpt, 0)]
    alpha = [f(a) for a in sys.dissipation_coeffs_numpy(t, state, deriv_min, deriv_max)]

    diss = f((right[0] - left[0])*alpha[0])
    hamiltonian = f(rates[0]*grad[0])
    for dim in range(1, dims):
        diss = f(diss + f((right[dim] - left[dim])*alpha[dim]))
        hamiltonian = f(hamiltonian + f(rates[dim]*grad[dim]))
    return f(f(hamiltonian + f(f(0.5)*diss))*f(t_step))


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


def deriv_bounds_hcl(V, grid, dims=None):
    '''
        Inside a HeteroCL kernel: global min and max of the left and right spatial derivatives of V in each
        dimension. Call it at the top level of the kernel, not inside a stage, so the returned scalars can be
        used by the following stages. Two stages:
            "Bounds" (loop b0 over the first grid dimension, which the kernel may run in parallel): the min and
                     max of each slice b0, in arrays indexed by b0, so the threads never write the same element
            "BoundsReduce" (sequential): the min and max over the slices
        Min and max do not depend on the order, so the result is the same as one sequential loop.
            dims: the dimensions to compute (default all); the others get 0, for dynamics whose dissipation
                  coefficients only depend on some of them (DERIV_BOUNDS_DIMS)
        Returns (deriv_min, deriv_max), lists of HeteroCL scalars.
    '''
    shape = tuple(int(n) for n in grid.pts_each_dim)
    dims = list(range(len(shape))) if dims is None else list(dims)
    deriv_min = [hcl.scalar(1e9 if d in dims else 0, f"deriv_min{d}", dtype=V.dtype) for d in range(len(shape))]
    deriv_max = [hcl.scalar(-1e9 if d in dims else 0, f"deriv_max{d}", dtype=V.dtype) for d in range(len(shape))]
    # Per-slice bounds, one element per index of the first grid dimension
    slice_min = {d: hcl.compute((shape[0],), lambda i: 1e9, name=f"slice_min{d}", dtype=V.dtype) for d in dims}
    slice_max = {d: hcl.compute((shape[0],), lambda i: -1e9, name=f"slice_max{d}", dtype=V.dtype) for d in dims}

    def body(index):
        b0 = index[0]
        for dim in dims:
            dV_L, dV_R = spa_deriv_hcl(index, V, grid, dim)
            for value in (dV_L, dV_R):
                with hcl.if_(value < slice_min[dim][b0]):
                    slice_min[dim][b0] = value
                with hcl.if_(value > slice_max[dim][b0]):
                    slice_max[dim][b0] = value
    with hcl.Stage("Bounds"):
        with hcl.for_(0, shape[0], name="b0") as b0:
            for_all_points(shape[1:], lambda rest: body([b0] + rest), prefix="b_")
    with hcl.Stage("BoundsReduce"):
        with hcl.for_(0, shape[0], name="r0") as r0:
            for dim in dims:
                with hcl.if_(slice_min[dim][r0] < deriv_min[dim][0]):
                    deriv_min[dim][0] = slice_min[dim][r0]
                with hcl.if_(slice_max[dim][r0] > deriv_max[dim][0]):
                    deriv_max[dim][0] = slice_max[dim][r0]
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
