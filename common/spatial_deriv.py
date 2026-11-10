# First order spatial derivatives (left and right) for a grid of any dimension.
#   spa_deriv:     numpy version
#   spa_deriv_hcl: same rules inside a HeteroCL kernel
import heterocl as hcl
from heterocl import tvm
import numpy as np


def spa_deriv(index, data, grid, dim, boundary):
    '''
        First order left and right spatial derivative of data at index along dimension dim.
            index: tuple of grid indices, one per dimension of data
            dim: dimension to differentiate along
            boundary: what to do at the edge of a non-periodic dimension
                "extrapolate": extrapolate a ghost point outside the grid, same as the first order
                    ENO scheme in optimized_dp (used by the 3d case)
                "one_sided": use the one-sided difference for both the left and the right
                    derivative (used by the 2d and 6d cases)
        Periodic dimensions (grid.pDim) wrap around.
        Returns (left_deriv, right_deriv).
    '''
    index = tuple(index)
    i = index[dim]
    n = data.shape[dim]
    dx = grid.dx[dim]
    before, after = index[:dim], index[dim+1:]

    def value_at(k):
        return data[before + (k,) + after]

    if dim in grid.pDim:
        left_deriv = (value_at(i) - value_at((i-1) % n))/dx
        right_deriv = (value_at((i+1) % n) - value_at(i))/dx
    elif i == 0:
        right_deriv = (value_at(i+1) - value_at(i))/dx
        if boundary == "extrapolate":
            left_boundary = value_at(i) + abs(value_at(i+1) - value_at(i))*np.sign(value_at(i))
            left_deriv = (value_at(i) - left_boundary)/dx
        elif boundary == "one_sided":
            left_deriv = right_deriv
        else:
            raise ValueError(f"unknown boundary: {boundary}")
    elif i == n-1:
        left_deriv = (value_at(i) - value_at(i-1))/dx
        if boundary == "extrapolate":
            right_boundary = value_at(i) + abs(value_at(i) - value_at(i-1))*np.sign(value_at(i))
            right_deriv = (right_boundary - value_at(i))/dx
        elif boundary == "one_sided":
            right_deriv = left_deriv
        else:
            raise ValueError(f"unknown boundary: {boundary}")
    else:
        left_deriv = (value_at(i) - value_at(i-1))/dx
        right_deriv = (value_at(i+1) - value_at(i))/dx

    return left_deriv, right_deriv


def spa_deriv_hcl(index, V, grid, dim, boundary):
    '''
        Same as spa_deriv, inside a HeteroCL kernel.
            index: list of loop variables, one per dimension of V
            V: HeteroCL tensor with the grid shape
            boundary: only "one_sided" is implemented here (the 3d HeteroCL kernel uses the
                first order ENO derivatives of optimized_dp, which extrapolate)
        Returns (left_deriv, right_deriv) as HeteroCL expressions.
    '''
    if boundary != "one_sided":
        raise NotImplementedError(f"boundary={boundary!r} is not implemented in spa_deriv_hcl, only 'one_sided'")
    i = index[dim]
    n = int(grid.pts_each_dim[dim])
    # A plain Python float in a kernel becomes a 32-bit constant, so build dx with the type of V
    dx = tvm.const(float(grid.dx[dim]), V.dtype)

    def value_at(k):
        idx = list(index)
        idx[dim] = k
        return V[tuple(idx)]

    left_deriv = hcl.scalar(0, f"deriv_L{dim}")
    right_deriv = hcl.scalar(0, f"deriv_R{dim}")
    if dim in grid.pDim:
        with hcl.if_(i == 0):
            left_deriv[0] = (value_at(i) - value_at(n - 1)) / dx
            right_deriv[0] = (value_at(i + 1) - value_at(i)) / dx
        with hcl.elif_(i == n - 1):
            left_deriv[0] = (value_at(i) - value_at(i - 1)) / dx
            right_deriv[0] = (value_at(0) - value_at(i)) / dx
        with hcl.else_():
            left_deriv[0] = (value_at(i) - value_at(i - 1)) / dx
            right_deriv[0] = (value_at(i + 1) - value_at(i)) / dx
    else:
        with hcl.if_(i == 0):
            right_deriv[0] = (value_at(i + 1) - value_at(i)) / dx
            left_deriv[0] = right_deriv[0]
        with hcl.elif_(i == n - 1):
            left_deriv[0] = (value_at(i) - value_at(i - 1)) / dx
            right_deriv[0] = left_deriv[0]
        with hcl.else_():
            left_deriv[0] = (value_at(i) - value_at(i - 1)) / dx
            right_deriv[0] = (value_at(i + 1) - value_at(i)) / dx
    return left_deriv[0], right_deriv[0]
