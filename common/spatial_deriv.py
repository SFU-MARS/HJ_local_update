# First order spatial derivatives (left and right) for a grid of any dimension, same as optimized_dp's
# 1st order ENO scheme (HJSolver accuracy "low"): periodic dimensions (grid.pDim) wrap around, other
# dimensions extrapolate a ghost point at the grid edge. All of them keep the float type of the data
# (32-bit, like HJSolver).
#   spa_deriv:        numpy, one grid point
#   spa_deriv_arrays: numpy, every grid point at once (same arithmetic)
#   spa_deriv_hcl:    inside a HeteroCL kernel, keeping the float type of V
import heterocl as hcl
from heterocl import tvm
import numpy as np


def spa_deriv(index, data, grid, dim):
    '''
        Left and right spatial derivative of data at index along dimension dim.
            index: tuple of grid indices, one per dimension of data
        Returns (left_deriv, right_deriv).
    '''
    index = tuple(index)
    i = index[dim]
    n = data.shape[dim]
    # dx with the float type of data (numpy turns a 32-bit scalar divided by a Python float into 64-bit)
    dx = data.dtype.type(grid.dx[dim])
    before, after = index[:dim], index[dim+1:]

    def value_at(k):
        return data[before + (k,) + after]

    if dim in grid.pDim:
        left_deriv = (value_at(i) - value_at((i-1) % n))/dx
        right_deriv = (value_at((i+1) % n) - value_at(i))/dx
    elif i == 0:
        right_deriv = (value_at(i+1) - value_at(i))/dx
        left_boundary = value_at(i) + abs(value_at(i+1) - value_at(i))*np.sign(value_at(i))
        left_deriv = (value_at(i) - left_boundary)/dx
    elif i == n-1:
        left_deriv = (value_at(i) - value_at(i-1))/dx
        right_boundary = value_at(i) + abs(value_at(i) - value_at(i-1))*np.sign(value_at(i))
        right_deriv = (right_boundary - value_at(i))/dx
    else:
        left_deriv = (value_at(i) - value_at(i-1))/dx
        right_deriv = (value_at(i+1) - value_at(i))/dx

    return left_deriv, right_deriv


def spa_deriv_arrays(data, grid, dim):
    '''
        spa_deriv at every grid point at once, same arithmetic.
        Returns (left_deriv, right_deriv) arrays with the shape of data.
    '''
    n = data.shape[dim]
    dx = data.dtype.type(grid.dx[dim])

    def sl(s):
        idx = [slice(None)] * data.ndim
        idx[dim] = s
        return tuple(idx)

    left = np.empty_like(data)
    right = np.empty_like(data)
    diff = (data[sl(slice(1, None))] - data[sl(slice(None, -1))])/dx
    left[sl(slice(1, None))] = diff
    right[sl(slice(None, -1))] = diff
    v0, v1 = data[sl(slice(0, 1))], data[sl(slice(1, 2))]
    vl, vl1 = data[sl(slice(n - 1, n))], data[sl(slice(n - 2, n - 1))]
    if dim in grid.pDim:
        left[sl(slice(0, 1))] = (v0 - vl)/dx
        right[sl(slice(n - 1, n))] = (v0 - vl)/dx
    else:
        left_boundary = v0 + abs(v1 - v0)*np.sign(v0)
        left[sl(slice(0, 1))] = (v0 - left_boundary)/dx
        right_boundary = vl + abs(vl - vl1)*np.sign(vl)
        right[sl(slice(n - 1, n))] = (right_boundary - vl)/dx
    return left, right


def spa_deriv_hcl(index, V, grid, dim):
    '''
        spa_deriv inside a HeteroCL kernel, keeping the float type of V.
            index: list of loop variables, one per dimension of V
            V: HeteroCL tensor with the grid shape
        Returns (left_deriv, right_deriv) as HeteroCL expressions.
    '''
    i = index[dim]
    n = int(grid.pts_each_dim[dim])
    # A plain Python float in a kernel becomes a 32-bit constant, so build dx with the type of V
    dx = tvm.const(float(grid.dx[dim]), V.dtype)

    def value_at(k):
        idx = list(index)
        idx[dim] = k
        return V[tuple(idx)]

    def abs_(x):
        return hcl.select(x > 0, x, -x)

    def sign_(x):
        return hcl.select(x > 0, tvm.const(1.0, V.dtype), hcl.select(x < 0, tvm.const(-1.0, V.dtype), tvm.const(0.0, V.dtype)))

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
            left_boundary = hcl.scalar(0, f"left_boundary{dim}")
            left_boundary[0] = value_at(i) + abs_(value_at(i + 1) - value_at(i)) * sign_(value_at(i))
            left_deriv[0] = (value_at(i) - left_boundary[0]) / dx
        with hcl.elif_(i == n - 1):
            left_deriv[0] = (value_at(i) - value_at(i - 1)) / dx
            right_boundary = hcl.scalar(0, f"right_boundary{dim}")
            right_boundary[0] = value_at(i) + abs_(value_at(i) - value_at(i - 1)) * sign_(value_at(i))
            right_deriv[0] = (right_boundary[0] - value_at(i)) / dx
        with hcl.else_():
            left_deriv[0] = (value_at(i) - value_at(i - 1)) / dx
            right_deriv[0] = (value_at(i + 1) - value_at(i)) / dx
    return left_deriv[0], right_deriv[0]
