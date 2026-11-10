# First order spatial derivatives (left and right) for a grid of any dimension.
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
