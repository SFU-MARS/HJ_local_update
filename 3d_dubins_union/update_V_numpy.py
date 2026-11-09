# First order spatial derivatives (left and right) on the 3D grid, used by local_update_numpy.py.
import numpy as np

'''
For 3D array
'''
def spa_deriv(index, data, grid, dim):
    '''
    First order left and right spatial derivative of data at index along dimension dim.
    Periodic dimensions (grid.pDim) wrap around, other dimensions extrapolate a ghost
    point at the boundary, same as the first order ENO scheme in optimized_dp.
    '''
    i = index[dim]
    n = data.shape[dim]
    dx = grid.dx[dim]

    def value_at(k):
        idx = list(index)
        idx[dim] = k
        return data[tuple(idx)]

    left_deriv = 0
    right_deriv = 0

    if dim not in grid.pDim:
        if i == 0:
            left_boundary = value_at(i) + abs(value_at(i+1) - value_at(i))*np.sign(value_at(i))
            left_deriv = (value_at(i) - left_boundary)/dx
            right_deriv = (value_at(i+1) - value_at(i))/dx
        elif i == n-1:
            right_boundary = value_at(i) + abs(value_at(i) - value_at(i-1))*np.sign(value_at(i))
            left_deriv = (value_at(i) - value_at(i-1))/dx
            right_deriv = (right_boundary - value_at(i))/dx
        else:
            left_deriv = (value_at(i) - value_at(i-1))/dx
            right_deriv = (value_at(i+1) - value_at(i))/dx

    else:
        left_deriv = (value_at(i) - value_at((i-1) % n))/dx
        right_deriv = (value_at((i+1) % n) - value_at(i))/dx

    return left_deriv, right_deriv


def spa_deriv3X1(i, j, k, data, grid):
    return spa_deriv((i, j, k), data, grid, 0)


def spa_deriv3X2(i, j, k, data, grid):
    return spa_deriv((i, j, k), data, grid, 1)


def spa_deriv3X3(i, j, k, data, grid):
    return spa_deriv((i, j, k), data, grid, 2)
