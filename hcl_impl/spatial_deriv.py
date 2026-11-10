# First order spatial derivatives (left and right) inside a HeteroCL kernel, any dimension.
# Same rules as common/spatial_deriv.py with boundary="one_sided".
import heterocl as hcl
from heterocl import tvm


def spa_deriv_one_sided(V, index, dim, grid):
    '''
        Left and right spatial derivative of V at index along dimension dim, inside a HeteroCL kernel.
            V: HeteroCL tensor with the grid shape
            index: list of loop variables, one per dimension
        Periodic dimensions (grid.pDim) wrap around. At the edge of a non-periodic dimension the
        one-sided difference is used for both the left and the right derivative.
        Returns (left, right) as HeteroCL expressions.
    '''
    i = index[dim]
    n = int(grid.pts_each_dim[dim])
    # A plain Python float in a kernel becomes a 32-bit constant, so build dx with the type of V
    dx = tvm.const(float(grid.dx[dim]), V.dtype)

    def at(k):
        idx = list(index)
        idx[dim] = k
        return V[tuple(idx)]

    left = hcl.scalar(0, f"deriv_L{dim}")
    right = hcl.scalar(0, f"deriv_R{dim}")
    if dim in grid.pDim:
        with hcl.if_(i == 0):
            left[0] = (at(i) - at(n - 1)) / dx
            right[0] = (at(i + 1) - at(i)) / dx
        with hcl.elif_(i == n - 1):
            left[0] = (at(i) - at(i - 1)) / dx
            right[0] = (at(0) - at(i)) / dx
        with hcl.else_():
            left[0] = (at(i) - at(i - 1)) / dx
            right[0] = (at(i + 1) - at(i)) / dx
    else:
        with hcl.if_(i == 0):
            right[0] = (at(i + 1) - at(i)) / dx
            left[0] = right[0]
        with hcl.elif_(i == n - 1):
            left[0] = (at(i) - at(i - 1)) / dx
            right[0] = left[0]
        with hcl.else_():
            left[0] = (at(i) - at(i - 1)) / dx
            right[0] = (at(i + 1) - at(i)) / dx
    return left[0], right[0]
