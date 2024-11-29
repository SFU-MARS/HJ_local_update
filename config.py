import numpy as np
import heterocl as hcl
from system import couple_u
from subsystem import subsys
from odp.Grid import Grid
from odp.Shapes import *


class Config:
    def __init__(
        self,
        lookback_length: float,
        number_of_grid_points: int,
        time_steps: float,
        small_number: float,
        subsys_1d: subsys,
        sys_2d: couple_u,
    ) -> None:
        self._lookback_length = lookback_length
        self._number_of_grid_points = number_of_grid_points
        self._time_steps = time_steps
        self._small_number = small_number
        self._tau = np.arange(
            start=0, stop=lookback_length + small_number, step=time_steps
        )
        self._sys_2d = sys_2d
        self._subsys_1d = subsys_1d

    def value_function_1d(self, grid: Grid):
        return ShapeRectangle(grid, [-1.0], [1.0])

    def value_function_2d(self, grid: Grid):
        return ShapeRectangle(grid, [-1.0, -1.0], [1.0, 1.0])


def construct1DGrid(
    number_of_grid_points: int,
    min_val: float = -4.0,
    max_val: float = 4.0,
) -> Grid:
    # construct grid from the number of grid points
    grid_min = np.array([min_val])
    grid_max = np.array([max_val])
    dims = grid_min.shape[0]
    N = np.array([number_of_grid_points])
    return Grid(grid_min, grid_max, dims, N)


def construct2DGrid(
    number_of_grid_points: int,
    min_val: float = -4.0,
    max_val: float = 4.0,
) -> Grid:
    # construct grid from the number of grid points
    grid_min = np.array([min_val, min_val])
    grid_max = np.array([max_val, max_val])
    dims = grid_min.shape[0]
    N = np.array([number_of_grid_points, number_of_grid_points])
    return Grid(grid_min, grid_max, dims, N)
