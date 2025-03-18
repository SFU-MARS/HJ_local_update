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
        time_step: float,
        small_number: float,
        subsys_1d: subsys,
        sys_2d: couple_u,
        use_union: bool,
        threshold2:float,
    ) -> None:
        self._lookback_length = lookback_length
        self._number_of_grid_points = number_of_grid_points
        self._time_step = time_step
        self._small_number = small_number
        self._tau = np.arange(
            start=0, stop=lookback_length + small_number, step=time_step
        )
        self._sys_2d = sys_2d
        self._subsys_1d = subsys_1d
        self._use_union = use_union
        self._threshold2 = threshold2

    def value_function_1d(self, grid: Grid):
        return ShapeRectangle(grid, [-1.0], [1.0])

    def value_function_2d(self, grid: Grid):
        return ShapeRectangle(grid, [-1.0, -1.0], [1.0, 1.0])
    
    def grid_2D(self):
        return construct2DGrid(number_of_grid_points=self._number_of_grid_points)

    def grid_1D(self):
        return construct1DGrid()


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


class Frontier:
    def __init__(
            self, 
            x, y,
    ):
        self._x = x
        self._y = y
        self.grid = np.zeros([x, y])
    
    # def add(self, x, y):
    #     self.grid[x][y] = True
    
    def add(self, index):
        self.grid[index[0]][index[1]] = True
        
    def reset(self):
        self.grid = np.zeros([self._x, self._y])

    def has(self, x, y):
        if self.grid[x][y]: 
            return True
        else:
            return False
        
    def isEmpty(self):
        if len(self.activeIndices()) == 0:
            return True
        else:
            return False

    def activeIndices(self):
        return np.argwhere(self.grid == True)
    
    def count(self):
        return len(self.activeIndices())
        