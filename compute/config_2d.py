import numpy as np
import heterocl as hcl
from dynamics.coupled_2d import couple_u
from subdynamics.coupled_2d import subsys
from dynamics import make_dynamics, make_subdynamics
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
        use_optimization1: bool,
        generate_big_delta: int,
        big_delta: float,
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
        self._use_optimization1 = use_optimization1
        self._generate_big_delta = generate_big_delta
        self._big_delta = big_delta

    def value_function_1d(self, grid: Grid):
        return ShapeRectangle(grid, [-1.0], [1.0])

    def value_function_2d(self, grid: Grid):
        if self._use_union:
            data_1 = ShapeRectangle(grid, [-5.0, -1.0], [5.0, 1.0])
            data_2 = ShapeRectangle(grid, [-1.0, -5.0], [1.0, 5.0])
            data =np.minimum(data_1, data_2)
            return data
        else:
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

    def hasIndex(self, index):
        return self.has(index[0], index[1])
        
    def isEmpty(self):
        if len(self.activeIndices()) == 0:
            return True
        else:
            return False

    def activeIndices(self):
        return np.argwhere(self.grid == True)
    
    def count(self):
        return len(self.activeIndices())


def initConfig(use_union: bool,
               grid_size: int,
               generate_big_delta: int,
               big_delta: float,
               dynamics: str = "coupled_2d",
               dynamics_params: dict = None) -> Config:
    '''
        Config of the 2d case. Used by numpy_impl/local_update_2d.py and hcl_impl/local_update_2d.py.
            dynamics, dynamics_params: full system dynamics by name and its parameters (see dynamics/__init__.py);
                the subsystem dynamics of the decomposition are built from the same name and parameters.
                Union uses uMode='max' (avoid), intersection uses uMode='min' (reach).
    '''
    combine = "union" if use_union else "intersection"
    sys_2d = make_dynamics(dynamics, dynamics_params, combine, dim=2)
    subsys_1d, = make_subdynamics(dynamics, dynamics_params, combine, dim=2)

    return Config(
        number_of_grid_points=grid_size,
        lookback_length=0.20,
        time_step=0.02,
        small_number=1e-5,
        sys_2d=sys_2d,
        subsys_1d=subsys_1d,
        use_union=use_union,
        threshold2=1e-6,
        use_optimization1=True,
        generate_big_delta=generate_big_delta,
        big_delta=big_delta,
    )
