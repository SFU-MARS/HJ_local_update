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
               dynamics_params: dict = None,
               t_step: float = 0.02,
               lookback_length: float = 0.20) -> Config:
    '''
        Config of the 2d case. Used by compute/direct_2d.py, compute/decomposition_2d.py and the 2d local updates.
            t_step, lookback_length: time step and look-back length, lookback_length / t_step time steps
            dynamics, dynamics_params: full system dynamics by name and its parameters (see dynamics/__init__.py);
                the subsystem dynamics of the decomposition are built from the same name and parameters.
                Union uses uMode='max' (avoid), intersection uses uMode='min' (reach).
    '''
    combine = "union" if use_union else "intersection"
    sys_2d = make_dynamics(dynamics, dynamics_params, combine, dim=2)
    subsys_1d, = make_subdynamics(dynamics, dynamics_params, combine, dim=2)

    return Config(
        number_of_grid_points=grid_size,
        lookback_length=lookback_length,
        time_step=t_step,
        small_number=1e-5,
        sys_2d=sys_2d,
        subsys_1d=subsys_1d,
        use_union=use_union,
        threshold2=1e-6,
        use_optimization1=True,
        generate_big_delta=generate_big_delta,
        big_delta=big_delta,
    )


def result_files_2d(combine, results_dir="results"):
    '''
        Names of the saved 2d results, all with time as the last axis, shape (x, y, t):
            direct: direct computation (compute/direct_2d.py)
            decomp: decomposition V = min or max(V1, V2) (compute/decomposition_2d.py)
            sub1, sub2: subsystem results V1, V2 on the 2D grid (compute/decomposition_2d.py)
    '''
    return {
        "direct": f"{results_dir}/result_direct_2d_{combine}.npy",
        "decomp": f"{results_dir}/result_decomp_2d_{combine}.npy",
        "sub1": f"{results_dir}/result_sub1_2d_{combine}.npy",
        "sub2": f"{results_dir}/result_sub2_2d_{combine}.npy",
    }


def load_results_2d(decomp_file, sub1_file, sub2_file, true_file=None):
    '''
        Used by the 2d local updates. Load the saved results (time as the last axis) and return them with
        time as the first axis, as the 2d code uses them:
            (DecompositionResult, direct computation result or None)
        Raises ValueError if the shapes do not match.
    '''
    from compute.decomposition_2d import DecompositionResult
    decomp = np.load(decomp_file)
    sub1 = np.load(sub1_file)
    sub2 = np.load(sub2_file)
    for path, data in ((sub1_file, sub1), (sub2_file, sub2)):
        if data.shape != decomp.shape:
            raise ValueError(f"shape mismatch: {path} {data.shape} vs {decomp_file} {decomp.shape}, "
                             "generate them with the same compute/decomposition_2d.py run")
    result_true = None
    if true_file is not None:
        result_true = np.load(true_file)
        if result_true.shape != decomp.shape:
            raise ValueError(f"shape mismatch: {true_file} {result_true.shape} vs {decomp_file} {decomp.shape}, "
                             "generate both with the same grid_size, t_step and lookback_length")
        result_true = np.moveaxis(result_true, -1, 0).copy()
    decomposition_result = DecompositionResult(
        upper_subsystem_results=np.moveaxis(sub1, -1, 0).copy(),
        lower_subsystem_results=np.moveaxis(sub2, -1, 0).copy(),
        combined_results=np.moveaxis(decomp, -1, 0).copy(),
    )
    return decomposition_result, result_true


def add_compute_args_2d(parser):
    '''
        Arguments of compute/direct_2d.py and compute/decomposition_2d.py. Use the same values for both,
        the local update reads the grid size and number of time steps from the saved files.
    '''
    from dynamics import add_dynamics_args
    parser.add_argument("--grid_size", type=int, default=101, help="Grid points per dimension, e.g. 101 for 101x101.")
    parser.add_argument("--combine", choices=["union", "intersection"], default="intersection",
                        help="union = min(V1, V2) with uMode max (avoid), intersection = max(V1, V2) with uMode min (reach).")
    parser.add_argument("--t_step", type=float, default=0.02, help="Time step.")
    parser.add_argument("--lookback_length", type=float, default=0.20, help="Look-back length, lookback_length / t_step time steps.")
    parser.add_argument("--results_dir", default="results", help="Directory of the saved results.")
    add_dynamics_args(parser, default="coupled_2d")


def config_from_compute_args_2d(parser, args):
    '''
        Config from the arguments of add_compute_args_2d.
    '''
    try:
        config = initConfig(use_union=args.combine == "union", grid_size=args.grid_size, generate_big_delta=2,
                            big_delta=0.2, dynamics=args.dynamics, dynamics_params=args.dynamics_params,
                            t_step=args.t_step, lookback_length=args.lookback_length)
    except (ValueError, TypeError) as e:
        parser.error(f"--dynamics {args.dynamics} --dynamics_params {args.dynamics_params}: {e}")
    print(f"combine: {args.combine}, grid size: {args.grid_size} x {args.grid_size}, "
          f"time steps: {len(config._tau) - 1} (t_step {args.t_step}, lookback_length {args.lookback_length})")
    print(f"Dynamics: {args.dynamics} {vars(config._sys_2d)}")
    return config


def add_local_update_args_2d(parser):
    '''
        Arguments shared by numpy_impl/local_update_2d.py and hcl_impl/local_update_2d.py.
    '''
    from dynamics import add_dynamics_args
    parser.add_argument("--params", default=None,
                        help="JSON parameter file, its keys are the argument names below without '--'. "
                        "Arguments given on the command line override the file.")
    parser.add_argument("--decomp_file", default=None,
                        help="Decomposition result from compute/decomposition_2d.py, shape (x, y, t). Required.")
    parser.add_argument("--sub1_file", default=None,
                        help="Subsystem 1 result V1 from compute/decomposition_2d.py, shape (x, y, t). Required.")
    parser.add_argument("--sub2_file", default=None,
                        help="Subsystem 2 result V2 from compute/decomposition_2d.py, shape (x, y, t). Required.")
    parser.add_argument("--true_file", default=None,
                        help="Optional direct computation result from compute/direct_2d.py, shape (x, y, t). "
                        "If not given, the local update runs without the comparison.")
    parser.add_argument("--combine", choices=["union", "intersection"], default="intersection",
                        help="union = min(V1, V2) with uMode max (avoid), intersection = max(V1, V2) with uMode min "
                        "(reach). Use the same as for the files.")
    parser.add_argument("--t_step", type=float, default=0.02, help="Time step used to compute the files.")
    parser.add_argument("--generate_big_delta", type=int, choices=[0, 1, 2], default=2,
                        help="0: constant --big_delta. 1: from the initial and final decomposition result. "
                        "2: a new big_delta at each time step (as described in the paper).")
    parser.add_argument("--big_delta", type=float, default=0.2, help="big_delta used with --generate_big_delta 0.")
    parser.add_argument("--debug", type=int, choices=[0, 1], default=0, help="1 prints the stats of each time step.")
    add_dynamics_args(parser, default="coupled_2d")


def setup_local_update_2d(parser, args):
    '''
        Check the arguments of add_local_update_args_2d, load the files and build the config, with the grid
        size and number of time steps read from the decomposition file.
        Returns (config, DecompositionResult, direct computation result or None), time as the first axis.
    '''
    import os
    for key in ("decomp_file", "sub1_file", "sub2_file"):
        if getattr(args, key) is None:
            parser.error(f"--{key} is required (on the command line or in --params)")
    for path in (args.decomp_file, args.sub1_file, args.sub2_file, args.true_file):
        if path is not None and not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    try:
        decomposition_result, result_true = load_results_2d(args.decomp_file, args.sub1_file, args.sub2_file, args.true_file)
    except ValueError as e:
        parser.error(str(e))
    num_steps, nx, ny = np.shape(decomposition_result.combined())
    if nx != ny:
        parser.error(f"{args.decomp_file}: the 2d grid must be square, got {nx} x {ny}")
    try:
        config = initConfig(use_union=args.combine == "union", grid_size=nx,
                            generate_big_delta=args.generate_big_delta, big_delta=args.big_delta,
                            dynamics=args.dynamics, dynamics_params=args.dynamics_params,
                            t_step=args.t_step, lookback_length=args.t_step * (num_steps - 1))
    except (ValueError, TypeError) as e:
        parser.error(f"--dynamics {args.dynamics} --dynamics_params {args.dynamics_params}: {e}")
    if len(config._tau) != num_steps:
        parser.error(f"{args.decomp_file} has {num_steps} time steps, which does not fit --t_step {args.t_step}")
    print(f"combine: {args.combine}")
    print(f"generate_big_delta: {args.generate_big_delta}")
    print(f"Grid size: {nx} x {ny}")
    print(f"Dynamics: {args.dynamics} {vars(config._sys_2d)}")
    print(f"Number of time_steps: {num_steps - 1}")
    print("")
    return config, decomposition_result, result_true
