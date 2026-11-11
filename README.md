# HJ_local_update

A decomposition approximates a full Hamilton-Jacobi value function by combining two subsystem value functions, V = min(V1, V2) or max(V1, V2). The approximation goes wrong near the switching surface V1 = V2 (the leaking corners). The local update recomputes only those points, plus the ones the error spreads to, so the result matches the direct computation of the full system.

## Setup

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp) (includes HeteroCL).
- Run every command from the repository root. Scripts run as modules, e.g. `python -m numpy_impl.local_update_3d`.

## Workflow

1. **Generate** the direct computation (ground truth) and the decomposition with `compute/`. They are saved to `results/`, with time as the last axis. The direct computation is optional: it is only used for the comparison in step 3, so you can skip it, e.g. when it is too slow (6d).
2. **Local update** with `numpy_impl/` or `hcl_impl/` (HeteroCL, much faster), from a parameter file in its `params/` folder.
3. **Results** are saved to `results/result_local_<dim>_<combine>.npy` (numpy) or `results/result_local_hcl_<dim>_<combine>.npy` (HeteroCL). If `true_file` (the direct computation result) is given, the local update also prints the error against it. It can be `null`, set `"true_file": null` in the parameter file or add `--true_file null`, and then the local update runs without the comparison. The HeteroCL version saves its compiled kernel in `kernels/` and loads it on later runs.

Times below were measured on a machine with 32 logical CPUs, HeteroCL with the cached kernel.

### Example: 2D Coupled Control

ẋ = ux, ẏ = uy with ux² + uy² ≤ uMax². Union (avoid) and intersection (reach).

```bash
# 1. generate: results/result_{direct,decomp,sub1,sub2}_2d_union.npy
python -m compute.direct_2d --grid_size 101 --combine union
python -m compute.decomposition_2d --grid_size 101 --combine union
# 2. local update
#   option 1: numpy
python -m numpy_impl.local_update_2d --params numpy_impl/params/local_update_2d_union.json
#   option 2: parallel HeteroCL
python -m hcl_impl.local_update_2d --params hcl_impl/params/local_update_2d_union.json
```

For intersection, use `--combine intersection` and `local_update_2d_intersection.json`.

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Correction time: numpy / HeteroCL |
|---|---|---|---|---|
| 41² | union | 10 | 640 / 0 / 0 | 0.054 s / 0.0039 s |
| 41² | intersection | 10 | 640 / 0 / 0 | 0.054 s / 0.0038 s |
| 101² | union | 10 | 2432 / 0 / 0 | 0.18 s / 0.017 s |
| 101² | intersection | 10 | 2440 / 0 / 0 | 0.18 s / 0.016 s |
| 201² | union | 10 | 5784 / 0 / 0 | 0.42 s / 0.060 s |
| 201² | intersection | 10 | 5800 / 0 / 0 | 0.39 s / 0.062 s |

numpy and HeteroCL correct the same points, values within 5e-16.

### Example: 3D Dubins Car

ẋ = v cos θ, ẏ = v sin θ, θ̇ = w with |w| ≤ wMax, from the rectangles |x| ≤ 0.5 and |y| ≤ 0.5. Union: the cross-shaped obstacle (avoid). Intersection: the square target (reach). The direct computation and decomposition use the optimized_dp HJSolver.

```bash
# 1. generate: results/result_{direct,decomp}_3d_union.npy, results/init_{1,2}_3d.npy
python -c "
from compute.direct_3d import direct_comp
from compute.decomposition_3d import decomposition
direct_comp(41, plot=False, combine='union')
decomposition(41, plot=False, combine='union')
"
# 2. local update
#   option 1: numpy
python -m numpy_impl.local_update_3d --params numpy_impl/params/local_update_3d_union.json
#   option 2: parallel HeteroCL
python -m hcl_impl.local_update_3d --params hcl_impl/params/local_update_3d_union.json
```

For intersection, use `combine='intersection'` and `local_update_3d_intersection.json`.

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Correction time: numpy / HeteroCL |
|---|---|---|---|---|
| 41³ | union | 25 | 55224 / 380 / 30 | 10.4 s / 0.008 s |
| 41³ | intersection | 25 | 54902 / 59 / 18 | 10.3 s / 0.008 s |

The remaining points are all within 1e-5, except 4 points of numpy union: the local update follows HJSolver closely but not exactly. The HeteroCL kernel runs in 32-bit floats like optimized_dp, and its first run (with the kernel build) takes about 0.2 s. The HJSolver direct computation takes 0.18 s.

### Example: 6D Quadrotor

State (x, y, vx, vy, θ, w), split into subsystems (x, vx, θ, w) and (y, vy, θ, w), which share θ, w and the controls. Union only: the obstacle x ≤ 0 or y ≤ 0 (avoid).

```bash
# small grid 9^6, with the direct computation (about 30 s)
# 1. generate: results/result_{direct,decomp}_6d_union.npy, results/init_{1,2}_6d.npy
python -m compute.direct_6d --grid_size 9
python -m compute.decomposition_6d --grid_size 9
# 2. local update
#   option 1: numpy
python -m numpy_impl.local_update_6d --params numpy_impl/params/local_update_6d_union.json
#   option 2: parallel HeteroCL
python -m hcl_impl.local_update_6d --params hcl_impl/params/local_update_6d_union.json

# default grid 21^6, without the direct computation (pure Python, would take hours)
# 1. generate the decomposition only
python -m compute.decomposition_6d --grid_size 21
# 2. local update, without the comparison
#   option 1: numpy (about 23 min)
python -m numpy_impl.local_update_6d --params numpy_impl/params/local_update_6d_union.json --true_file null
#   option 2: parallel HeteroCL (about 1 min)
python -m hcl_impl.local_update_6d --params hcl_impl/params/local_update_6d_union.json --true_file null
```

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Correction time: numpy / HeteroCL |
|---|---|---|---|---|
| 7⁶ | union | 5 | 74074 / 0 / 0 | 4.4 s / 0.057 s |
| 9⁶ | union | 5 | 286339 / 0 / 0 | 17 s / 0.20 s |
| 21⁶ | union | 5 | not computed (no direct computation) | 23 min / 35 s |

numpy and HeteroCL correct the same points, values within 9e-16 (at 21⁶: 87.9 million points changed, within 6e-17). At 21⁶ the decomposition takes about 35 s and the HeteroCL local update about 1 minute in total, each with a peak of 12 GB of memory. The decomposition file is 4.1 GB.

## Parameter Files

Each local update reads its arguments from a JSON file in `numpy_impl/params/` or `hcl_impl/params/`, one per case and setting, e.g. `local_update_3d_intersection.json`. Keys are the argument names without `--`. Arguments given on the command line override the file, and `null` (in the file or on the command line) means not given. Unknown keys and invalid values stop the script with an error.


| Argument | Cases | Default | Description |
|---|---|---|---|
| `--params` | all | none | JSON parameter file |
| `--decomp_file` | all | required | Decomposition result |
| `--true_file` | all | none | Direct computation result, for the comparison |
| `--combine` | all | 2d: `intersection`, 3d/6d: `union` | `union` or `intersection` |
| `--dynamics`, `--dynamics_params` | all | the case's system, `{}` | See [Dynamics](#dynamics) |
| `--t_step` | all | 2d/6d: `0.02`, 3d: `0.04` | Time step used to compute the files |
| `--cache_dir` | HeteroCL | `kernels` | Compiled kernel cache |
| `--sub1_file`, `--sub2_file` | 2d | required | Subsystem results V1, V2 |
| `--generate_big_delta`, `--big_delta` | 2d | `2`, `0.2` | δ of the leaking set |
| `--debug` | 2d | `0` | `1` prints the stats of each time step |
| `--init_1`, `--init_2` | 3d, 6d | `results/init_{1,2}_<dim>.npy` | Subsystem values at t = 0 |
| `--leaking_mode` | 3d, 6d | `smallest` | `smallest` or `delta` |

Compute scripts (`compute/direct_2d.py`, `decomposition_2d.py`, `direct_6d.py`, `decomposition_6d.py`):

| Argument | Default | Description |
|---|---|---|
| `--grid_size` | 2d: `101`, 6d: `21` | Grid points per dimension |
| `--combine` | 2d: `intersection`, 6d: `union` | `union` or `intersection` |
| `--dynamics`, `--dynamics_params` | the case's system, `{}` | See [Dynamics](#dynamics) |
| `--t_step` | `0.02` | Time step |
| `--lookback_length` | 2d: `0.2`, 6d: `0.1` | Look-back length, `lookback_length / t_step` time steps |
| `--results_dir` | `results` | Output folder |

The 3d compute functions are Python functions: `direct_comp(num, combine=..., lookback_length=1.0, t_step=0.04, accuracy="low", plot=True)` and `decomposition(...)` with the same arguments.

## Code Layout

    dynamics/      full systems: coupled_2d.py, dubins_3d.py, quadrotor_6d.py (numpy and HeteroCL methods)
    subdynamics/   their subsystems, same file names
    compute/       direct computation and decomposition per case, config_2d.py and config_6d.py
    numpy_impl/    numpy local updates: local_update_{2d,3d,6d}.py, params/
    hcl_impl/      HeteroCL local updates: local_update_{2d,3d,6d}.py, kernel_cache.py, params/
    common/        shared code: spatial derivative, leaking set, statistics, parameter files, union/intersection
    results/       saved results (not in git)
    kernels/       compiled HeteroCL kernels (not in git)

## Known Limitations

- 6d intersection is not implemented.
- The 6d direct computation is a pure Python loop, practical only for small grids (about 30 s for 9⁶).
- `common/plotting.py` needs `pre_plot`, which the current optimized_dp no longer has. No script imports it.
