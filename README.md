# HJ_local_update

A decomposition approximates a full Hamilton-Jacobi value function by combining two subsystem value functions, V = min(V1, V2) or max(V1, V2). The approximation goes wrong near the switching surface V1 = V2 (the leaking corners). The local update recomputes only those points, plus the ones the error spreads to, so the result matches the direct computation of the full system.

## Setup

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp) (includes HeteroCL).
- Run every command from the repository root. Scripts run as modules, e.g. `python -m numpy_impl.local_update_3d`.

## Workflow

1. **Generate** the direct computation (ground truth) and the decomposition with `compute/`. In every case both are solved with optimized_dp's HJSolver: the direct computation solves the full system (2D, 3D or 6D), and the decomposition solves each subsystem (1D for 2d, 2D for 3d, 4D for 6d) and combines them. HJSolver runs with `accuracy="low"` (1st order ENO + forward Euler) and exactly one step per time step; if `--t_step` is above its CFL bound, the scripts stop with an error. The results are saved to `results/`, with time as the last axis. The direct computation is optional: it is only used for the comparison in step 3.
2. **Local update** with `numpy_impl/` or `hcl_impl/` (HeteroCL, much faster), from a parameter file in its `params/` folder. At each updated point it takes the same step as HJSolver, including the Lax-Friedrichs dissipation.
3. **Results** are saved to `results/result_local_<dim>_<combine>.npy` (numpy) or `results/result_local_hcl_<dim>_<combine>.npy` (HeteroCL). If `true_file` (the direct computation result) is given, the local update also prints the error against it. It can be `null`, set `"true_file": null` in the parameter file or add `--true_file null`, and then the local update runs without the comparison. The HeteroCL version saves its compiled kernel in `kernels/` and loads it on later runs.

In the tables below, times are the time-stepping loops only, measured on a machine with 32 logical CPUs: HJSolver's "Total kernel time" for the direct computation and the decomposition (summed over the subsystems), and the correction loop for the local update. Kernel builds are not included: each HJSolver call builds its kernel first (about 0.1–0.4 s more), and the HeteroCL local update times are from a second run, which loads the kernel saved by the first. The input copies before the loop are not included either.

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

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Direct computation | Decomposition | Local update: numpy / HeteroCL |
|---|---|---|---|---|---|---|
| 41² | union | 10 | 1233 / 0 / 0 | 0.024 s | 0.016 s | 0.15 s / 0.0013 s |
| 41² | intersection | 10 | 897 / 0 / 0 | 0.023 s | 0.015 s | 0.098 s / 0.0013 s |
| 101² | union | 10 | 4569 / 0 / 0 | 0.023 s | 0.015 s | 0.49 s / 0.0024 s |
| 101² | intersection | 10 | 3145 / 0 / 0 | 0.024 s | 0.016 s | 0.31 s / 0.0023 s |
| 201² | union | 10 | 10473 / 40 / 40 | 0.026 s | 0.016 s | 1.1 s / 0.0057 s |
| 201² | intersection | 10 | 7277 / 0 / 0 | 0.025 s | 0.016 s | 0.62 s / 0.0057 s |

numpy and HeteroCL correct the same points, values within 2e-15. All points are within 2e-6 of the direct computation (HJSolver runs in 32-bit floats).

### Example: 3D Dubins Car

ẋ = v cos θ, ẏ = v sin θ, θ̇ = w with |w| ≤ wMax, from the rectangles |x| ≤ 0.5 and |y| ≤ 0.5. Union: the cross-shaped obstacle (avoid). Intersection: the square target (reach).

```bash
# 1. generate: results/result_{direct,decomp}_3d_union.npy, results/init_{1,2}_3d.npy
python -m compute.direct_3d --grid_size 41 --combine union
python -m compute.decomposition_3d --grid_size 41 --combine union
# 2. local update
#   option 1: numpy
python -m numpy_impl.local_update_3d --params numpy_impl/params/local_update_3d_union.json
#   option 2: parallel HeteroCL
python -m hcl_impl.local_update_3d --params hcl_impl/params/local_update_3d_union.json
```

For intersection, use `--combine intersection` and `local_update_3d_intersection.json`.

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Direct computation | Decomposition | Local update: numpy / HeteroCL |
|---|---|---|---|---|---|---|
| 41³ | union | 25 | 55224 / 380 / 30 | 0.047 s | 0.050 s | 10.2 s / 0.0097 s |
| 41³ | intersection | 25 | 54902 / 59 / 18 | 0.047 s | 0.051 s | 10.3 s / 0.0085 s |

The remaining points are all within 1e-5, except 4 points of numpy union: the local update follows HJSolver closely but not exactly. The HeteroCL kernel runs in 32-bit floats like optimized_dp, and its first run (with the kernel build) takes about 0.2 s.

### Example: 6D Quadrotor

State (x, y, vx, vy, θ, w), split into subsystems (x, vx, θ, w) and (y, vy, θ, w), which share θ, w and the controls. Union: the obstacle x ≤ 0 or y ≤ 0 (avoid). Intersection: the target x ≤ 0 and y ≤ 0 (reach).

```bash
# 1. generate: results/result_{direct,decomp}_6d_union.npy, results/init_{1,2}_6d.npy
python -m compute.direct_6d --grid_size 21 --combine union
python -m compute.decomposition_6d --grid_size 21 --combine union
# 2. local update
#   option 1: numpy (small grids only, e.g. --grid_size 9)
python -m numpy_impl.local_update_6d --params numpy_impl/params/local_update_6d_union.json
#   option 2: parallel HeteroCL (about 25 s)
python -m hcl_impl.local_update_6d --params hcl_impl/params/local_update_6d_union.json
```

Use `--grid_size 9` for a quick run (seconds).

For intersection, use `--combine intersection` and `local_update_6d_intersection.json`.

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Direct computation | Decomposition | Local update: numpy / HeteroCL |
|---|---|---|---|---|---|---|
| 7⁶ | union | 10 | 72058 / 17 / 17 | 0.098 s | 0.085 s | 13.7 s / 0.028 s |
| 7⁶ | intersection | 10 | 71538 / 19 / 19 | 0.10 s | 0.086 s | 13.8 s / 0.022 s |
| 9⁶ | union | 10 | 279471 / 58 / 58 | 0.17 s | 0.085 s | 51.6 s / 0.11 s |
| 9⁶ | intersection | 10 | 276310 / 61 / 61 | 0.15 s | 0.084 s | 51.8 s / 0.097 s |
| 21⁶ | union | 10 | 26250708 / – / 1537 | 11.1 s | 0.12 s | – / 6.5 s |
| 21⁶ | intersection | 10 | 26200126 / – / 411 | 11.1 s | 0.11 s | – / 6.3 s |

All points are within 1e-5 of the direct computation (at most 3e-6). numpy is not run at 21⁶. numpy and HeteroCL correct the same points, values within 3e-15. At 21⁶ (85.8 million points) the whole direct computation script takes about 40 s and 14 GB, the decomposition script 11 s and 8 GB, and the HeteroCL local update about 25 s in total and up to 24 GB (including the memory-mapped input file). Each result file is 7.5 GB.

## Parameter Files

Each local update reads its arguments from a JSON file in `numpy_impl/params/` or `hcl_impl/params/`, one per case and setting, e.g. `local_update_3d_intersection.json`. Keys are the argument names without `--`. Arguments given on the command line override the file, and `null` (in the file or on the command line) means not given. Unknown keys and invalid values stop the script with an error.


| Argument | Cases | Default | Description |
|---|---|---|---|
| `--params` | all | none | JSON parameter file |
| `--decomp_file` | all | required | Decomposition result |
| `--true_file` | all | none | Direct computation result, for the comparison |
| `--combine` | all | 2d: `intersection`, 3d/6d: `union` | `union` or `intersection` |
| `--dynamics`, `--dynamics_params` | all | the case's system, `{}` | Dynamics name from [dynamics/__init__.py](dynamics/__init__.py) and its parameters as JSON, e.g. `{"uMax": 1.5}`; the same as for the files |
| `--t_step` | all | 2d: `0.02`, 3d: `0.04`, 6d: `0.01` | Time step used to compute the files |
| `--cache_dir` | HeteroCL | `kernels` | Compiled kernel cache |
| `--sub1_file`, `--sub2_file` | 2d | required | Subsystem results V1, V2 |
| `--generate_big_delta`, `--big_delta` | 2d | `2`, `0.2` | δ of the leaking set |
| `--debug` | 2d | `0` | `1` prints the stats of each time step |
| `--init_1`, `--init_2` | 3d, 6d | `results/init_{1,2}_<dim>.npy` | Subsystem values at t = 0 |
| `--leaking_mode` | 3d, 6d | `smallest` | `smallest` or `delta` |

Compute scripts (`compute/direct_<dim>.py`, `compute/decomposition_<dim>.py`), the same for every case:

| Argument | Default | Description |
|---|---|---|
| `--grid_size` | 2d: `101`, 3d: `41`, 6d: `21` | Grid points per dimension |
| `--combine` | 2d: `intersection`, 3d/6d: `union` | `union` or `intersection` |
| `--dynamics`, `--dynamics_params` | the case's system, `{}` | Dynamics name from [dynamics/__init__.py](dynamics/__init__.py) and its parameters as JSON |
| `--t_step` | 2d: `0.02`, 3d: `0.04`, 6d: `0.01` | Time step, at most HJSolver's CFL bound (the scripts stop with an error otherwise) |
| `--lookback_length` | 2d: `0.2`, 3d: `1.0`, 6d: `0.1` | Look-back length, `lookback_length / t_step` time steps |
| `--results_dir` | `results` | Output folder |

3d only: `--accuracy` (HJSolver accuracy, default `low`, which matches the local updates) and `--plot` (also save an interactive plot to `plots/`).

## Code Layout

    dynamics/      full systems: coupled_2d.py, dubins_3d.py, quadrotor_6d.py (numpy and HeteroCL methods)
    subdynamics/   their subsystems, same file names
    compute/       direct computation and decomposition per case (HJSolver, hj_solver.py), config_2d.py and config_6d.py
    numpy_impl/    numpy local updates: local_update_{2d,3d,6d}.py, params/
    hcl_impl/      HeteroCL local updates: local_update_{2d,3d,6d}.py, kernel_cache.py, params/
    common/        shared code: HJSolver step and spatial derivative (hj_update.py, spatial_deriv.py), leaking set,
                   statistics, parameter files, union/intersection
    results/       saved results (not in git)
    kernels/       compiled HeteroCL kernels (not in git)

## Known Limitations

- `common/plotting.py` needs `pre_plot`, which the current optimized_dp no longer has. No script imports it.
