# HJ_local_update

A decomposition approximates a full Hamilton-Jacobi value function by combining two subsystem value functions, V = min(V1, V2) or max(V1, V2). The approximation goes wrong near the switching surface V1 = V2 (the leaking corners). The local update recomputes only those points, plus the ones the error spreads to, so the result matches the direct computation of the full system.

## Setup

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp) (includes HeteroCL).
- Run every command from the repository root. Scripts run as modules, e.g. `python -m numpy_impl.local_update_3d`.

## Workflow

1. **Generate** the direct computation (ground truth) and the decomposition with `compute/`. In every case both are solved with optimized_dp's HJSolver: the direct computation solves the full system (2D, 3D or 6D), and the decomposition solves each subsystem (1D for 2d, 2D for 3d, 4D for 6d) and combines them. HJSolver runs with `accuracy="low"` (1st order ENO + forward Euler) and exactly one step per time step; if `--t_step` is above its CFL bound, the scripts stop with an error. The results are saved to `results/` in 32-bit floats (as HJSolver computes them): the direct computation with time as the last axis, shape `(x, y, …, t)` (as HJSolver returns it), and the decomposition with time as the first axis, shape `(t, x, y, …)`, so the local update can read it without reordering. The direct computation is optional: it is only used for the comparison in step 3.
2. **Local update** with `numpy_impl/` or `hcl_impl/` (HeteroCL, much faster), from a parameter file in its `params/` folder. Every case uses the same growing-region method (`growing_region.py`): the leaking set comes from the subsystem values at t = 0 (V1 = V2 and its neighbours), and at each time step the region is that set plus the points corrected at the previous step and their neighbours. With `grow_within_step` (on for 2d), the region also grows within each step until no new point changes. At each updated point it takes the same step as HJSolver, including the Lax-Friedrichs dissipation, in 32-bit floats (`common/hj_update.py`).
3. **Results** are saved to `results/result_local_<dim>_<combine>.npy` (numpy) or `results/result_local_hcl_<dim>_<combine>.npy` (HeteroCL), time as the first axis like the decomposition. If `true_file` (the direct computation result) is given, the local update also prints the error against it at the final time; the direct computation result is only used for this comparison, never by the local update. It can be `null`, set `"true_file": null` in the parameter file or add `--true_file null`, and then the local update runs without the comparison. The HeteroCL version saves its compiled kernel in `kernels/` and loads it on later runs.

In the tables below, times are the time-stepping loops only, measured on a machine with 32 logical CPUs: HJSolver's "Total kernel time" for the direct computation and the decomposition (summed over the subsystems), and the correction loop for the local update. Kernel builds are not included: each HJSolver call builds its kernel first (about 0.1–0.4 s more), and the HeteroCL local update times are from a second run, which loads the kernel saved by the first. The input copies before the loop are not included either.

### Example: 2D Coupled Control

ẋ = ux, ẏ = uy with ux² + uy² ≤ uMax², split into subsystems x and y, from the intervals |x| ≤ 1 and |y| ≤ 1. Union: the cross-shaped obstacle (avoid). Intersection: the square target (reach). The subsystems each get the full uMax, so their dissipation is larger than the full system's: the decomposition is also wrong at kinks of V away from the leaking set, which is why 2d grows the region within each step.

```bash
# 1. generate: results/result_{direct,decomp}_2d_union.npy, results/init_{1,2}_2d.npy
python -m compute.direct_2d --grid_size 101 --combine union
python -m compute.decomposition_2d --grid_size 101 --combine union
# 2. local update
#   option 1: numpy
python -m numpy_impl.local_update_2d --params numpy_impl/params/local_update_2d_union.json
#   option 2: parallel HeteroCL
python -m hcl_impl.local_update_2d --params hcl_impl/params/local_update_2d_union.json
```

For intersection, use `--combine intersection` and `local_update_2d_intersection.json`.

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Direct computation (odp) | Decomposition (odp) | Local update: numpy / HeteroCL |
|---|---|---|---|---|---|---|
| 41² | union | 10 | 1233 / 0 / 0 | 0.024 s | 0.016 s | 0.21 s / 0.0013 s |
| 41² | intersection | 10 | 897 / 0 / 0 | 0.024 s | 0.016 s | 0.14 s / 0.0011 s |
| 101² | union | 10 | 4569 / 0 / 0 | 0.024 s | 0.016 s | 0.66 s / 0.0024 s |
| 101² | intersection | 10 | 3145 / 0 / 0 | 0.024 s | 0.016 s | 0.43 s / 0.0018 s |
| 201² | union | 10 | 10473 / 0 / 0 | 0.025 s | 0.016 s | 1.5 s / 0.0060 s |
| 201² | intersection | 10 | 7277 / 0 / 0 | 0.025 s | 0.016 s | 0.91 s / 0.0033 s |

All points are within 5e-7 of the direct computation (32-bit rounding), and numpy and HeteroCL update the same points, within 2.4e-7 of each other. 2d grows the region within each step (`grow_within_step`), in the HeteroCL kernel too.

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

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Direct computation (odp) | Decomposition (odp) | Local update: numpy / HeteroCL |
|---|---|---|---|---|---|---|
| 41³ | union | 25 | 55224 / 6 / 30 | 0.047 s | 0.051 s | 20.9 s / 0.0048 s |
| 41³ | intersection | 25 | 54902 / 2 / 18 | 0.046 s | 0.051 s | 20.9 s / 0.0048 s |

The remaining points are 32-bit rounding that builds up over the 25 time steps: every point is within 2.4e-6 of the direct computation. numpy matches HJSolver exactly after the first step, HeteroCL within one rounding step (6e-8), since its kernel adds up the same terms in a different order.

### Example: 6D Quadrotor

State (x, y, vx, vy, θ, w), split into subsystems (x, vx, θ, w) and (y, vy, θ, w), which share θ, w and the controls. Union: the obstacle x ≤ 0 or y ≤ 0 (avoid). Intersection: the target x ≤ 0 and y ≤ 0 (reach).

```bash
# 1. generate: results/result_{direct,decomp}_6d_union.npy, results/init_{1,2}_6d.npy
python -m compute.direct_6d --grid_size 21 --combine union
python -m compute.decomposition_6d --grid_size 21 --combine union
# 2. local update
#   option 1: numpy (small grids only, e.g. --grid_size 9)
python -m numpy_impl.local_update_6d --params numpy_impl/params/local_update_6d_union.json
#   option 2: parallel HeteroCL (about 8 s in total with the comparison, under 1 s of it the local update)
python -m hcl_impl.local_update_6d --params hcl_impl/params/local_update_6d_union.json
```

Use `--grid_size 9` for a quick run (HeteroCL under a second, numpy about 100 s).

For intersection, use `--combine intersection` and `local_update_6d_intersection.json`.

| Grid | Combine | Time steps | Points off > 1e-6 at the final time: decomposition / numpy / HeteroCL | Direct computation (odp) | Decomposition (odp) | Local update: numpy / HeteroCL |
|---|---|---|---|---|---|---|
| 7⁶ | union | 10 | 72058 / 0 / 0 | 0.10 s | 0.084 s | 28.6 s / 0.0059 s |
| 7⁶ | intersection | 10 | 71538 / 0 / 0 | 0.099 s | 0.086 s | 27.7 s / 0.0092 s |
| 9⁶ | union | 10 | 279471 / 0 / 0 | 0.16 s | 0.087 s | 104 s / 0.024 s |
| 9⁶ | intersection | 10 | 276310 / 0 / 0 | 0.17 s | 0.086 s | 102 s / 0.022 s |
| 21⁶ | union | 10 | 26250708 / – / 0 | 10.8 s | 0.12 s | – / 0.82 s |
| 21⁶ | intersection | 10 | 26200126 / – / 0 | 10.9 s | 0.13 s | – / 0.87 s |

All points are within 1e-6 of the direct computation (32-bit rounding), and numpy and HeteroCL are within 4.8e-7 of each other. numpy is not run at 21⁶. At 21⁶ (85.8 million points) each result file is 3.8 GB. The local update recomputes 28% of the point updates of the direct computation (239 of 858 million over the 10 steps); the whole HeteroCL script takes about 7 s and 5.2 GB: about 2 s reading the decomposition file and setting up the HeteroCL arrays, under 1 s for the local update, 2.8 s writing the 3.8 GB result. The comparison with the direct computation adds about 1.4 s (8.8 GB peak). The HeteroCL version corrects the decomposition result in place, so it only writes the updated points.

## Parameter Files

Each local update reads its arguments from a JSON file in `numpy_impl/params/` or `hcl_impl/params/`, one per case and setting, e.g. `local_update_3d_intersection.json`. Every case takes the same arguments (`common/local_update_setup.py`), only the defaults differ. Keys are the argument names without `--`. Arguments given on the command line override the file, and `null` (in the file or on the command line) means not given. Unknown keys and invalid values stop the script with an error.


| Argument | Cases | Default | Description |
|---|---|---|---|
| `--params` | all | none | JSON parameter file |
| `--decomp_file` | all | required | Decomposition result |
| `--true_file` | all | none | Direct computation result, for the comparison |
| `--combine` | all | 2d: `intersection`, 3d/6d: `union` | `union` or `intersection` |
| `--dynamics`, `--dynamics_params` | all | the case's system, `{}` | Dynamics name from [dynamics/__init__.py](dynamics/__init__.py) and its parameters as JSON, e.g. `{"uMax": 1.5}`; the same as for the files |
| `--t_step` | all | 2d: `0.02`, 3d: `0.04`, 6d: `0.01` | Time step used to compute the files |
| `--init_1`, `--init_2` | all | `results/init_{1,2}_<dim>.npy` | Subsystem values at t = 0, for the leaking set |
| `--leaking_mode` | all | `smallest` | `smallest` (V1 = V2 and its neighbours) or `delta` (\|V1 − V2\| < δ) |
| `--grow_within_step` | all | 2d: `true`, 3d/6d: `false` | Also grow the region within each time step |
| `--cache_dir` | HeteroCL | `kernels` | Compiled kernel cache |

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
    compute/       direct computation and decomposition per case (HJSolver, hj_solver.py),
                   config_{2d,3d,6d}.py: grid, initial values, file names and local update defaults of each case
    numpy_impl/    numpy local updates: local_update_{2d,3d,6d}.py, growing_region.py, params/
    hcl_impl/      HeteroCL local updates: local_update_{2d,3d,6d}.py, growing_region.py, kernel_cache.py, params/
    common/        shared code: local update arguments and setup (local_update_setup.py), HJSolver step and spatial
                   derivative (hj_update.py, spatial_deriv.py), leaking set, statistics, parameter files, union/intersection,
                   reading the large decomposition file straight into HeteroCL (npy_io.py)
    results/       saved results (not in git)
    kernels/       compiled HeteroCL kernels (not in git)

## Known Limitations

- `common/plotting.py` needs `pre_plot`, which the current optimized_dp no longer has. No script imports it.
