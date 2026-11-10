# HJ_local_update

Local update of decomposed Hamilton-Jacobi reachability results: the full value function is approximated by combining two subsystem value functions, and the local update corrects that approximation only near the points where it goes wrong (the leaking set), so it matches the direct computation of the full system.

### Prerequisite

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp) (includes HeteroCL)

### Layout

The dimension of each case is in the file name: `_2d`, `_3d`, `_6d`.

    dynamics/          full systems: coupled_2d.py, dubins_3d.py, quadrotor_6d.py
    subdynamics/       subsystems of each full system, same file names
    compute/           direct computation (ground truth) and decomposition of each case
    numpy_impl/        local update, numpy: local_update_2d.py, local_update_3d.py, local_update_6d.py
      params/          parameter files of the numpy local updates
    hcl_impl/          local update, HeteroCL: local_update_2d.py, local_update_3d.py
      params/          parameter files of the HeteroCL local updates
    common/            code shared by the cases (spatial derivative, leaking set, statistics, parameter files, plots)
    results/           saved results, created when you run (not in git)
    kernels/           compiled HeteroCL kernels, created when you run (not in git)

Run every command from the repository root. Scripts run as modules, e.g. `python -m numpy_impl.local_update_3d`.

### Union and Intersection

`--combine` sets how the two subsystem value functions are combined:

- `union`: V = min(V1, V2)
- `intersection`: V = max(V1, V2)

| Case | Union | Intersection | Default |
|---|---|---|---|
| 2d | ✓ | ✓ | `intersection` |
| 3d | ✓ | not implemented, stops with an error | `union` |
| 6d | ✓ | not implemented, stops with an error | `union` |

Saved result files carry the dimension and the setting, e.g. `results/result_decomp_3d_union.npy`.

## 3D Dubins Car

### Running Local Update

First generate the direct and decomposition results (both default to `accuracy='low'` and `t_step=0.04` to match the local update scheme):

```bash
python -c "
from compute.direct_3d import direct_comp
from compute.decomposition_3d import decomposition
direct_comp(41, plot=False, combine='union')
decomposition(41, plot=False, combine='union')
"
```

This saves to `results/`:

- `result_direct_3d_union.npy`: direct computation (ground truth), shape (x, y, θ, t)
- `result_decomp_3d_union.npy`: decomposition V = min(V1, V2), shape (x, y, θ, t)
- `init_1_3d.npy`, `init_2_3d.npy`: subsystem initial values V1(x, θ) and V2(y, θ) on the 3D grid, shape (x, y, θ)

Then run the local update with a parameter file, numpy version:

```bash
python -m numpy_impl.local_update_3d --params numpy_impl/params/local_update_3d.json
```

or the parallel HeteroCL version:

```bash
python -m hcl_impl.local_update_3d --params hcl_impl/params/local_update_3d.json
```

The results are saved to `results/result_local_3d_union.npy` (numpy) and `results/result_local_hcl_3d_union.npy` (HeteroCL). The HeteroCL version saves the compiled kernel in `kernels/` on the first run and loads it on later runs.

#### Parameter Files

Each implementation keeps its parameter files next to its scripts:

- [numpy_impl/params/local_update_3d.json](numpy_impl/params/local_update_3d.json)
- [hcl_impl/params/local_update_3d.json](hcl_impl/params/local_update_3d.json)

Each key is an argument from the table below without `--`, for example:

```json
{
    "decomp_file": "results/result_decomp_3d_union.npy",
    "true_file": "results/result_direct_3d_union.npy",
    "init_1": "results/init_1_3d.npy",
    "init_2": "results/init_2_3d.npy",
    "combine": "union",
    "leaking_mode": "smallest",
    "t_step": 0.04
}
```

Keys left out of the file take their default value, and `"true_file": null` skips the comparison. An unknown key or an invalid `combine` or `leaking_mode` stops the script with an error. To try another setting, copy a parameter file and edit it. Arguments given on the command line override the file:

```bash
python -m hcl_impl.local_update_3d --params hcl_impl/params/local_update_3d.json --leaking_mode delta
```

All arguments can also be given on the command line alone, without `--params`.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--params` | no | none | JSON parameter file holding the arguments below |
| `--decomp_file` | yes | — | Decomposition result (in the parameter file or on the command line) |
| `--true_file` | no | none | Direct computation result. If not given, the local update runs and saves its result without the comparison |
| `--init_1`, `--init_2` | no | `results/init_1_3d.npy`, `results/init_2_3d.npy` | Subsystem initial values, used for the leaking set |
| `--combine` | no | `union` | `union` or `intersection`, only `union` is implemented for 3d |
| `--leaking_mode` | no | `smallest` | Initial leaking set, `smallest` or `delta`, see below |
| `--t_step` | no | `0.04` | Time step used to compute the files |
| `--cache_dir` | no | `kernels` | HeteroCL version only: directory where compiled kernels are saved and loaded |

### Leaking Set

At each time step, the local update recomputes the leaking set plus the points corrected at the previous step and their neighbours. The rest keep the decomposition value. The leaking set is where the decomposition itself goes wrong, near the switching surface V1 = V2:

- `--leaking_mode smallest` (default): the switching surface V1 = V2 plus its neighbours. The smallest leaking set that still gives the full correction, and it needs no δ. Check it with `--true_file` before trusting it on a new problem.
- `--leaking_mode delta`: |V1 − V2| < δ, with δ = max|V(t=0) − V(t=T)| of the decomposition. Conservative, about 3 times more leaking points.

Leaking corners on the 41³ grid (HeteroCL, δ = 0.997):

| Leaking set | Leaking points | True leaking corners covered | Points with error > 1e-5 after local update |
|---|---|---|---|
| True leaking corners | 3238 | — | — |
| `delta` | 26937 | 3238 / 3238 | 0 |
| `smallest` | 9717 | 3238 / 3238 | 0 |

True leaking corners are the points where the decomposition error starts. At some time step, such a point differs from the direct computation by > 1e-5 while none of its stencil points did at the step before. All of them have |V1 − V2| = 0 or 0.2 (one grid cell). Errors elsewhere (51323 points at the final time) spread from these corners and are caught by the growing update region. A leaking set that is too large only costs time. One that misses true leaking corners leaves errors uncorrected without any warning.

### Computation Time

41 x 41 x 41 grid, 25 time steps (`t_step=0.04`, `accuracy='low'`), default `--leaking_mode smallest`, measured on a machine with 32 logical CPUs (times vary by machine).

| Method | Time-stepping loop (s) | Total (s) | Points with error > 1e-5 |
|---|---|---|---|
| Direct computation (`compute/direct_3d.py`, HJSolver) | 0.046 | 0.177 | — |
| Local update, HeteroCL (`hcl_impl/local_update_3d.py`), first run | 0.039 | 0.208 | 0 |
| Local update, HeteroCL (`hcl_impl/local_update_3d.py`), cached kernel | 0.008 | 0.025 | 0 |
| Local update, numpy (`numpy_impl/local_update_3d.py`) | 10.6 | 10.6 | 4 |

- Time-stepping loop: HJSolver "Total kernel time" for direct computation, "Total correction time" for the local update.
- Total: the full HJSolver call for direct computation (kernel build + setup + loop); kernel build/load + input copy + loop for the HeteroCL local update; the loop only for the numpy local update (no build or copy step).
- The decomposition result is read from file, so its computation time is not included.
- Error is measured against the direct computation at the final time step (the decomposition alone has 51323 points with error > 1e-5).
- With `--leaking_mode delta` the error is the same and the time is about 10% higher (numpy 11.4 s), since most updated points come from the growing region rather than the leaking set.

## 2D Coupled Control

The leaking corner issue shows up in two settings, and `--combine` picks which:

- `union`: V = min(V1, V2), avoid (uMode `max`)
- `intersection`: V = max(V1, V2), reach (uMode `min`)

The local update runs the direct computation and the decomposition itself (numpy, in `compute/`), then corrects the decomposition and prints the comparison. Nothing is saved. Numpy version:

```bash
python -m numpy_impl.local_update_2d --grid_size 101 --combine union
python -m numpy_impl.local_update_2d --grid_size 101 --combine intersection
```

HeteroCL version, with a parameter file ([hcl_impl/params/local_update_2d.json](hcl_impl/params/local_update_2d.json)) or the same arguments on the command line:

```bash
python -m hcl_impl.local_update_2d --params hcl_impl/params/local_update_2d.json
python -m hcl_impl.local_update_2d --params hcl_impl/params/local_update_2d.json --combine intersection
```

Both versions correct the same points at every time step: the leaking corners |V1 − V2| < δ, plus the neighbours of every corrected point that changed by more than 1e-6, repeated until no new point joins. The HeteroCL version does this in one kernel per time step, with 64-bit floats like numpy. Its values agree with numpy within 5e-16, since the compiled arithmetic rounds a little differently in the last bit. It saves the compiled kernel in `kernels/` on the first run and loads it on later runs.

| Argument | Default | Description |
|---|---|---|
| `--params` | none | HeteroCL version only: JSON parameter file holding the arguments below |
| `--combine` | `intersection` | `union` or `intersection` (replaces the old `--use_union 1/0`) |
| `--grid_size` | `101` | Grid points per dimension |
| `--generate_big_delta` | `2` | `0`: constant `--big_delta`, `1`: from the initial and final decomposition result, `2`: a new δ at each time step |
| `--big_delta` | `0.2` | δ used when `--generate_big_delta 0` |
| `--debug` | `0` | `1` prints the stats of each time step |
| `--cache_dir` | `kernels` | HeteroCL version only: directory where compiled kernels are saved and loaded |

Correction time, 10 time steps, measured on a machine with 32 logical CPUs (HeteroCL with the cached kernel):

| Grid | Combine | Points corrected | numpy (s) | HeteroCL (s) |
|---|---|---|---|---|
| 101 x 101 | union | 19070 | 0.176 | 0.017 |
| 101 x 101 | intersection | 19070 | 0.175 | 0.016 |
| 201 x 201 | union | 47174 | 0.418 | 0.060 |
| 201 x 201 | intersection | 43550 | 0.392 | 0.062 |

Both versions leave 0 points with error > 1e-6 against the direct computation in all four cases.

## 6D Quadrotor

The local update runs the decomposition itself, finds the leaking set with `new_theory_L` (`common/leaking.py`) and corrects it. The grid size (`num = 21`) and look-back length (`lookback_length = 0.1`) are set at the top of `numpy_impl/local_update_6d.py`.

```bash
python -m numpy_impl.local_update_6d
```

It saves `results/result_decomp_6d_union.npy` and `results/result_local_6d_union.npy`. The direct computation (`compute/direct_6d.py`) is not run, so there is no comparison.

## Code Structure

    numpy_impl/local_update_2d.py
    ├── compute/direct_2d.py, compute/decomposition_2d.py, compute/config_2d.py
    ├── dynamics/coupled_2d.py, subdynamics/coupled_2d.py
    └── common/            (spatial_deriv, stats, combine)

    hcl_impl/local_update_2d.py
    ├── compute/direct_2d.py, compute/decomposition_2d.py, compute/config_2d.py
    ├── dynamics/coupled_2d.py (opt_ctrl, dynamics in HeteroCL)
    ├── hcl_impl/spatial_deriv.py, hcl_impl/kernel_cache.py
    └── common/            (stats, params, combine)

    numpy_impl/local_update_3d.py
    ├── dynamics/dubins_3d.py
    └── common/            (spatial_deriv, leaking, stats, params, combine)

    hcl_impl/local_update_3d.py
    ├── hcl_impl/kernel_cache.py
    ├── common/            (leaking, stats, params, combine)
    └── odp                (DubinsCar, 1st order ENO derivatives)

    compute/direct_3d.py   (odp HJSolver, DubinsCar)
    compute/decomposition_3d.py (odp HJSolver)
    └── subdynamics/dubins_3d.py

    numpy_impl/local_update_6d.py
    ├── compute/decomposition_6d.py
    │   └── subdynamics/quadrotor_6d.py
    ├── dynamics/quadrotor_6d.py
    └── common/            (spatial_deriv, leaking, combine)

`common/plotting.py` holds the overlay plots from the 2d and 6d cases. It needs `pre_plot` from `odp.Plots`, which the current optimized_dp no longer has, so it cannot be imported with the current version. No script imports it.
