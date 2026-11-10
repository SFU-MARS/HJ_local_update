# HJ_local_update

### Prerequisite

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp) (includes HeteroCL)

### Running Local Update

Run all commands inside `3d_dubins_union`.

First generate the direct and decomposition results (both default to `accuracy='low'` and `t_step=0.04` to match the local update scheme):

```bash
python -c "
from direct import direct_comp
from decomposition import decomposition
direct_comp(41, plot=False)
decomposition(41, plot=False)
"
```

This saves to `results/`:

- `result_direct_3d.npy`: direct computation (ground truth), shape (x, y, θ, t)
- `result_decomp_3d.npy`: decomposition V = min(V1, V2), shape (x, y, θ, t)
- `init_1.npy`, `init_2.npy`: subsystem initial values V1(x, θ) and V2(y, θ) on the 3D grid, shape (x, y, θ)

Then run the local update, numpy version:

```bash
python local_update_numpy.py --decomp_file results/result_decomp_3d.npy --true_file results/result_direct_3d.npy --init_1 results/init_1.npy --init_2 results/init_2.npy --t_step 0.04
```

or the parallel HeteroCL version:

```bash
python local_update_hcl.py --decomp_file results/result_decomp_3d.npy --true_file results/result_direct_3d.npy --init_1 results/init_1.npy --init_2 results/init_2.npy --t_step 0.04
```

The results are saved to `results/result_local_3d.npy` (numpy) and `results/result_local_hcl_3d.npy` (HeteroCL). The HeteroCL version saves the compiled kernel in `kernels/` on the first run and loads it on later runs.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--decomp_file` | yes | — | Decomposition result |
| `--true_file` | no | none | Direct computation result. If not given, the local update runs and saves its result without the comparison |
| `--init_1`, `--init_2` | no | `results/init_1.npy`, `results/init_2.npy` | Subsystem initial values, used for the leaking set |
| `--leaking_mode` | no | `smallest` | Initial leaking set, `smallest` or `delta`, see below |
| `--t_step` | no | `0.04` | Time step used to compute the files |

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
| Direct computation (`direct.py`, HJSolver) | 0.046 | 0.177 | — |
| Local update, HeteroCL (`local_update_hcl.py`), first run | 0.039 | 0.208 | 0 |
| Local update, HeteroCL (`local_update_hcl.py`), cached kernel | 0.008 | 0.025 | 0 |
| Local update, numpy (`local_update_numpy.py`) | 10.6 | 10.6 | 4 |

- Time-stepping loop: HJSolver "Total kernel time" for direct computation, "Total correction time" for the local update.
- Total: the full HJSolver call for direct computation (kernel build + setup + loop); kernel build/load + input copy + loop for the HeteroCL local update; the loop only for the numpy local update (no build or copy step).
- The decomposition result is read from file, so its computation time is not included.
- Error is measured against the direct computation at the final time step (the decomposition alone has 51323 points with error > 1e-5).
- With `--leaking_mode delta` the error is the same and the time is about 10% higher (numpy 11.4 s), since most updated points come from the growing region rather than the leaking set.

#### Code Structure

    local_update_numpy.py
    ├── update_V_numpy.py
    └── system.py

    local_update_hcl.py
    ├── local_update_numpy.py  (dilate, get_leaking_set, printCorrectnessStatistics)
    └── odp                    (DubinsCar, 1st order ENO derivatives)

    direct.py                  (odp HJSolver, DubinsCar)
    decomposition.py           (odp HJSolver)
    └── subsystem.py
