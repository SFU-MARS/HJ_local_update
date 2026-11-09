# HJ_local_update

### Prerequisite

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp)

### Running Local Update

First generate the direct and decomposition results (both default to `accuracy='low'` and `t_step=0.04` to match the local update scheme):

```bash
python -c "
from direct import direct_comp
from decomposition import decomposition
direct_comp(41, plot=False)
decomposition(41, plot=False)
"
```

Then run the local update, numpy version:

```bash
python local_update_numpy.py --decomp_file results/result_decomp_3d.npy --true_file results/result_direct_3d.npy --t_step 0.04
```

or the parallel HeteroCL version:

```bash
python local_update_hcl.py --decomp_file results/result_decomp_3d.npy --true_file results/result_direct_3d.npy --t_step 0.04
```

The HeteroCL version saves the compiled kernel in `kernels/` on the first run and loads it on later runs.

Run all commands inside `3d_dubins_union`.

### Computation Time

41 x 41 x 41 grid, 25 time steps (`t_step=0.04`, `accuracy='low'`), 32 CPU threads.

| Method | Time-stepping loop (s) | Total (s) | Points with error > 1e-5 |
|---|---|---|---|
| Direct computation (`direct.py`, HJSolver) | 0.046 | 0.177 | — |
| Local update, HeteroCL (`local_update_hcl.py`), first run | 0.039 | 0.208 | 0 |
| Local update, HeteroCL (`local_update_hcl.py`), cached kernel | 0.008 | 0.025 | 0 |
| Local update, numpy (`local_update_numpy.py`) | 11.4 | 11.4 | 4 |

- Time-stepping loop: HJSolver "Total kernel time" for direct computation, "Total correction time" for the local update.
- Total: the full HJSolver call for direct computation (kernel build + setup + loop); kernel build/load + input copy + loop for the HeteroCL local update.
- The decomposition result is read from file, so its computation time is not included.
- Error is measured against the direct computation at the final time step (the decomposition alone has 51323 points with error > 1e-5).

#### Code Structure

    local_update_numpy.py
    ├── update_V_numpy.py
    └── system.py

    local_update_hcl.py
    └── local_update_numpy.py

    direct.py
    decomposition.py
    └── subsystem.py
