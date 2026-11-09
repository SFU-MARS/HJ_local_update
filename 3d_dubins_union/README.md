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

Then run the local update:

```bash
python local_update_numpy.py --decomp_file results/result_decomp_3d.npy --true_file results/result_direct_3d.npy --t_step 0.04
```

Run all commands inside `3d_dubins_union`.

#### Code Structure

    local_update_numpy.py
    ├── update_V_numpy.py
    └── system.py

    direct.py
    decomposition.py
    └── subsystem.py
