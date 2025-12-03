# HJ_local_update

### Prerequisite 

- Install [Optimized_dp](https://github.com/SFU-MARS/optimized_dp) 

### Running Local Update

For Union Cases, use the following command:

```bash 
python local_update_numpy.py --grid_size 101 --use_union 1
```
For Intersection Cases, use the following command:
```bash 
python local_update_numpy.py --grid_size 101 --use_union 0 
```

Use the `--grid_size` command-line argument to change the grid size. 

#### Code Structure

    local_update_numpy.py
    ├── direct_numpy.py
    ├── decomposition_numpy.py
    │    ├── update_V_numpy.py
    │    └── subsystem.py 
    ├── config.py    
    ├── update_V_numpy.py
    ├── system.py
    └── set_2plot.py

