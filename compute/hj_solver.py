# optimized_dp's HJSolver as used by every direct computation and decomposition in compute/.
import contextlib
import io

import numpy as np
from odp.solver import HJSolver


def solve_hj(sys, grid, data, tau, accuracy="low"):
    '''
        Solve the HJ PDE of sys on grid from the initial value data over the times tau with HJSolver
        (TargetSetMode "None", saving every time step).
            accuracy: "low" (1st order ENO + forward Euler) matches the local updates
        Returns the value function with time as the last axis, index 0 is t = 0, as 32-bit floats
        (HJSolver computes in 32-bit floats, so the conversion is exact).

        The local updates take exactly one forward Euler step per time step, so HJSolver must too.
        HJSolver splits a time step when it is larger than its CFL bound, so this stops with an error
        if HJSolver took more steps than len(tau) - 1: use a smaller t_step.
    '''
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = HJSolver(sys, grid, data, tau, {"TargetSetMode": "None"},
                          saveAllTimeSteps=True, accuracy=accuracy)
    log = out.getvalue().splitlines()
    steps = sum(1 for line in log if line.startswith("Computational time to integrate"))
    for line in log:
        if line.startswith("Total kernel time"):
            print(f"HJSolver {grid.dims}D: {line}")
    if steps != len(tau) - 1:
        raise ValueError(f"HJSolver took {steps} steps for {len(tau) - 1} time steps, because the time step "
                         f"{tau[1] - tau[0]:g} is larger than its CFL bound. Use a smaller --t_step.")
    # HJSolver stores the initial value at index -1, flip so that index 0 is t = 0
    return np.flip(result, axis=-1).astype(np.float32)
