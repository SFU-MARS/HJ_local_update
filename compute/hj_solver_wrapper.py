# A thin wrapper around optimized_dp's HJSolver, used by every direct computation and decomposition
# in compute/.
#
# Why not call HJSolver directly?
# -------------------------------
# solve_hj still calls HJSolver(..., saveAllTimeSteps=True). It only adds what is needed so the
# results can be compared fairly with the local updates:
#
#   1. Same time steps as the local update (the important one).
#      If the time step is larger than HJSolver's CFL bound, HJSolver quietly splits it into
#      smaller steps. The local update always takes exactly one step per time step, so the two
#      would no longer compute the same thing, and they would disagree even where the local
#      update is right. solve_hj counts HJSolver's steps and stops with a clear error
#      ("Use a smaller --t_step") instead of saving a result that cannot be compared.
#
#   2. Time goes forward: index 0 is t = 0.
#      HJSolver puts t = 0 at the last index. solve_hj flips it, which is the order every saved
#      file and the local update use. One flip here, instead of one in every script.
#
#   3. 32-bit floats.
#      HJSolver computes in 32-bit floats, so saving them as such is exact. It halves the file
#      size (a 21^6 result is 3.8 GB) and matches the precision of the local update.
#
#   4. Short logs.
#      HJSolver prints a line for every step. solve_hj keeps only the total kernel time.
#
#   5. Same settings everywhere.
#      TargetSetMode "None", and accuracy "low" by default: the scheme the local update uses
#      (1st order ENO + forward Euler). Set once here, so the scripts cannot drift apart.

import contextlib
import io

import numpy as np
from odp.solver import HJSolver


def solve_hj(sys, grid, data, tau, accuracy="low", target_set_mode="None"):
    '''
        Solve the HJ PDE of sys on grid from the initial value data over the times tau with HJSolver
        (TargetSetMode "None", saving every time step).
            accuracy: "low" (1st order ENO + forward Euler) matches the local updates
            target_set_mode: "None" (default) gives the backward reachable set; "minVWithV0" the tube, for the
                             local updates with brt=True
        Returns the value function with time as the last axis, index 0 is t = 0, as 32-bit floats
        (HJSolver computes in 32-bit floats, so the conversion is exact).

        The local updates take exactly one forward Euler step per time step, so HJSolver must too.
        HJSolver splits a time step when it is larger than its CFL bound, so this stops with an error
        if HJSolver took more steps than len(tau) - 1: use a smaller t_step.
    '''
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = HJSolver(sys, grid, data, tau, {"TargetSetMode": target_set_mode},
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
