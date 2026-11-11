# Local update of the 6D quadrotor decomposition result, numpy version.
# Growing-region update of numpy_impl/growing_region.py, in 32-bit floats like HJSolver: corrects the
# decomposition result near the leaking corners so it matches the direct computation (compute/direct_6d.py).
# Settings of the case: CASE_6D in compute/config_6d.py.
import argparse

from compute.config_6d import CASE_6D
from common.local_update_setup import save_and_compare, setup_local_update
from numpy_impl.growing_region import local_update_from_file


def main():
    '''
        Read the decomposition (compute/decomposition_6d.py), run the local update, save it to
        results/result_local_6d_<combine>.npy and, if --true_file is given, compare it with the direct computation.
    '''
    # Steps 1-3: arguments (--params file and command line), dynamics, grid and initial leaking set
    args, sys, grid, leaking = setup_local_update(argparse.ArgumentParser(), CASE_6D)

    # Step 4: run the local update
    result_local = local_update_from_file(args.decomp_file, leaking, grid, sys, args.t_step,
                                          grow_within_step=args.grow_within_step)

    # Step 5: save the result and compare it with the direct computation
    save_and_compare(args, result_local, "6d")


if __name__ == "__main__":
    main()
