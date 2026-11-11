# Local update of the 3D Dubins car decomposition result, parallel HeteroCL version.
# Growing-region update of hcl_impl/growing_region.py (same as numpy_impl/local_update_3d.py), in 32-bit floats
# like HJSolver; the compiled kernel is cached in kernels/ to skip rebuilding.
# Settings of the case: CASE_3D in compute/config_3d.py.
import argparse

from compute.config_3d import CASE_3D
from common.local_update_setup import save_and_compare, setup_local_update
from hcl_impl.growing_region import local_update_hcl_from_file


def main():
    '''
        Read the decomposition (compute/decomposition_3d.py), run the HeteroCL local update, save it to
        results/result_local_hcl_3d_<combine>.npy and, if --true_file is given, compare it with the direct computation.
    '''
    # Steps 1-3: arguments (--params file and command line), dynamics, grid and initial leaking set
    args, sys, grid, leaking = setup_local_update(argparse.ArgumentParser(), CASE_3D, hcl=True)

    # Step 4: run the local update
    result_local = local_update_hcl_from_file(args.decomp_file, leaking, grid, sys, args.t_step,
                                              cache_dir=args.cache_dir, grow_within_step=args.grow_within_step)

    # Step 5: save the result and compare it with the direct computation
    save_and_compare(args, result_local, "3d", hcl=True)


if __name__ == "__main__":
    main()
