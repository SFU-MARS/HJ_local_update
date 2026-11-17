# Comparison of the decomposition and local update results with the direct computation.
import numpy as np


def print_error_counts(error, number_of_precision_points=8):
    '''
        Print the number of entries of the absolute error array error larger than 1e-1, 1e-2, ...
    '''
    for precision in range(1, number_of_precision_points):
        precision_value = 1 / (10**precision)
        print(f"{precision_value}: {np.count_nonzero(error > precision_value)}")


def away_from_edges(shape, periodic_dims=(), width=2):
    '''
        Boolean mask of the points more than width - 1 points away from every edge of the non-periodic dimensions
        (where HJSolver extrapolates past the grid).
    '''
    mask = np.ones(shape, dtype=bool)
    for dim, n in enumerate(shape):
        if dim in periodic_dims or n <= 2 * width:
            continue
        index = [slice(None)] * len(shape)
        index[dim] = list(range(width)) + list(range(n - width, n))
        mask[tuple(index)] = False
    return mask


def printCorrectnessStatistics(true_final, decomp_final, local_final, periodic_dims=()):
    '''
        Used by every local update (common/local_update_setup.py). Compares the decomposition and the local
        update with the direct computation at the final time step; the arguments are the value functions at
        that time, all with the grid shape.
            periodic_dims: periodic dimensions of the grid, which have no edge
    '''
    print("Processing stats")
    print("================================================")
    true_final, decomp_final, local_final = (np.asarray(a) for a in (true_final, decomp_final, local_final))
    error_decomp = abs(decomp_final - true_final)
    error_local = abs(local_final - true_final)

    total_num_of_points = true_final.size
    print(f"The total number of points: {total_num_of_points}")
    print(f"Decomposition method: Number of points with different values (1e-6): {np.count_nonzero(error_decomp > 1e-6)}")
    print(f"After local update, Number of points with different values (1e-6): {np.count_nonzero(error_local > 1e-6)}")
    # The same, without the 2 points next to each grid edge, where HJSolver extrapolates and 32-bit rounding grows
    inner = away_from_edges(true_final.shape, periodic_dims)
    print(f"Away from the grid edges ({np.count_nonzero(inner)} points, 2 points from every non-periodic edge left out): "
          f"decomposition {np.count_nonzero(error_decomp[inner] > 1e-6)}, local update {np.count_nonzero(error_local[inner] > 1e-6)}")

    print(f"The average absolute error between direct computation and decomposition: {error_decomp.mean(dtype=np.float64):0.10f}")
    print(f"The average absolute error between direct computation and local updates: {error_local.mean(dtype=np.float64):0.10f}")
    print(f"The maximum absolute error between direct computation and decomposition: {error_decomp.max():0.10f}")
    print(f"The maximum absolute error between direct computation and local updates: {error_local.max():0.10f}")

    # Zero level set comparison
    true_inside = true_final < 0
    print(f"Points with different sign (set mismatch), decomposition: {np.count_nonzero((decomp_final < 0) != true_inside)}")
    print(f"Points with different sign (set mismatch), local update: {np.count_nonzero((local_final < 0) != true_inside)}")

    print("Number of values with error b/w direct computation and decomposition:")
    print_error_counts(error_decomp)
    print("Number of values with error b/w direct computation and local update:")
    print_error_counts(error_local)
