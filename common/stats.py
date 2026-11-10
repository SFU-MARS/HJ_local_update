# Comparison of the decomposition and local update results with the direct computation.
import numpy as np


def compareArrays(array1, array2, number_of_precision_points=8, debug=False):
    '''
        Print the number of entries where array1 and array2 differ by more than 1e-1, 1e-2, ...
    '''
    diff_array = abs(array1 - array2)
    if debug:
        print(f"array1: \n{array1}")
        print(f"array2: \n{array2}")
        print(f"diff_array: \n{diff_array}")
    for precision in range(1, number_of_precision_points):
        precision_value = 1 / (10**precision)
        number_of_entires_with_error = len(np.argwhere(diff_array > precision_value))
        print(f"{precision_value}: {number_of_entires_with_error}")


def printCorrectnessStatistics(result_true, result_decomp, result_local):
    '''
        Used by the 3d local updates. Arrays have time as the last axis, shape (..., t).
    '''
    print("Processing stats")
    print("================================================")

    '''
    Comparision with direct computation at the final time step
    '''
    true_final = result_true[..., -1]
    decomp_final = result_decomp[..., -1]
    local_final = result_local[..., -1]

    total_num_of_points = true_final.size
    print(f"The total number of points: {total_num_of_points}")
    print(f"Decomposition method: Number of points with different values (1e-6): {np.argwhere(abs(decomp_final - true_final) > 1e-6).shape}")
    print(f"After local update, Number of points with different values (1e-6): {np.argwhere(abs(local_final - true_final) > 1e-6).shape}")

    print(f"The average absolute error between direct computation and decomposition: {np.sum(abs(decomp_final - true_final)) / total_num_of_points:0.10f}")
    print(f"The average absolute error between direct computation and local updates: {np.sum(abs(local_final - true_final)) / total_num_of_points:0.10f}")
    print(f"The maximum absolute error between direct computation and decomposition: {np.max(abs(decomp_final - true_final)):0.10f}")
    print(f"The maximum absolute error between direct computation and local updates: {np.max(abs(local_final - true_final)):0.10f}")

    # Zero level set comparison
    print(f"Points with different sign (set mismatch), decomposition: {np.sum((decomp_final < 0) != (true_final < 0))}")
    print(f"Points with different sign (set mismatch), local update: {np.sum((local_final < 0) != (true_final < 0))}")

    print("Number of values with error b/w direct computation and decomposition:")
    compareArrays(true_final, decomp_final)
    print("Number of values with error b/w direct computation and local update:")
    compareArrays(true_final, local_final)
