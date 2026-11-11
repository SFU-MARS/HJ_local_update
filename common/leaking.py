# Leaking set: points where the decomposition can differ from the direct computation.
#   dilate, get_leaking_set: every local update (numpy and HeteroCL)
#   ground_truth_L, new_theory_L, ground_truth_delta: analysis helpers, not used by the scripts
import numpy as np


def dilate(mask, grid):
    '''
        Add the neighbours of every True point along each dimension (the stencil of the
        first order scheme). Periodic dimensions wrap around.
    '''
    out = mask.copy()
    for dim in range(mask.ndim):
        if dim in grid.pDim:
            out |= np.roll(mask, 1, axis=dim) | np.roll(mask, -1, axis=dim)
        else:
            lo = [slice(None)] * mask.ndim
            hi = [slice(None)] * mask.ndim
            lo[dim] = slice(None, -1)
            hi[dim] = slice(1, None)
            out[tuple(hi)] |= mask[tuple(lo)]
            out[tuple(lo)] |= mask[tuple(hi)]
    return out


def get_leaking_set(data_init_upper, data_init_lower, result_decomp, grid, mode="smallest"):
    '''
        Initial leaking set: points where the decomposition min(V1, V2) can be wrong.
            mode="smallest" (default): the switching surface V1 = V2 plus its neighbours,
                dilate(|V1 - V2| < 1e-9). Smallest set that gives the full correction, no tuning.
            mode="delta": |V1 - V2| < delta, with delta = max|initial value - final
                decomposition result| (new_theory_L below). Conservative, larger set.
            result_decomp: decomposition result with time as the first axis (only used by mode="delta")
    '''
    if mode == "smallest":
        leaking = dilate(abs(data_init_upper - data_init_lower) < 1e-9, grid)
    elif mode == "delta":
        delta = np.max(abs(result_decomp[0] - result_decomp[-1]))
        print('The delta value is', delta)
        leaking = abs(data_init_upper - data_init_lower) < delta
    else:
        raise ValueError(f"unknown leaking mode: {mode}")
    print(f"The number of leaking corner points: {leaking.sum()} / {leaking.size}")
    return leaking


def ground_truth_L(data_true, data_approx):
    '''
    Get the ground truth of the leaking corners
    '''
    result_diff = data_approx - data_true
    indice = np.argwhere(abs(result_diff) > 1e-6)
    return indice
    
def new_theory_L(data_approx, data_init, data_upper, data_lower):
    '''
    Test the new theory of finding the leaking corners
    '''
    # delta_1 = np.max(abs(data_upper[0] - data_upper[-1]))
    # delta_2 = np.max(abs(data_lower[0] - data_lower[-1]))
    # print('The delta 1 value is', delta_1)
    # print('The delta 2 value is', delta_2)
    # delta = max(delta_1,delta_2)

    delta = np.max(abs(data_init - data_approx))
    print('The delta value is', delta)
    
    indice = np.argwhere(abs(data_upper[0] - data_lower[0]) < delta)
    
    return indice

def ground_truth_delta(data_true, data_approx, data_upper, data_lower):
    
    indices = np.argwhere(abs(data_true - data_approx) > 1e-3)
    print("The number of leaking corners is", len(indices))
    delta = np.max(data_upper[indices] - data_lower[indices])
    
    print("The ground truth delta value is", delta)  
    
    return delta