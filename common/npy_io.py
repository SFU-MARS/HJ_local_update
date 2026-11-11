# Fast conversion between the time-last .npy files (results/, shape (..., t)) and time-first arrays (t, ...) used
# by the HeteroCL local updates. The files are several GB at 21^6; the reordering is done in blocks of points, on
# several threads (numpy releases the GIL while copying).
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BLOCK = 1 << 20      # points per block
THREADS = min(16, os.cpu_count() or 1)


def _blocks(n):
    return [(a, min(a + BLOCK, n)) for a in range(0, n, BLOCK)]


def read_time_first(path, out):
    '''
        Read the time-last .npy file path, shape (..., t), into the C-contiguous time-first array out, shape (t, ...).
    '''
    data = np.load(path, mmap_mode='r')
    num_steps = data.shape[-1]
    assert out.shape == (num_steps,) + data.shape[:-1] and out.flags.c_contiguous
    src = data.reshape(-1, num_steps)
    dst = out.reshape(num_steps, -1)

    def copy(block):
        a, b = block
        dst[:, a:b] = src[a:b].T
    with ThreadPoolExecutor(THREADS) as pool:
        list(pool.map(copy, _blocks(src.shape[0])))


def save_npy(path, arr):
    '''
        np.save(path, arr), also fast when arr is a time-first array viewed with time as the last axis
        (np.moveaxis(result, 0, -1)): np.save writes such a view element by element (15 s for 3.4 GB).
    '''
    time_first = np.moveaxis(arr, -1, 0)
    if arr.flags.c_contiguous or not time_first.flags.c_contiguous:
        np.save(path, arr)
        return
    num_steps = arr.shape[-1]
    src = time_first.reshape(num_steps, -1)
    blocks = _blocks(src.shape[1])
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(arr.dtype),
                                                 "fortran_order": False, "shape": arr.shape})
        # Reorder the blocks on several threads, write them in order
        with ThreadPoolExecutor(THREADS) as pool:
            for block in pool.map(lambda ab: np.ascontiguousarray(src[:, ab[0]:ab[1]].T), blocks):
                f.write(block.data)
