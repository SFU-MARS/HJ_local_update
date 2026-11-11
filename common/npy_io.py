# Fast reading of the large .npy files (several GB at 21^6).
import numpy as np


def read_npy_into(path, out):
    '''
        Read the .npy file path into the existing C-contiguous array out (same shape and type), straight from the
        file into out's memory: np.load would allocate a new array first and need one more copy.
    '''
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        if fortran_order or tuple(shape) != out.shape or np.dtype(dtype) != out.dtype or not out.flags.c_contiguous:
            raise ValueError(f"{path}: {shape} {dtype} (fortran_order {fortran_order}) does not fit the "
                             f"array {out.shape} {out.dtype}")
        view = memoryview(out.reshape(-1).view(np.uint8))
        done = 0
        while done < len(view):
            n = f.readinto(view[done:])
            if not n:
                raise ValueError(f"{path}: file is shorter than its header says")
            done += n
