# Cache of compiled HeteroCL kernels, so a kernel built once with the same settings is loaded instead of rebuilt.
import hashlib
import os

from heterocl.tvm import module as hcl_module


def load_or_build_kernel(build, key, name, cache_dir="kernels"):
    '''
        Load the compiled kernel from cache_dir if it was built before with the same key, otherwise
        call build() and save the kernel there.
            build: function with no arguments that returns the compiled kernel
            key: tuple of everything the kernel depends on (grid, settings, dynamics, source code of
                 the kernel), so any change builds a new kernel
            name: start of the file name, e.g. "local_update_2d"
        Returns (kernel, "loaded" or "built").
    '''
    path = os.path.join(cache_dir, f"{name}_{hashlib.md5(repr(key).encode()).hexdigest()[:12]}.so")
    if os.path.isfile(path):
        return hcl_module.load(path), "loaded"
    kernel = build()
    os.makedirs(cache_dir, exist_ok=True)
    kernel.export_library(path)
    return kernel, "built"
