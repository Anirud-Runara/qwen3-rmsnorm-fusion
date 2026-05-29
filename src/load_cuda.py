"""Load the CUDA denominator extension, building via JIT if needed."""
import os
import torch.utils.cpp_extension as ext

# Bypass CUDA version check (13.1 driver is forward-compatible with 12.8 toolkit)
_orig_check = ext._check_cuda_version
ext._check_cuda_version = lambda *a, **k: None

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

denominator_cuda = ext.load(
    name="denominator_cuda",
    sources=[
        os.path.join(_ROOT, "csrc", "denominator.cpp"),
        os.path.join(_ROOT, "csrc", "denominator_kernel.cu"),
    ],
    extra_cuda_cflags=["-arch=sm_120", "-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)

ext._check_cuda_version = _orig_check
