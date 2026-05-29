from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="denominator_cuda",
    ext_modules=[
        CUDAExtension(
            name="denominator_cuda",
            sources=[
                "csrc/denominator.cpp",
                "csrc/denominator_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-arch=sm_90", "-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
