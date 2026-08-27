import os

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="MonoDGPDeterministicMSDA",
    version="1.0",
    ext_modules=[
        CUDAExtension(
            name="MonoDGPDeterministicMSDA",
            sources=["msda_deterministic_backward.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
)
