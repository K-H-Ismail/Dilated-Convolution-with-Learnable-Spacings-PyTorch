from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dcls_full_cpp',
      ext_modules=[cpp_extension.CppExtension('dcls_full_cpp', ['src/dcls_full.cpp','src/cuda/dcls_full_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

