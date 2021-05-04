from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dcls_cpp',
      ext_modules=[cpp_extension.CppExtension('dcls_cpp', ['src/dcls.cpp','src/cuda/dcls_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='dcls_full_cpp',
      ext_modules=[cpp_extension.CppExtension('dcls_full_cpp', ['src/dcls_full.cpp','src/cuda/dcls_full_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='sparse_weight_conv_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_weight_conv_cpp', ['src/sparse_weight_conv.cpp','src/cuda/sparse_weight_conv_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='dcls_1d_cpp',
      ext_modules=[cpp_extension.CppExtension('dcls_1d_cpp', ['src/dcls_1d.cpp','src/cuda/dcls_1d_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

