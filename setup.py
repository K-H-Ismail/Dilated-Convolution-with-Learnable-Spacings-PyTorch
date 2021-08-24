'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import sys
import os
import glob
import setuptools
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension


requirements = ["torch"]

def get_extensions():
    if CUDA_HOME is None:
        print('CUDA_HOME is None. Install Without CUDA Extension')
        return None
    else:
        print('Install With CUDA Extension')
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'DCLS/src')

    ext_list = ['dcls_1d',
                'dcls_2_1d',                
                'dcls_2d',
                'dcls_3_1d',
                'dcls_3_2d',                
                'dcls_3d',                
                'dcls',
                'im2col_dcls']    
    if not sys.platform == 'win32':
        # win32 does not support cuSparse
        ext_list.extend(['spmm', 
                         'sparse_weight_conv'])
    
    extra_compile_args = {'cxx': ['-g'], 'nvcc': ['-use_fast_math']}
    
    extension = CUDAExtension
    define_macros = [("WITH_CUDA", None)]
    ext_modules = list([
        extension(
            ext_name,
            glob.glob(os.path.join(extensions_dir, ext_name + '.cpp')) + glob.glob(os.path.join(extensions_dir, 'cuda', ext_name + '_cuda_kernel.cu')),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=[ 'cusparse']            
        ) for ext_name in ext_list])


    return ext_modules

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=requirements,
    name="DCLS",
    version="0.0.1",
    author="Ismail Khalfaoui Hassani",
    author_email="ismail.khalfaoui-hassani@univ-tlse3.fr",
    description="Dilated convolution with learnable spacings, built on PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension
    }
)




