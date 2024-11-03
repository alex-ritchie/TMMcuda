# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='TMMcuda',
    ext_modules=[
        CUDAExtension(
            'TMMcuda',
            ['tmm.cpp', 'tmm.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

