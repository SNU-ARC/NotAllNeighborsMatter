import glob
import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

from torchsparse import __version__
from sys import argv

def _argparse(pattern, argv, is_flag=True, is_list=False):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern == arg.split("=")[0]]
        if is_list:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                val = arr[0].split("=")[1]
                if "," in val:
                    return val.split(","), argv
                else:
                    return [val], argv
        else:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                return arr[0].split("=")[1], argv



if ((torch.cuda.is_available() and CUDA_HOME is not None)
        or (os.getenv('FORCE_CUDA', '0') == '1')):
    device = 'cuda'
else:
    device = 'cpu'

sources = [os.path.join('torchsparse', 'backend', f'pybind_{device}.cpp')]
for fpath in glob.glob(os.path.join('torchsparse', 'backend', '**', '*')):
    if ((fpath.endswith('_cpu.cpp') and device in ['cpu', 'cuda'])
            or (fpath.endswith('_cuda.cu') and device == 'cuda')):
        sources.append(fpath)

extension_type = CUDAExtension if device == 'cuda' else CppExtension

extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': ['-O3']
}

setup(
    name='torchsparse',
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type('torchsparse.backend',
                       sources,
                       extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
