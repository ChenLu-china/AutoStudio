"""
This file is part of autostudio
Copyright (C) 
@file setup.py
@author LuChen, The Nullmax AI; ZhenJun ZHao, The Chinese University of Hong Kong
@brief 
"""


import os
import sys
import logging
import subprocess
from copy import deepcopy
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(SCRIPT_DIR, 'VERSION'),"r") as f: VERSION = f.read()

if torch.cuda.is_available():
    if os.name == "nt":
        def find_cl_path():
            import glob
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(glob.glob(
                    r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition),
                               reverse=True)
                if paths:
                    return paths[0]


        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ["PATH"] += ";" + cl_path
    
    # Some containers set this to contain old
    os.environ['TORCH_CUDA_ARCH_LIST'] = ""

    common_libary_dirs = []
    if '--fix-lcuda' in sys.argv:
        sys.argv.remove('--fix-lcuda')
        common_libary_dirs.append(os.path.join(os.environ.get('CUDA_HOME'), 'lib64', 'stubs'))
    
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor

    def get_cuda_bare_metal_version():
        raw_output = subprocess.check_output([os.path.join(CUDA_HOME, 'bin', 'nvcc'), "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]

        return raw_output, bare_metal_major, bare_metal_minor
else:
    raise EnvironmentError(
    "PyTorch CUDA is unavailable. nr3d_lib requires PyTorch to be installed with the CUDA backend.")


def get_ext_geometry():
    # Modified from https://github.com/ashawkey/torch-ngp
    nvcc_flags = [
        '-O3', '-std=c++14',
        '-D__CUDA_NO_HALF_OPERATORS__', 
        '-D__CUDA_NO_HALF_CONVERSIONS__', 
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
        f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
    ]

    if os.name == "posix":
        c_flags = ["-std=c++14"]
        nvcc_flags += [
            "-Xcompiler=-mf16c",
            "-Xcompiler=-Wno-float-conversion",
            "-Xcompiler=-fno-strict-aliasing",
			"-Xcudafe=--diag_suppress=unrecognized_gcc_pragma",
        ]
    elif os.name == "nt":
        c_flags = ["/std:c++14"]

    library_dirs = deepcopy(common_libary_dirs)

    scr_dir = os.path.abspath(os.path.join(SCRIPT_DIR, 'src'))
    ext = CUDAExtension(
        name='auto_studio_lib.geometry',
        sources=[os.path.join(scr_dir, 'geometry', f) for f in [
          'Geometry.cpp',
        ]],
        include_dirs=[
            os.path.join(SCRIPT_DIR, "External", "yaml-cpp", "include"),
        ],
        extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            }, 
        library_dirs=library_dirs
    )
    return ext


def get_extensions():
    ext_modules = []
    ext_modules.append(get_ext_geometry())
    return ext_modules


setup(
    name="auto_studio",
    version=VERSION,
    description="auto_studio",
    keywords="auto_studio",
    author="LuChen, Zhenjun Zhao",
    author_email="chen-lu.chen@connect.polyu.hk, ",
    download_url="https://github.com/ChenLu-china/AutoStudio",
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension}
)