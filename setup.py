#!/usr/bin/env python3
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


setup(
    name="exocomet_search",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(
        "scripts/analysis_tools_cython.pyx", compiler_directives={"language_level": 3}
    ),
    include_dirs=[numpy.get_include()],
    package_dir={'': '.'},
)
