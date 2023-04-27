#!/usr/bin/env python3
from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name="light_curve_analysis",
    ext_modules=cythonize(
        "scripts/analysis_tools_cython.pyx", compiler_directives={"language_level": 3}
    ),
    include_dirs=[numpy.get_include()],
)
