from Cython.Build import cythonize
import sys
import numpy as np
from distutils.core import setup
from distutils.extension import Extension

try:
    numpy_include = np.get_include()
    print(np.get_include())
except AttributeError:
    numpy_include = np.get_numpy_include()
ext_modules = [
    Extension(
        'bbox',
        sources=['bbox.c'],
        include_dirs=[numpy_include]
    ),
    Extension(
        'cython_nms',
        include_dirs=[numpy_include],
        sources=['cython_nms.c'],
    )
]
setup(
    ext_modules=ext_modules
)