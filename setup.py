import numpy
from setuptools import setup, Extension

module_hist = Extension('hist',
                    sources = ['src/hist.cpp'],
                    language='c++',
                    include_dirs=[numpy.get_include()],
                    extra_compile_args=[],
                    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])

setup (name = 'hist',
       version = '0.1.0',
       description = 'Computes histogram of a double array',
       install_requires=['numpy'],
       ext_modules = [module_hist],
       zip_safe=False)
