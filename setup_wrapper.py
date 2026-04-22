"""
Cython build script for wrapper.py.
Compiles wrapper.py → wrapper.cpython-3xx-*.so
Run: python setup_wrapper.py build_ext --inplace
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="wrapper",
    ext_modules=cythonize(
        ["wrapper.py"],
        compiler_directives={"language_level": 3},
    ),
)
