# coding=utf-8
# python3

"""
python compile_to_c.py build_ext

"""

from distutils.core import setup
from Cython.Build import cythonize

# 调用setup方法，将Cython.Build.cythonize返回的结果传进去
setup(ext_modules=cythonize([
    "timekeeping.py"
]))
