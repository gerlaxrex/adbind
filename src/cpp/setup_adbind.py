from setuptools import setup, Extension
import pybind11

extra_compile_args = []
extra_link_args = []

extra_compile_args = ['-std=c++14']
ext_modules = [
    Extension(
        'adbind',
        ['Variable.cpp', 'bindings.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='adbind',
    version='0.1.0',
    author='Jerlax',
    author_email='re.gerlando@gmail.com',
    description='adbind: very simple reverse-mode autodiff with c++ bindings',
    ext_modules=ext_modules
)