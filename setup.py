#!/usr/bin/env python
from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="pyms",
    version="1.0",
    rust_extensions=[RustExtension("pyms.native", binding=Binding.PyO3)],
    packages=["pyms"],
    zip_safe=False,
)
